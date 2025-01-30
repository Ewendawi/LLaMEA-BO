from typing import Callable
from scipy.stats import qmc
import numpy as np
import torch
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, PeriodicKernel
from gpytorch.likelihoods import GaussianLikelihood
from scipy.optimize import minimize

class EnsembleAdaptiveTrustRegionBOv1:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
        self.trust_region_radius = 2.0
        self.best_x = None
        self.best_y = float('inf')
        self.dtype = torch.float32
        self.iteration = 0
        self.kernel_types = ['matern', 'rbf', 'periodic']
        self.acquisition_type = 'ei'
        self.exploration_rate = 0.5
        self.beta = 2.0
        self.num_multistarts = 5
        self.models = {}
        self.model_scores = {k: 0 for k in self.kernel_types}
        self.model_weights = {k: 1/len(self.kernel_types) for k in self.kernel_types}


    def _sample_points(self, n_points: int) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X: np.ndarray, y: np.ndarray, kernel_type: str) -> gpytorch.models.ExactGP:
        X_tensor = torch.tensor(X, dtype=self.dtype, device=self.device)
        y_tensor = torch.tensor(y, dtype=self.dtype, device=self.device).squeeze()
        
        likelihood = GaussianLikelihood().to(self.device, dtype=self.dtype)
        if kernel_type == 'rbf':
            kernel = ScaleKernel(RBFKernel()).to(self.device, dtype=self.dtype)
        elif kernel_type == 'periodic':
            kernel = ScaleKernel(PeriodicKernel()).to(self.device, dtype=self.dtype)
        else:
            kernel = ScaleKernel(MaternKernel(nu=2.5)).to(self.device, dtype=self.dtype)

        class GPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = kernel

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        model = GPModel(X_tensor, y_tensor, likelihood).to(self.device, dtype=self.dtype)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = ExactMarginalLogLikelihood(likelihood, model)
        
        model.train()
        likelihood.train()
        
        for _ in range(50):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = -mll(output, y_tensor)
            loss.backward()
            optimizer.step()
        
        model.eval()
        likelihood.eval()
        return model
    
    def _expected_improvement(self, X: np.ndarray, model: gpytorch.models.ExactGP) -> np.ndarray:
        X_tensor = torch.tensor(X, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            posterior = model(X_tensor)
            mu = posterior.mean
            sigma = posterior.stddev
        
        if self.y is None or len(self.y) == 0:
          return np.zeros((X.shape[0], 1))

        best_y = torch.min(torch.tensor(self.y, dtype=self.dtype, device=self.device))
        imp = mu - best_y
        Z = imp / sigma
        ei = imp * torch.distributions.Normal(0,1).cdf(Z) + sigma * torch.distributions.Normal(0,1).log_prob(Z).exp()
        ei[sigma == 0] = 0
        return ei.cpu().numpy().reshape(-1, 1)
    
    def _probability_of_improvement(self, X: np.ndarray, model: gpytorch.models.ExactGP) -> np.ndarray:
        X_tensor = torch.tensor(X, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            posterior = model(X_tensor)
            mu = posterior.mean
            sigma = posterior.stddev
        
        if self.y is None or len(self.y) == 0:
          return np.zeros((X.shape[0], 1))

        best_y = torch.min(torch.tensor(self.y, dtype=self.dtype, device=self.device))
        imp = mu - best_y
        Z = imp / sigma
        pi = torch.distributions.Normal(0,1).cdf(Z)
        pi[sigma == 0] = 0
        return pi.cpu().numpy().reshape(-1, 1)

    def _ucb(self, X: np.ndarray, model: gpytorch.models.ExactGP) -> np.ndarray:
        X_tensor = torch.tensor(X, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            posterior = model(X_tensor)
            mu = posterior.mean
            sigma = posterior.stddev
        return (mu + self.beta * sigma).cpu().numpy().reshape(-1, 1)
    
    def _thompson_sampling(self, X: np.ndarray, model: gpytorch.models.ExactGP) -> np.ndarray:
        X_tensor = torch.tensor(X, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            posterior = model(X_tensor)
            samples = posterior.rsample()
        return samples.cpu().numpy().reshape(-1, 1)
    
    def _optimize_acquisition(self, models: dict, bounds: np.ndarray, n_starts: int) -> np.ndarray:
        def obj_func(x):
            x_tensor = torch.tensor(x.reshape(1,-1), dtype=self.dtype, device=self.device)
            acq_values = []
            for kernel_type, model in models.items():
              if np.random.rand() < self.exploration_rate:
                  acq = self._thompson_sampling(x_tensor.cpu().numpy(), model)
              elif self.acquisition_type == 'ei':
                  acq = self._expected_improvement(x_tensor.cpu().numpy(), model)
              elif self.acquisition_type == 'pi':
                  acq = self._probability_of_improvement(x_tensor.cpu().numpy(), model)
              else:
                  acq = self._ucb(x_tensor.cpu().numpy(), model)
              acq_values.append(acq.item() * self.model_weights[kernel_type])

            return -np.sum(acq_values)
        
        best_x = None
        best_acq_val = float('inf')
        
        for _ in range(n_starts):
            initial_x = np.random.uniform(bounds[0], bounds[1])
            result = minimize(obj_func, initial_x, bounds=list(zip(bounds[0], bounds[1])), method='L-BFGS-B', options={'maxiter': 20})
            if result.fun < best_acq_val:
                best_acq_val = result.fun
                best_x = result.x
        
        return best_x.reshape(1, -1)


    def _select_next_points(self, batch_size: int) -> np.ndarray:
        if self.best_x is None:
          center = np.mean(self.bounds, axis=0)
        else:
          center = self.best_x
        
        lower_bound = np.maximum(center - self.trust_region_radius, self.bounds[0])
        upper_bound = np.minimum(center + self.trust_region_radius, self.bounds[1])
        
        
        next_points = []
        for _ in range(batch_size):
            next_point = self._optimize_acquisition(self.models, np.stack([lower_bound, upper_bound]), self.num_multistarts)
            next_points.append(next_point)
            
        return np.concatenate(next_points, axis=0)
    
    def _update_model_scores(self):
        if len(self.X) < 2*self.dim:
            return
        
        for kernel_type, model in self.models.items():
            with torch.no_grad():
                X_tensor = torch.tensor(self.X, dtype=self.dtype, device=self.device)
                posterior = model(X_tensor)
                log_likelihood = posterior.log_prob(torch.tensor(self.y.squeeze(), dtype=self.dtype, device=self.device))
                self.model_scores[kernel_type] += log_likelihood.item()

        max_score = max(self.model_scores.values())
        min_score = min(self.model_scores.values())

        for kernel_type in self.kernel_types:
            if max_score - min_score == 0:
                self.model_weights[kernel_type] = 1/len(self.kernel_types)
            else:
                self.model_weights[kernel_type] = (self.model_scores[kernel_type] - min_score) / (max_score - min_score)
        
        total_weight = sum(self.model_weights.values())
        for kernel_type in self.kernel_types:
            self.model_weights[kernel_type] /= total_weight


    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        n_initial_points = min(2 * self.dim, self.budget // 5)
        self.X = self._sample_points(n_initial_points)
        self.y = np.array([func(x) for x in self.X]).reshape(-1, 1)
        self.best_y = np.min(self.y)
        self.best_x = self.X[np.argmin(self.y)]
        
        for kernel_type in self.kernel_types:
            self.models[kernel_type] = self._fit_model(self.X, self.y, kernel_type)

        rest_of_budget = self.budget - n_initial_points
        
        while rest_of_budget > 0:
            self.iteration += 1
            
            batch_size = min(rest_of_budget, max(1, int(np.sqrt(self.dim) * np.log(self.iteration) / 2)))
            next_points = self._select_next_points(batch_size)
            next_y = np.array([func(x) for x in next_points]).reshape(-1, 1)
            
            self.X = np.concatenate((self.X, next_points), axis=0)
            self.y = np.concatenate((self.y, next_y), axis=0)
            
            current_best_y = np.min(self.y)
            if current_best_y < self.best_y:
              self.best_y = current_best_y
              self.best_x = self.X[np.argmin(self.y)]
            
            for kernel_type in self.kernel_types:
                self.models[kernel_type] = self._fit_model(self.X, self.y, kernel_type)
            
            rest_of_budget -= batch_size
            
            if self.best_x is not None:
                self.trust_region_radius = min(2.0, 0.5 * np.linalg.norm(self.bounds[1]-self.bounds[0]) / (1.0 + np.sqrt(self.dim) * np.log(len(self.y))))
            
            if self.iteration < 5:
                self.acquisition_type = 'ei'
            elif self.iteration < 10:
                self.acquisition_type = 'pi'
            else:
                self.acquisition_type = 'ucb'
            
            self.beta = 2.0 + np.sqrt(np.log(self.iteration))
            self.exploration_rate = 0.5 * np.exp(-0.1 * self.iteration)
            self._update_model_scores()
            
        return self.best_y, self.best_x