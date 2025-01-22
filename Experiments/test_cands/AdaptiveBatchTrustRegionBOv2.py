from typing import Callable
from scipy.stats import qmc
import numpy as np
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler

class AdaptiveBatchTrustRegionBOv2:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trust_region_radius = 1.0
        self.prev_best_y = float('inf')
        self.lr = 0.1
        self.exploration_weight = 0.5 # Initial exploration weight

    def _sample_points(self, n_points) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        X = self.scaler_X.fit_transform(X)
        y = self.scaler_y.fit_transform(y)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1).to(self.device)

        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        model = ExactGPModel(X, y, likelihood).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        training_iterations = 50
        for i in range(training_iterations):
            optimizer.zero_grad()
            output = model(X)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        likelihood.eval()
        return model, likelihood

    def _acquisition_function(self, X, current_best_x):
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model(X_tensor)
            mu = posterior.mean
            sigma = posterior.variance.sqrt()

        best_y = torch.min(torch.tensor(self.scaler_y.transform(self.y), dtype=torch.float32).to(self.device))
        
        imp = mu - best_y
        z = imp / sigma
        ei = imp * torch.distributions.Normal(0, 1).cdf(z) + sigma * torch.distributions.Normal(0, 1).log_prob(z).exp()
        ei[sigma <= 0] = 0
        
        # Exploration bonus based on distance to the best point
        distances = np.linalg.norm(X - current_best_x, axis=1)
        exploration_bonus = np.exp(-distances / self.trust_region_radius)
        
        # Combine exploration and exploitation
        acq_values =  (1 - self.exploration_weight) * ei.cpu().numpy().reshape(-1, 1) + self.exploration_weight * exploration_bonus.reshape(-1, 1)
        return acq_values

    def _select_next_points(self, batch_size, current_best_x) -> np.ndarray:
        n_candidates = min(1000, 100 * self.dim)
        candidates = self._sample_points(n_candidates)
        
        # Trust region implementation
        distances = np.linalg.norm(candidates - current_best_x, axis=1)
        candidates = candidates[distances <= self.trust_region_radius]
        if len(candidates) == 0:
            candidates = self._sample_points(n_candidates)
        
        acq_values = self._acquisition_function(candidates, current_best_x)
        indices = np.argsort(-acq_values.flatten())[:batch_size]
        return candidates[indices]

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        n_initial_points = min(20, 2 * self.dim)
        self.X = self._sample_points(n_initial_points)
        self.y = np.array([[func(x)] for x in self.X])
        
        best_idx = np.argmin(self.y)
        best_x = self.X[best_idx]
        best_y = self.y[best_idx][0]

        rest_of_budget = self.budget - n_initial_points
        while rest_of_budget > 0:
            self.model, self.likelihood = self._fit_model(self.X, self.y)
            
            # Dynamic batch size adjustment based on model uncertainty
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                X_tensor = torch.tensor(self.scaler_X.transform(self.X), dtype=torch.float32).to(self.device)
                posterior = self.model(X_tensor)
                uncertainty = posterior.variance.sqrt().mean()
            
            batch_size = min(max(1, int(rest_of_budget* (1 - torch.tanh(uncertainty).cpu().numpy())/2.0 + 1)), rest_of_budget)

            next_points = self._select_next_points(batch_size, best_x)
            next_values = np.array([[func(x)] for x in next_points])

            self.X = np.vstack((self.X, next_points))
            self.y = np.vstack((self.y, next_values))
            
            current_best_idx = np.argmin(self.y)
            if self.y[current_best_idx][0] < best_y:
                best_y = self.y[current_best_idx][0]
                best_x = self.X[current_best_idx]

            # Adjust trust region radius, learning rate and exploration weight
            if best_y < self.prev_best_y:
                self.trust_region_radius *= 1.1
                self.lr *= 1.05
                self.exploration_weight *= 0.95
            else:
                self.trust_region_radius *= 0.9
                self.lr *= 0.95
                self.exploration_weight *= 1.05

            self.trust_region_radius = np.clip(self.trust_region_radius, 0.1, 2.0)
            self.lr = np.clip(self.lr, 0.001, 0.2)
            self.exploration_weight = np.clip(self.exploration_weight, 0.1, 0.9)
            self.prev_best_y = best_y

            rest_of_budget -= batch_size
        return best_y, best_x