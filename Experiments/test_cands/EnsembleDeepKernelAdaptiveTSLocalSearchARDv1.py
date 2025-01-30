from typing import Callable
from scipy.stats import qmc
import numpy as np
import torch
import gpytorch
from scipy.optimize import minimize

class EnsembleDeepKernelAdaptiveTSLocalSearchARDv1:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        self.initial_points_multiplier = 5
        self.min_batch_size = 1
        self.max_batch_size = 5
        self.exploration_weight = 0.5
        self.exploitation_weight = 0.5
        self.local_search_iterations = 3
        self.feature_dim = 32
        self.best_y = float('inf')
        self.previous_best_y = float('inf')
        self.func_evals = 0
        self.models = []
        self.likelihoods = []
        self.num_ensemble = 3  

    def _sample_points(self, n_points) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device).flatten()
        
        self.models = []
        self.likelihoods = []
        for _ in range(self.num_ensemble):
            
            class FeatureExtractor(torch.nn.Module):
                def __init__(self, input_dim, feature_dim):
                    super().__init__()
                    self.linear1 = torch.nn.Linear(input_dim, 64)
                    self.relu = torch.nn.ReLU()
                    self.linear2 = torch.nn.Linear(64, feature_dim)

                def forward(self, x):
                    x = self.relu(self.linear1(x))
                    x = self.linear2(x)
                    return x


            class DeepGPModel(gpytorch.models.ExactGP):
                def __init__(self, train_x, train_y, likelihood, feature_dim, dim, device):
                    super(DeepGPModel, self).__init__(train_x, train_y, likelihood)
                    self.mean_module = gpytorch.means.ConstantMean()
                    self.feature_extractor = FeatureExtractor(dim, feature_dim).to(device)
                    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim))
                    self.device = device

                def forward(self, x):
                    features = self.feature_extractor(x)
                    mean_x = self.mean_module(features)
                    covar_x = self.covar_module(features)
                    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            model = DeepGPModel(X, y, likelihood, self.feature_dim, self.dim, self.device).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            
            model.train()
            likelihood.train()
            
            training_iterations = 50
            for _ in range(training_iterations):
                optimizer.zero_grad()
                output = model(X)
                loss = -mll(output, y)
                loss.backward()
                optimizer.step()
            model.eval()
            likelihood.eval()
            self.models.append(model)
            self.likelihoods.append(likelihood)
        return self.models, self.likelihoods

    def _acquisition_function(self, X, models, likelihoods, best_y) -> np.ndarray:
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        acq_values = np.zeros((X.shape[0], 1))
        
        for model, likelihood in zip(models, likelihoods):
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                posterior = likelihood(model(X))
                mean = posterior.mean
                std = posterior.stddev
                thompson_samples = posterior.sample(sample_shape=torch.Size([10]))
                thompson_acq = thompson_samples.cpu().numpy().mean(axis=0).reshape(-1,1)
                
                improvement = (best_y - mean).clamp(min=0)
                z = improvement / std
                ei = improvement * torch.distributions.normal.Normal(0, 1).cdf(z) + std * torch.distributions.normal.Normal(0, 1).log_prob(z).exp()
                ei_acq = ei.cpu().numpy().reshape(-1, 1)
                
                uncertainty = std.cpu().numpy().reshape(-1, 1)
                
                acq_values += (self.exploration_weight * thompson_acq + self.exploitation_weight * ei_acq + 0.1 * uncertainty)
        
        return acq_values / len(models)
    

    def _select_next_points(self, models, likelihoods, best_y, batch_size) -> np.ndarray:
        n_candidates = 1000
        candidates = self._sample_points(n_candidates)
        acq_values = self._acquisition_function(candidates, models, likelihoods, best_y)
        indices = np.argsort(acq_values.flatten())[-batch_size:]
        return candidates[indices]
    
    def _local_search(self, func, x0):
        
        def obj_func(x):
            if self.func_evals >= self.budget:
              raise Exception("Budget Exceeded")
            self.func_evals += 1
            return func(x)
        
        bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
        
        result = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': self.local_search_iterations})
        return result.fun, result.x
    
    def _update_weights(self):
        if self.best_y < self.previous_best_y:
            improvement = (self.previous_best_y - self.best_y) / (abs(self.previous_best_y) + 1e-8)
            self.exploration_weight = max(0.1, self.exploration_weight * (1 - improvement))
            self.exploitation_weight = min(0.9, self.exploitation_weight * (1 + improvement))
            self.previous_best_y = self.best_y
        else:
            self.exploration_weight = min(0.9, self.exploration_weight * 1.05)
            self.exploitation_weight = max(0.1, self.exploitation_weight * 0.95)
            
        if self.models and self.likelihoods:
            avg_uncertainty = 0
            for model, likelihood in zip(self.models, self.likelihoods):
                with torch.no_grad():
                    X_tensor = torch.tensor(self.X, dtype=torch.float32).to(self.device)
                    posterior = likelihood(model(X_tensor))
                    std = posterior.stddev.cpu().numpy()
                    avg_uncertainty += np.mean(std)
            avg_uncertainty /= len(self.models)
            if avg_uncertainty < 0.1:
                self.exploration_weight = min(0.9, self.exploration_weight * 1.1)
                self.exploitation_weight = max(0.1, self.exploitation_weight * 0.9)
        self.exploration_weight = max(0.1, min(0.9, self.exploration_weight))
        self.exploitation_weight = max(0.1, min(0.9, self.exploitation_weight))


    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        n_initial_points = self.dim * self.initial_points_multiplier
        self.X = self._sample_points(n_initial_points)
        self.y = np.array([func(x) for x in self.X]).reshape(-1, 1)
        self.func_evals += n_initial_points
        
        best_idx = np.argmin(self.y)
        self.best_y = self.y[best_idx].item()
        best_x = self.X[best_idx]
        self.previous_best_y = self.best_y

        rest_of_budget = self.budget - n_initial_points
        while rest_of_budget > 0:
            models, likelihoods = self._fit_model(self.X, self.y)
            
            batch_size = min(self.max_batch_size, self.min_batch_size + int(rest_of_budget / self.budget * (self.max_batch_size - self.min_batch_size)))
            next_points = self._select_next_points(models, likelihoods, self.best_y, batch_size)
            
            next_y = []
            for x in next_points:
              if self.func_evals >= self.budget:
                  break
              next_y.append(func(x))
              self.func_evals += 1
            next_y = np.array(next_y).reshape(-1,1)
            
            if next_y.size == 0:
                break


            self.X = np.concatenate((self.X, next_points[:next_y.size]), axis=0)
            self.y = np.concatenate((self.y, next_y), axis=0)
            
            current_best_idx = np.argmin(self.y)
            if self.y[current_best_idx].item() < self.best_y:
                self.best_y = self.y[current_best_idx].item()
                best_x = self.X[current_best_idx]
            
            try:
              local_search_y, local_search_x = self._local_search(func, best_x)
              if local_search_y < self.best_y:
                  self.best_y = local_search_y
                  best_x = local_search_x
            except Exception as e:
                if "Budget Exceeded" in str(e):
                  break
            
            self._update_weights()
            rest_of_budget -= batch_size
            
        return self.best_y, best_x
