from typing import Callable
from scipy.stats import qmc
import numpy as np
import torch
import gpytorch
from scipy.stats import norm
from torch.autograd import grad

class GPHybridAdaptiveBatchBOv2:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.likelihood = None
        self.best_y = float('inf')
        self.exploration_weight = 0.5
        self.min_batch_size = 1
        self.max_batch_size = 10
        self.grad_norm_threshold = 1.0

    def _sample_points(self, n_points) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device).flatten()

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = gpytorch.models.ExactGP(X_tensor, y_tensor, self.likelihood)
        
        mean_module = gpytorch.means.ConstantMean()
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.model.mean_module = mean_module
        self.model.covar_module = covar_module
        
        self.model = self.model.to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        self.model.train()
        self.likelihood.train()
        
        training_iter = 50
        for i in range(training_iter):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = -mll(output, y_tensor)
            loss.backward()
            optimizer.step()
        
        self.model.eval()
        self.likelihood.eval()


    def _expected_improvement(self, X) -> np.ndarray:
        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model(X_tensor)
            mean = posterior.mean
            std = torch.sqrt(posterior.variance)
        
        mean = mean.cpu().numpy().reshape(-1, 1)
        std = std.cpu().numpy().reshape(-1, 1)
        
        imp =  self.best_y - mean
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std <= 0] = 0
        return ei

    def _uncertainty(self, X) -> np.ndarray:
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model(X_tensor)
            std = torch.sqrt(posterior.variance)
        return std.cpu().numpy().reshape(-1, 1)
    
    def _acquisition_function(self, X) -> np.ndarray:
        ei_values = self._expected_improvement(X)
        uncertainty_values = self._uncertainty(X)
        
        combined_acq = (1 - self.exploration_weight) * ei_values + self.exploration_weight * uncertainty_values
        return combined_acq

    def _calculate_grad_norm(self, X):
      X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(self.device)
      
      with gpytorch.settings.fast_pred_var():
          posterior = self.model(X_tensor)
          mean = posterior.mean
      
      grad_mean = grad(mean.sum(), X_tensor)[0]
      if grad_mean is None:
        return 0.0
      grad_norm = torch.norm(grad_mean, dim=1)
      return torch.max(grad_norm).cpu().item()

    def _select_next_points(self, batch_size) -> np.ndarray:
        n_candidates = min(1000, 100 * self.dim)
        candidates = self._sample_points(n_candidates)
        acq_values = self._acquisition_function(candidates)
        indices = np.argsort(-acq_values.flatten())[:batch_size]
        return candidates[indices]

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        n_initial_points = min(20, 5 * self.dim)
        self.X = self._sample_points(n_initial_points)
        self.y = np.array([func(x) for x in self.X]).reshape(-1, 1)
        
        best_idx = np.argmin(self.y)
        self.best_y = self.y[best_idx].item()
        best_x = self.X[best_idx].copy()
        
        rest_of_budget = self.budget - n_initial_points
        
        while rest_of_budget > 0:
            self._fit_model(self.X, self.y)
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                X_tensor = torch.tensor(self.X, dtype=torch.float32).to(self.device)
                posterior = self.model(X_tensor)
                std = torch.sqrt(posterior.variance)
                max_std = torch.max(std).item()
            
            grad_norm = self._calculate_grad_norm(self.X)
            
            batch_size = min(rest_of_budget, max(self.min_batch_size, int(np.ceil(max_std * 5))))
            if grad_norm > self.grad_norm_threshold:
                batch_size = min(batch_size, self.max_batch_size)
            else:
                batch_size = min(batch_size, self.min_batch_size)
                
            next_points = self._select_next_points(batch_size)
            next_y = np.array([func(x) for x in next_points]).reshape(-1, 1)
            
            self.X = np.concatenate((self.X, next_points), axis=0)
            self.y = np.concatenate((self.y, next_y), axis=0)
            
            current_best_idx = np.argmin(self.y)
            if self.y[current_best_idx].item() < self.best_y:
                self.best_y = self.y[current_best_idx].item()
                best_x = self.X[current_best_idx].copy()
                
            self.exploration_weight = max(0.01, self.exploration_weight * 0.95)
            self.grad_norm_threshold = max(0.1, self.grad_norm_threshold * 0.95)


            rest_of_budget -= batch_size
        return self.best_y, best_x