### Description
The problem involves minimizing two functions from the BBOB test suite, F24-LunacekBiRastrigin and F19-GriewankRosenbrock, in a 5-dimensional space with bounds [-5.0, 5.0] for each dimension. 
To address this problem, we can design a Bayesian Optimization algorithm that utilizes a Gaussian Process (GP) as the surrogate model and the Expected Improvement (EI) acquisition function to select the next points to evaluate. 
The GP will be used to model the objective function, and the EI acquisition function will be used to balance exploration and exploitation. 
We will also use a QMC sampling method to initialize the GP model.

### /Description

### Code
```python
from typing import Callable
import numpy as np
import torch
import gpytorch
from scipy.stats import qmc, norm

class AdaptiveAcquisitionBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.n_dims = dim
        self.bounds = np.array([[-5.0] * self.n_dims, [5.0] * self.n_dims])
        self.model = None
        self.likelihood = None
        self.acquisition_fn = self._expected_improvement
        self.model_losses = []
    
    def _sample_points(self, n_points) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=self.n_dims)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])
    
    def _fit_model(self, X, y):
        class GPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(GPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
            
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
        train_x = torch.tensor(X, dtype=torch.float32)
        train_y = torch.tensor(y, dtype=torch.float32).reshape(-1)  # Reshape to match expected shape
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = GPModel(train_x, train_y, self.likelihood)
        
        self.model.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        for _ in range(50):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss = loss.mean()  # Ensure loss is a scalar
            loss.backward()
            optimizer.step()
        
        return self.model
    
    def _get_model_loss(self, model, X, y) -> np.float64:
        model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            predictions = self.likelihood(model(torch.tensor(X, dtype=torch.float32))).mean.numpy()
        return np.mean((predictions - y) ** 2)
    
    def _expected_improvement(self, X) -> np.ndarray:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            posterior = self.likelihood(self.model(X_tensor))
            mean = posterior.mean.numpy()
            std = posterior.stddev.numpy()
        
        best_y = np.min(self.model.train_targets.numpy())
        Z = (best_y - mean) / std
        ei = (best_y - mean) * norm.cdf(Z) + std * norm.pdf(Z)
        return ei
    
    def _select_next_points(self, batch_size) -> np.ndarray:
        candidate_points = self._sample_points(1000)
        ei_values = self._expected_improvement(candidate_points)  # Corrected function name
        top_indices = np.argsort(ei_values)[-batch_size:]
        return candidate_points[top_indices]
    
    def __call__(self, func: Callable[[np.ndarray], np.ndarray]) -> tuple[np.float64, np.ndarray]:
        
        n_initial_points = min(10 * self.n_dims, self.budget // 2)
        X = self._sample_points(n_initial_points)
        y = func(X)
        
        self.model = self._fit_model(X, y)
        self.model_losses.append(self._get_model_loss(self.model, X, y))
        
        rest_of_budget = self.budget - n_initial_points
        while rest_of_budget > 0:
            batch_size = min(5, rest_of_budget)
            next_X = self._select_next_points(batch_size)
            next_y = func(next_X)
            
            X = np.vstack([X, next_X])
            y = np.vstack([y, next_y])
            
            self.model = self._fit_model(X, y)
            self.model_losses.append(self._get_model_loss(self.model, X, y))
            
            rest_of_budget -= batch_size
        
        return np.min(y), X[np.argmin(y)]
```

### /Code