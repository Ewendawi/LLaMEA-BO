# Description
**DABO: Distribution-Aware Bayesian Optimization:** This algorithm introduces a distribution-aware exploration strategy within Bayesian Optimization. It uses a Gaussian Process Regression (GPR) model for surrogate modeling. The acquisition function combines Expected Improvement (EI) with a distribution matching term. This term encourages sampling from a distribution that is close to the distribution of promising regions identified by the GPR model. Kernel Density Estimation (KDE) is used to estimate the distribution of promising regions. A modified sampling strategy is employed to sample points from the learned distribution. The initial exploration is performed using Latin Hypercube Sampling (LHS).

# Justification
This algorithm aims to address the limitations of previous algorithms by explicitly modeling and leveraging the distribution of promising regions in the search space.  The distribution matching term in the acquisition function guides the search towards areas where the model predicts high potential, while KDE provides a flexible way to estimate the distribution. This approach balances exploration and exploitation by focusing on regions that are both promising and relatively unexplored. The error in CGHBO was due to a broadcasting error when sampling from the covariance matrix. DABO avoids this by using KDE to model the distribution.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KernelDensity

class DABO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim
        self.kde = None
        self.best_x = None
        self.best_y = float('inf')
        self.distribution_weight = 0.1

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, method='lhs'):
        # sample points
        # return array of shape (n_points, n_dims)
        if method == 'lhs':
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        elif method == 'kde':
            if self.kde is None:
                return self._sample_points(n_points, method='lhs')
            else:
                # Sample from KDE
                samples = self.kde.sample(n_points)
                samples = np.clip(samples, self.bounds[0], self.bounds[1])
                return samples
        else:
            raise ValueError("Invalid sampling method.")

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X, model):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if self.best_y is None:
            ei = np.zeros_like(mu)
        else:
            imp = self.best_y - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0  # handle zero sigma

        # Add distribution matching term
        if self.kde is not None:
            log_likelihood = self.kde.score_samples(X).reshape(-1, 1)
            ei = ei + self.distribution_weight * np.exp(log_likelihood) # Use exp to avoid negative values

        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # return array of shape (batch_size, n_dims)
        
        # Sample candidate points using KDE
        candidate_points = self._sample_points(100 * batch_size, method='kde')

        # Calculate acquisition function values
        model = self._fit_model(self.X, self.y)
        acquisition_values = self._acquisition_function(candidate_points, model)

        # Select top batch_size points based on acquisition function values
        indices = np.argsort(-acquisition_values.flatten())[:batch_size]
        selected_points = candidate_points[indices]

        return selected_points

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y
    
    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        # Do not change the function signature
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))
        
        # Update best seen value
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

        # Update KDE
        if self.X is not None:
            # Identify promising regions (e.g., top 20% of evaluated points)
            threshold = np.percentile(self.y, 20)
            promising_points = self.X[self.y <= threshold]

            if len(promising_points) > self.dim + 1:  # Ensure enough points for KDE
                self.kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(promising_points)
            else:
                self.kde = None
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        initial_X = self._sample_points(self.n_init, method='lhs')
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        
        batch_size = 5
        while self.n_evals < self.budget:
            # Optimization
            model = self._fit_model(self.X, self.y)
            
            # select points by acquisition function
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<DABO>", line 140, in __call__
 140->         self._update_eval_points(initial_X, initial_y)
  File "<DABO>", line 124, in _update_eval_points
 122 |             # Identify promising regions (e.g., top 20% of evaluated points)
 123 |             threshold = np.percentile(self.y, 20)
 124->             promising_points = self.X[self.y <= threshold]
 125 | 
 126 |             if len(promising_points) > self.dim + 1:  # Ensure enough points for KDE
IndexError: boolean index did not match indexed array along dimension 1; dimension is 5 but corresponding boolean dimension is 1
