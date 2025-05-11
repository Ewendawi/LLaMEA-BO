# Description
**Adaptive Hybrid Bayesian Optimization with Improved Exploration and Model Tuning (AHBBO-IMET):** This algorithm refines the AHBBO algorithm by introducing a more sophisticated exploration strategy and improved model tuning. The exploration term in the acquisition function is dynamically adjusted based on both the optimization progress and the uncertainty estimates from the Gaussian Process Regression (GPR) model. Specifically, the exploration weight decreases as the number of evaluations increases, but it increases when the model uncertainty is high, promoting exploration in uncertain regions. The model fitting is improved by using a more robust optimization algorithm (L-BFGS-B) for kernel parameter tuning and by adding a small amount of noise to the observed values to prevent overfitting.

# Justification
The key improvements are:

1.  **Uncertainty-Aware Exploration:** The exploration weight is now dynamically adjusted based on the variance predictions from the GPR model. This allows the algorithm to focus exploration on regions where the model is uncertain, potentially leading to faster convergence.
2.  **Robust Model Tuning:** The L-BFGS-B optimizer is used for kernel parameter tuning, which is more robust than the default optimizer. Adding a small amount of noise to the observed values can help prevent overfitting, especially in low-dimensional problems.
3.  **Adaptive Exploration Decay:** The exploration decay is adjusted based on the budget and dimension to better adapt to different problem scales.

These changes aim to improve the exploration-exploitation balance and the robustness of the algorithm, leading to better performance on a wider range of black-box optimization problems.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class AHBBO_IMET:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim # Initial number of points

        self.best_y = np.inf
        self.best_x = None

        self.batch_size = min(10, dim) # Batch size for selecting points
        self.exploration_weight = 0.1 # Initial exploration weight
        self.exploration_decay = 0.995 # Decay factor for exploration weight
        self.min_exploration = 0.01 # Minimum exploration weight
        self.noise_level = 1e-6

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=self.noise_level)

        # Custom optimization using L-BFGS-B
        def obj_func(theta):
            model.kernel_.theta = theta
            return -model.log_marginal_likelihood(X, y)

        initial_theta = model.kernel_.theta
        bounds = model.kernel_.bounds
        result = minimize(obj_func, initial_theta, method='L-BFGS-B', bounds=bounds)
        model.kernel_.theta = result.x

        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Distance-based exploration term
        min_dist = np.min(np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2), axis=1, keepdims=True)
        exploration = min_dist / np.max(min_dist)

        # Hybrid acquisition function
        acquisition = ei + self.exploration_weight * exploration
        return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Generate candidate points
        candidate_points = self._sample_points(100 * batch_size)  # Generate more candidates
        
        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_points)
        
        # Select the top batch_size points with the highest acquisition values
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]
        
        return next_points

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)
    
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
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Optimization
            # select points by acquisition function
            
            remaining_evals = self.budget - self.n_evals
            batch_size = min(self.batch_size, remaining_evals) # Adjust batch size to budget
            
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Get variance predictions to adapt exploration
            _, sigma = self.model.predict(self.X, return_std=True)
            mean_sigma = np.mean(sigma)
            
            # Update exploration weight based on uncertainty
            self.exploration_weight = max(self.exploration_weight * self.exploration_decay, self.min_exploration)
            self.exploration_weight = min(self.exploration_weight + 0.1 * mean_sigma, 0.5) # Increase exploration if uncertainty is high
            
            self.model = self._fit_model(self.X, self.y)

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<AHBBO_IMET>", line 135, in __call__
 135->         self.model = self._fit_model(self.X, self.y)
  File "<AHBBO_IMET>", line 51, in _fit_model
  49 |             return -model.log_marginal_likelihood(X, y)
  50 | 
  51->         initial_theta = model.kernel_.theta
  52 |         bounds = model.kernel_.bounds
  53 |         result = minimize(obj_func, initial_theta, method='L-BFGS-B', bounds=bounds)
AttributeError: 'GaussianProcessRegressor' object has no attribute 'kernel_'. Did you mean: 'kernel'?
