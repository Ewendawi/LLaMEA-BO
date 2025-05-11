# Description
**EHBBO_AFLS_v4: Enhanced Hybrid Bayesian Optimization with Adaptive Acquisition, Focused Local Search, Dynamic Temperature Scaling with Memory, and Adaptive Kernel.** This algorithm builds upon EHBBO_AFLS_v3 by incorporating an adaptive kernel for the Gaussian Process model. The kernel's length scale is dynamically adjusted based on the distribution of sampled points and their function values. This allows the GP to better model functions with varying degrees of smoothness and correlation lengths. A secondary local search initialization strategy based on random sampling within the adaptive radius is added to increase diversity.

# Justification
The key improvements are:

1.  **Adaptive Kernel:** Instead of a fixed kernel, the length scale of the Matern kernel is adapted during the optimization process. This is crucial for handling functions with varying smoothness. The length scale is adjusted based on the variance of the evaluated points in each dimension. This allows the GP to automatically adjust its sensitivity to changes in different dimensions.

2.  **Diverse Local Search Initialization:** To avoid getting trapped in local optima, an alternative local search initialization strategy is introduced. With a small probability, the local search is initialized with a random point within the adaptive radius of the current best, rather than always perturbing the best point. This promotes exploration.

3.  **Refined Memory Update:** The memory update mechanism is refined to prioritize points that are both close to the current best solution and have good function values. This helps to maintain a memory of promising regions in the search space.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize

class EHBBO_AFLS_v4:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * (dim + 1) # Number of initial samples, increased for higher dimensions
        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.alpha = 0.5 # Initial weight for Thompson Sampling
        self.memory_X = None
        self.memory_y = None
        self.memory_size = 50  # Maximum size of the memory
        self.length_scale = np.ones(dim)  # Initial length scale for the kernel

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

        # Augment data with memory
        if self.memory_X is not None:
            X = np.vstack((X, self.memory_X))
            y = np.vstack((y, self.memory_y))

        # Adaptive kernel length scale
        self.length_scale = np.std(X, axis=0) + 1e-6  # Ensure non-zero length scale
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=self.length_scale, nu=2.5)  # Matern kernel
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)
        self.gp.fit(X, y)
        return self.gp

    def _expected_improvement(self, X):
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)
        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei.reshape(-1, 1)

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)

        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)

        # Thompson Sampling
        fraction_evaluated = self.n_evals / self.budget
        temperature = (1.0 / (1.0 + np.exp(- (self.best_y + 1e-9)))) * (1 - fraction_evaluated) + 0.01 # Dynamic temperature
        thompson_samples = np.random.normal(mu, temperature * sigma).reshape(-1, 1)

        # Expected Improvement
        ei = self._expected_improvement(X)

        # Adaptive weighting
        self.alpha = self.alpha * (0.9 + 0.1 * fraction_evaluated) # Refined alpha decay
        acquisition = self.alpha * thompson_samples + (1 - self.alpha) * ei
        return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Focused Local Search around best point
        x_next = []
        remaining_budget_fraction = 1 - (self.n_evals / self.budget)
        local_search_radius = 0.15 * remaining_budget_fraction # Adaptive radius, slower decay

        for _ in range(batch_size):
            # Memory-informed initialization
            if self.memory_X is not None and np.random.rand() < 0.8: # 80% chance of using memory
                distances = np.linalg.norm(self.memory_X - self.best_x, axis=1)
                within_radius = distances < local_search_radius
                if np.any(within_radius):
                    best_memory_index = np.argmin(self.memory_y[within_radius])
                    x_start = self.memory_X[within_radius][best_memory_index]
                else:
                    x_start = self.best_x + np.random.normal(0, local_search_radius, self.dim)
                    x_start = np.clip(x_start, self.bounds[0], self.bounds[1])
            else: # 20% chance of random initialization within radius
                x_start = self.best_x + np.random.uniform(-local_search_radius, local_search_radius, self.dim)
                x_start = np.clip(x_start, self.bounds[0], self.bounds[1])

            res = minimize(lambda x: -self._acquisition_function(x.reshape(1, -1))[0,0],
                           x_start,
                           bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)],
                           method='L-BFGS-B',
                           options={'maxiter': 20})  # Limit iterations
            x_next.append(res.x)

        return np.array(x_next)

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

        # Update memory
        if self.memory_X is None:
            self.memory_X = new_X
            self.memory_y = new_y
        else:
            # Prioritize points close to the best and with good function values
            distances = np.linalg.norm(new_X - self.best_x, axis=1)
            priority = np.exp(-distances / local_search_radius) * np.exp(-(new_y - self.best_y)**2) # Gaussian priority
            
            self.memory_X = np.vstack((self.memory_X, new_X))
            self.memory_y = np.vstack((self.memory_y, new_y))
            
            # Memory management: keep only the best 'memory_size' points based on priority
            if len(self.memory_X) > self.memory_size:
                priority_existing = np.exp(-np.linalg.norm(self.memory_X[:-len(new_X)] - self.best_x, axis=1) / local_search_radius) * np.exp(-(self.memory_y[:-len(new_X)] - self.best_y)**2)
                combined_priority = np.concatenate((priority_existing.flatten(), priority.flatten()))
                indices = np.argsort(combined_priority)[-self.memory_size:]
                self.memory_X = self.memory_X[indices]
                self.memory_y = self.memory_y[indices]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Optimization loop
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Select next points by acquisition function
            batch_size = min(self.budget - self.n_evals, max(1, self.dim // 2)) # Adaptive batch size
            X_next = self._select_next_points(batch_size)

            # Evaluate the points
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<EHBBO_AFLS_v4>", line 186, in __call__
 186->             self._update_eval_points(X_next, y_next)
  File "<EHBBO_AFLS_v4>", line 151, in _update_eval_points
 149 |             # Prioritize points close to the best and with good function values
 150 |             distances = np.linalg.norm(new_X - self.best_x, axis=1)
 151->             priority = np.exp(-distances / local_search_radius) * np.exp(-(new_y - self.best_y)**2) # Gaussian priority
 152 |             
 153 |             self.memory_X = np.vstack((self.memory_X, new_X))
NameError: name 'local_search_radius' is not defined
