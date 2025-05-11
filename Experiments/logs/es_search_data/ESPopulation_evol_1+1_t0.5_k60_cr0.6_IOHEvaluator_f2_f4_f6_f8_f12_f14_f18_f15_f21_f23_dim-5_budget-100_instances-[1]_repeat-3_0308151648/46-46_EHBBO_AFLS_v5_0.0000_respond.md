# Description
**EHBBO_AFLS_v5: Enhanced Hybrid Bayesian Optimization with Adaptive Acquisition, Focused Local Search, Dynamic Temperature Scaling, Memory, Adaptive Kernel, and Active Data Augmentation.** This algorithm refines EHBBO_AFLS_v4 by incorporating active data augmentation to enhance the GP model's accuracy and robustness, especially in regions with limited data. It also introduces a more sophisticated local search strategy with adaptive restart mechanisms to escape local optima. Finally, the Thompson sampling temperature is refined.

# Justification
The key improvements in EHBBO_AFLS_v5 are:

1.  **Active Data Augmentation:** To address the issue of limited data, especially in high-dimensional spaces, active data augmentation is introduced. This involves generating synthetic data points in regions where the GP model has high uncertainty. The uncertainty is quantified using the GP's predictive variance. By adding these synthetic points to the training data, the GP model becomes more robust and accurate, leading to better exploration and exploitation. The number of synthetic points is adaptive based on the remaining budget.

2.  **Enhanced Local Search with Adaptive Restarts:** The local search strategy is enhanced with an adaptive restart mechanism. If the local search fails to find a better solution after a certain number of iterations, the search is restarted from a different point in the vicinity of the current best solution. This helps to escape local optima and explore a wider region of the search space. The number of restarts is adaptive based on the remaining budget.

3. **Refined Thompson Sampling Temperature:** The temperature scaling in Thompson sampling is further refined to balance exploration and exploitation. The new temperature scaling is proportional to the predictive variance and inversely proportional to the number of evaluations. This ensures that the algorithm explores more in the initial stages and exploits more in the later stages.

These changes aim to improve the algorithm's ability to handle complex, high-dimensional black box optimization problems by enhancing the GP model's accuracy, improving the local search strategy, and refining the exploration-exploitation balance.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

class EHBBO_AFLS_v5:
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
        self.kernel_type = 'matern'
        self.local_search_iterations = 20

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

        # Adaptive kernel selection
        rbf_kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0)
        matern_kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, nu=2.5)

        gp_rbf = GaussianProcessRegressor(kernel=rbf_kernel, n_restarts_optimizer=5, alpha=1e-6)
        gp_matern = GaussianProcessRegressor(kernel=matern_kernel, n_restarts_optimizer=5, alpha=1e-6)

        gp_rbf.fit(X, y)
        gp_matern.fit(X, y)

        log_likelihood_rbf = gp_rbf.log_marginal_likelihood(gp_rbf.kernel_.theta)
        log_likelihood_matern = gp_matern.log_marginal_likelihood(gp_matern.kernel_.theta)

        if log_likelihood_rbf > log_likelihood_matern:
            self.gp = gp_rbf
            self.kernel_type = 'rbf'
        else:
            self.gp = gp_matern
            self.kernel_type = 'matern'

        # Active data augmentation
        if self.n_evals < self.budget * 0.7: # Augment only in the first 70% of the budget
            n_augment = int((self.budget - self.n_evals) * 0.05)  # Augment up to 5% of remaining budget
            X_augment = self._sample_points(n_augment)
            mu, sigma = self.gp.predict(X_augment, return_std=True)
            # Add synthetic points with values around the mean, perturbed by the uncertainty
            y_augment = mu.reshape(-1,1) + np.random.normal(0, sigma, size=(n_augment, 1))
            X = np.vstack((X, X_augment))
            y = np.vstack((y, y_augment))

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
        # Refined temperature scaling
        temperature = sigma / (1 + self.n_evals) + 0.001
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
        n_restarts = int(3 * remaining_budget_fraction + 1) # Adaptive restarts

        # Memory-informed initialization
        if self.memory_X is not None:
            distances = np.linalg.norm(self.memory_X - self.best_x, axis=1)
            within_radius = distances < local_search_radius
            if np.any(within_radius):
                best_memory_index = np.argmin(self.memory_y[within_radius])
                x_start = self.memory_X[within_radius][best_memory_index]
            else:
                x_start = self.best_x + np.random.normal(0, local_search_radius, self.dim)
                x_start = np.clip(x_start, self.bounds[0], self.bounds[1])
        else:
            x_start = self.best_x + np.random.normal(0, local_search_radius, self.dim)  # Perturb best point
            x_start = np.clip(x_start, self.bounds[0], self.bounds[1]) # Clip to bounds

        for _ in range(batch_size):
            best_x_local = None
            best_acq = float('inf')
            for _ in range(n_restarts):
                res = minimize(lambda x: -self._acquisition_function(x.reshape(1, -1))[0,0],
                               x_start,
                               bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)],
                               method='L-BFGS-B',
                               options={'maxiter': self.local_search_iterations})  # Limit iterations
                if -res.fun < best_acq:
                    best_acq = -res.fun
                    best_x_local = res.x
                x_start = self.best_x + np.random.normal(0, local_search_radius, self.dim)
                x_start = np.clip(x_start, self.bounds[0], self.bounds[1])
            x_next.append(best_x_local)

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
            # Replace worst memory point if the new point is better
            for i in range(len(new_X)):
                if new_y[i][0] < np.max(self.memory_y):
                    worst_index = np.argmax(self.memory_y)
                    self.memory_X[worst_index] = new_X[i]
                    self.memory_y[worst_index] = new_y[i]

        # Memory management: keep only the best 'memory_size' points
        if self.memory_X is not None and len(self.memory_X) > self.memory_size:
            indices = np.argsort(self.memory_y.flatten())[:self.memory_size]
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
  File "<EHBBO_AFLS_v5>", line 213, in __call__
 213->             self._fit_model(self.X, self.y)
  File "<EHBBO_AFLS_v5>", line 76, in _fit_model
  74 |             mu, sigma = self.gp.predict(X_augment, return_std=True)
  75 |             # Add synthetic points with values around the mean, perturbed by the uncertainty
  76->             y_augment = mu.reshape(-1,1) + np.random.normal(0, sigma, size=(n_augment, 1))
  77 |             X = np.vstack((X, X_augment))
  78 |             y = np.vstack((y, y_augment))
  File "mtrand.pyx", line 1540, in numpy.random.mtrand.RandomState.normal
  File "_common.pyx", line 600, in numpy.random._common.cont
  File "_common.pyx", line 518, in numpy.random._common.cont_broadcast_2
  File "_common.pyx", line 245, in numpy.random._common.validate_output_shape
ValueError: Output size (4, 1) is not compatible with broadcast dimensions of inputs (4, 4).
