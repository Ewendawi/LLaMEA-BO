# Description
**EHBBO_AFLS_v4: Enhanced Hybrid Bayesian Optimization with Adaptive Acquisition, Focused Local Search, Dynamic Temperature Scaling with Memory, and Adaptive Kernel.** This algorithm builds upon EHBBO_AFLS_v3 by incorporating an adaptive kernel for the Gaussian Process model. The kernel adapts its length scale based on the distribution of points in the search space, allowing for better model fitting in regions with varying density. The local search is also enhanced with a dynamic radius that adjusts based on the success rate of previous local searches.

# Justification
The key improvements are:

1.  **Adaptive Kernel:** The original algorithm uses a fixed Matern kernel. By allowing the kernel's length scale to adapt, the Gaussian Process can better model functions with varying smoothness. The length scale is adapted based on the median distance between neighboring points, which provides a measure of the density of points in the search space. This is crucial for handling functions with different characteristics across the search space.

2.  **Enhanced Local Search:** The local search radius is now adapted based on the success rate of previous local searches. If the local search consistently finds better points, the radius is increased to explore further. If the local search is not successful, the radius is decreased to focus on the immediate vicinity of the best point. This allows the algorithm to efficiently balance exploration and exploitation in the local search phase.

3.  **Memory-informed Initialization Improvement:** Uses the average of the best memory points within the radius instead of just the single best.

These changes are designed to improve the algorithm's ability to model complex functions and efficiently search the search space, leading to better optimization performance.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors

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
        self.local_search_success_rate = 0.5 # Initial success rate
        self.prev_best_y = float('inf')


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
        if len(X) > 1:
            neighbors = NearestNeighbors(n_neighbors=min(len(X), 10), algorithm='ball_tree').fit(X)
            distances, _ = neighbors.kneighbors(X)
            median_distance = np.median(distances[:, 1:])  # Exclude the first neighbor (self)
            length_scale = median_distance
        else:
            length_scale = 1.0  # Default length scale

        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=length_scale, nu=2.5)  # Matern kernel
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

        # Adaptive local search radius based on success rate
        if self.best_y < self.prev_best_y:
            self.local_search_success_rate = 0.8 * self.local_search_success_rate + 0.2 # Increase success rate
            local_search_radius *= 1.1 # Increase radius
        else:
            self.local_search_success_rate = 0.8 * self.local_search_success_rate # Decrease success rate
            local_search_radius *= 0.9 # Decrease radius
        local_search_radius = np.clip(local_search_radius, 0.01, 0.5) # Clip radius

        # Memory-informed initialization
        if self.memory_X is not None:
            distances = np.linalg.norm(self.memory_X - self.best_x, axis=1)
            within_radius = distances < local_search_radius
            if np.any(within_radius):
                # Use average of best memory points within radius
                best_memory_indices = np.argsort(self.memory_y[within_radius].flatten())[:min(3, np.sum(within_radius))] # Average of top 3
                x_start = np.mean(self.memory_X[within_radius][best_memory_indices], axis=0)
            else:
                x_start = self.best_x + np.random.normal(0, local_search_radius, self.dim)
                x_start = np.clip(x_start, self.bounds[0], self.bounds[1])
        else:
            x_start = self.best_x + np.random.normal(0, local_search_radius, self.dim)  # Perturb best point
            x_start = np.clip(x_start, self.bounds[0], self.bounds[1]) # Clip to bounds

        for _ in range(batch_size):
            res = minimize(lambda x: -self._acquisition_function(x.reshape(1, -1))[0,0],
                           x_start,
                           bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)],
                           method='L-BFGS-B',
                           options={'maxiter': 20})  # Limit iterations
            x_next.append(res.x)
            x_start = self.best_x + np.random.normal(0, local_search_radius, self.dim)
            x_start = np.clip(x_start, self.bounds[0], self.bounds[1])
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
            self.memory_X = np.vstack((self.memory_X, new_X))
            self.memory_y = np.vstack((self.memory_y, new_y))

        # Memory management: keep only the best 'memory_size' points
        if len(self.memory_X) > self.memory_size:
            indices = np.argsort(self.memory_y.flatten())[:self.memory_size]
            self.memory_X = self.memory_X[indices]
            self.memory_y = self.memory_y[indices]

        self.prev_best_y = self.best_y

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
## Feedback
 The algorithm EHBBO_AFLS_v4 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1705 with standard deviation 0.1045.

took 379.16 seconds to run.