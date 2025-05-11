# Description
EHBBO_AFLS_v6: Enhanced Hybrid Bayesian Optimization with Adaptive Acquisition, Focused Local Search, Dynamic Temperature Scaling, Memory, Adaptive Kernel, Gradient-Enhanced Local Search, and Adaptive Local Search Radius. This algorithm builds upon EHBBO_AFLS_v5 by incorporating a more sophisticated adaptive local search radius strategy based on the recent performance of the local search. If the local search consistently finds better points, the radius shrinks more slowly, and vice versa. This allows the algorithm to adapt better to the local landscape and potentially escape local optima more effectively. Also, the memory update strategy is improved by considering both novelty and improvement relative to the *current best* rather than the worst in memory.

# Justification
The key improvements are:

1.  **Adaptive Local Search Radius:** The original algorithm used a linearly decaying local search radius, which might not be optimal for all functions. The new adaptive strategy adjusts the radius based on the recent success of the local search. If the local search is consistently improving the best value, the radius decay is slowed down, allowing for more focused exploration. Conversely, if the local search is not yielding improvements, the radius decay is accelerated, encouraging broader exploration. This helps to balance exploration and exploitation more effectively.

2.  **Improved Memory Update:** The memory update strategy now considers the improvement relative to the *current best* solution, rather than just replacing the worst memory point if the new point is better than the worst memory point. This ensures that the memory stores points that are truly valuable in the context of the best known solution, leading to a more effective utilization of the memory for guiding the search.

3. **Gradient Approximation Improvement:** The gradient approximation uses a central difference scheme. This can be unstable in some cases. Adding a small regularization term can help to stabilize the gradient approximation.

These changes aim to improve the algorithm's ability to adapt to different function landscapes and escape local optima, while maintaining computational efficiency.

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

class EHBBO_AFLS_v6:
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
        self.local_search_radius = 0.15  # Initial local search radius
        self.radius_decay_rate = 0.95 # Initial decay rate
        self.success_rate = 0.0 # Track success rate of local search

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
        temperature = (0.5 / (1.0 + np.exp(10 * (fraction_evaluated - 0.5)))) # Sigmoid temperature
        thompson_samples = np.random.normal(mu, temperature * sigma).reshape(-1, 1)

        # Expected Improvement
        ei = self._expected_improvement(X)

        # Adaptive weighting
        self.alpha = self.alpha * (0.9 + 0.1 * fraction_evaluated) # Refined alpha decay
        acquisition = self.alpha * thompson_samples + (1 - self.alpha) * ei
        return acquisition

    def _approximate_gradient(self, func, x, epsilon=1e-5):
        # Approximate gradient using finite differences
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)
            # Add regularization term to stabilize gradient
            grad[i] /= (1 + 1e-6 * np.linalg.norm(x))
        return grad

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Focused Local Search around best point
        x_next = []
        # Adaptive radius
        #remaining_budget_fraction = 1 - (self.n_evals / self.budget)
        #self.local_search_radius = 0.15 * remaining_budget_fraction # Adaptive radius, slower decay

        # Memory-informed initialization
        if self.memory_X is not None:
            distances = np.linalg.norm(self.memory_X - self.best_x, axis=1)
            within_radius = distances < self.local_search_radius
            if np.any(within_radius):
                best_memory_index = np.argmin(self.memory_y[within_radius])
                x_start = self.memory_X[within_radius][best_memory_index]
            else:
                x_start = self.best_x + np.random.normal(0, self.local_search_radius, self.dim)
                x_start = np.clip(x_start, self.bounds[0], self.bounds[1])
        else:
            x_start = self.best_x + np.random.normal(0, self.local_search_radius, self.dim)  # Perturb best point
            x_start = np.clip(x_start, self.bounds[0], self.bounds[1]) # Clip to bounds

        successful_local_searches = 0

        for _ in range(batch_size):
            # Gradient-enhanced local search
            def objective(x):
                return -self._acquisition_function(x.reshape(1, -1))[0,0]

            def gradient(x):
                # Approximate gradient of the acquisition function
                return -self._approximate_gradient(lambda x_in: self._acquisition_function(x_in.reshape(1, -1))[0,0], x)

            res = minimize(objective,
                           x_start,
                           jac=gradient,
                           bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)],
                           method='L-BFGS-B',
                           options={'maxiter': 20})  # Limit iterations
            x_next.append(res.x)

            # Check if local search improved the best value
            current_acq = -objective(res.x)
            best_acq = -objective(self.best_x)

            if current_acq > best_acq:
                successful_local_searches += 1

            x_start = self.best_x + np.random.normal(0, self.local_search_radius, self.dim)
            x_start = np.clip(x_start, self.bounds[0], self.bounds[1])

        # Update local search radius adaptively
        self.success_rate = 0.9 * self.success_rate + 0.1 * (successful_local_searches / batch_size) # Exponential smoothing
        if self.success_rate > 0.5:
             self.radius_decay_rate = min(0.99, self.radius_decay_rate + 0.005) #Slower decay if successful
        else:
             self.radius_decay_rate = max(0.8, self.radius_decay_rate - 0.005) #Faster decay if unsuccessful
        self.local_search_radius *= self.radius_decay_rate
        self.local_search_radius = max(0.001, self.local_search_radius) # Ensure radius doesn't vanish

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
            # Replace worst memory point if the new point is better and novel
            for i in range(len(new_X)):
                # Improvement relative to current best
                if new_y[i][0] < self.best_y:
                    # Novelty check: ensure the new point is not too close to existing memory points
                    distances = np.linalg.norm(self.memory_X - new_X[i], axis=1)
                    if np.min(distances) > 0.01: # Novelty threshold
                        # Replace the memory point furthest from the current best
                        distances_to_best = np.linalg.norm(self.memory_X - self.best_x, axis=1)
                        worst_index = np.argmax(distances_to_best)
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
## Feedback
 The algorithm EHBBO_AFLS_v6 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1751 with standard deviation 0.1115.

took 499.16 seconds to run.