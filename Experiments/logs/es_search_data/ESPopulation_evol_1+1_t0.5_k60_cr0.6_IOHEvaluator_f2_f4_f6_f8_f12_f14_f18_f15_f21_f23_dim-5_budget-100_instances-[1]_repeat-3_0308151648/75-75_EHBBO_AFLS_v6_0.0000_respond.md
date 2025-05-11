# Description
EHBBO_AFLS_v6: Enhanced Hybrid Bayesian Optimization with Adaptive Acquisition, Focused Local Search, Dynamic Temperature Scaling, Memory, Adaptive Kernel, Gradient-Enhanced Local Search, and Gradient-Based Initial Exploration. This algorithm enhances EHBBO_AFLS_v5 by incorporating a gradient-based initial exploration strategy to improve the initial sampling phase, which is crucial for high-dimensional problems. It also refines the local search radius adaptation and memory update strategy to further improve performance.

# Justification
The key improvements in this version are:

1.  **Gradient-Based Initial Exploration:** The initial sampling is enhanced by incorporating gradient information. Instead of purely random sampling, a small number of points are selected based on the estimated gradient of the function. This helps to quickly identify promising regions in the search space, especially in high-dimensional problems. The `_gradient_based_initial_exploration` function is added to achieve this.

2.  **Refined Local Search Radius Adaptation:** The adaptation of the local search radius is refined to provide a more balanced exploration-exploitation trade-off. The radius now adapts based on both the remaining budget and the function's landscape (estimated by the GP's uncertainty).

3.  **Improved Memory Update:** The memory update strategy is improved by considering both the improvement in function value and the novelty of the point. This helps to maintain a diverse set of promising solutions in the memory.

4.  **Adaptive Epsilon for Gradient Approximation:** A small modification is made to the approximate gradient to make the epsilon adaptive to the remaining budget, to improve gradient approximation accuracy.

These changes are designed to improve the algorithm's ability to explore the search space effectively, exploit promising regions, and adapt to different problem characteristics.

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

    def _approximate_gradient(self, func, x, epsilon):
        # Approximate gradient using finite differences
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)
        return grad

    def _gradient_based_initial_exploration(self, func, n_points=5):
        # Explore the space based on gradient information
        X = self._sample_points(n_points)
        y = self._evaluate_points(func, X)
        self._update_eval_points(X, y)
        
        # Fit GP model
        self._fit_model(self.X, self.y)

        # Calculate gradients at sampled points
        epsilon = 1e-5
        gradients = np.array([self._approximate_gradient(func, x, epsilon) for x in self.X])

        # Select points with large gradients
        grad_magnitudes = np.linalg.norm(gradients, axis=1)
        top_indices = np.argsort(grad_magnitudes)[-n_points:]
        
        # Perturb these points and evaluate
        X_explore = []
        for i in top_indices:
            x_new = self.X[i] + np.random.normal(0, 0.1, self.dim) # Perturb
            x_new = np.clip(x_new, self.bounds[0], self.bounds[1]) # Clip
            X_explore.append(x_new)
        X_explore = np.array(X_explore)
        y_explore = self._evaluate_points(func, X_explore)
        
        return X_explore, y_explore

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Focused Local Search around best point
        x_next = []
        remaining_budget_fraction = 1 - (self.n_evals / self.budget)
        # Adaptive radius, slower decay, consider GP uncertainty
        mu, sigma = self.gp.predict(self.best_x.reshape(1, -1), return_std=True)
        local_search_radius = 0.15 * remaining_budget_fraction * (1 + sigma[0])

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
            # Gradient-enhanced local search
            def objective(x):
                return -self._acquisition_function(x.reshape(1, -1))[0,0]

            def gradient(x):
                # Approximate gradient of the acquisition function
                epsilon = 1e-5 * remaining_budget_fraction
                return -self._approximate_gradient(lambda x_in: self._acquisition_function(x_in.reshape(1, -1))[0,0], x, epsilon)

            res = minimize(objective,
                           x_start,
                           jac=gradient,
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
            # Replace worst memory point if the new point is better and novel
            for i in range(len(new_X)):
                if new_y[i][0] < np.max(self.memory_y):
                    # Novelty check: ensure the new point is not too close to existing memory points
                    distances = np.linalg.norm(self.memory_X - new_X[i], axis=1)
                    if np.min(distances) > 0.01: # Novelty threshold
                        worst_index = np.argmax(self.memory_y)
                        self.memory_X[worst_index] = new_X[i]
                        self.memory_y[worst_index] = new_y[i]
                #Also, replace if the point is better than the worst in memory, even if not novel
                elif new_y[i][0] < np.max(self.memory_y):
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
        X_init = self._sample_points(self.n_init - 5) #Reduce initial random samples
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Gradient-based initial exploration
        X_explore, y_explore = self._gradient_based_initial_exploration(func, n_points=5)
        self._update_eval_points(X_explore, y_explore)

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
  File "<EHBBO_AFLS_v6>", line 251, in __call__
 251->         X_explore, y_explore = self._gradient_based_initial_exploration(func, n_points=5)
  File "<EHBBO_AFLS_v6>", line 122, in _gradient_based_initial_exploration
 122->         gradients = np.array([self._approximate_gradient(func, x, epsilon) for x in self.X])
  File "<EHBBO_AFLS_v6>", line 122, in <listcomp>
 122->         gradients = np.array([self._approximate_gradient(func, x, epsilon) for x in self.X])
  File "<EHBBO_AFLS_v6>", line 108, in _approximate_gradient
 106 |             x_plus[i] += epsilon
 107 |             x_minus[i] -= epsilon
 108->             grad[i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)
 109 |         return grad
 110 | 
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
