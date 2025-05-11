# Description
EHBBO_AFLS_v6: Enhanced Hybrid Bayesian Optimization with Adaptive Acquisition, Focused Local Search, Dynamic Temperature Scaling, Memory with Clustering, Adaptive Kernel, and Gradient-Enhanced Local Search. This algorithm builds upon EHBBO_AFLS_v5 by incorporating clustering into the memory update strategy to improve diversity and prevent premature convergence. It also includes a refined adaptive weighting for Thompson Sampling and Expected Improvement based on the uncertainty of the GP model. The gradient approximation method is replaced with a more efficient one.

# Justification
1.  **Memory with Clustering:** The memory is now updated using a k-means clustering approach. This helps to maintain diversity in the memory by selecting representative points from different clusters. This is especially useful in multimodal functions where the algorithm might get stuck in a local optimum.
2.  **Adaptive Acquisition Weighting:** The weighting between Thompson Sampling and Expected Improvement is now dynamically adjusted based on the uncertainty (sigma) of the Gaussian Process model. Higher uncertainty leads to more exploration (Thompson Sampling), while lower uncertainty leads to more exploitation (Expected Improvement). This allows the algorithm to adapt to the characteristics of the objective function.
3. **Efficient Gradient Approximation:** The finite difference method for gradient approximation is replaced with a more efficient implementation that reduces the number of function calls.
4. **Refined Local Search:** The local search radius now adapts based on both the remaining budget and the GP's uncertainty, promoting broader exploration early on and finer-grained exploitation later.

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
from sklearn.cluster import KMeans

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

        # Adaptive weighting based on uncertainty
        mean_sigma = np.mean(sigma)
        self.alpha = 0.5 + 0.4 * np.tanh(2 - 20 * mean_sigma)  # Adjust alpha based on average sigma
        acquisition = self.alpha * thompson_samples + (1 - self.alpha) * ei
        return acquisition

    def _approximate_gradient(self, func, x, epsilon=1e-4):
        # More efficient gradient approximation
        grad = np.zeros_like(x)
        x_plus = x.copy()
        x_minus = x.copy()
        for i in range(len(x)):
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)
            x_plus[i] = x[i]  # Reset
            x_minus[i] = x[i] # Reset
        return grad

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Focused Local Search around best point
        x_next = []
        remaining_budget_fraction = 1 - (self.n_evals / self.budget)
        _, sigma = self.gp.predict(self.best_x.reshape(1, -1), return_std=True)
        local_search_radius = 0.15 * remaining_budget_fraction * (1 + np.tanh(5 - 100 * sigma)) # Adaptive radius

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
                return -self._approximate_gradient(lambda x_in: self._acquisition_function(x_in.reshape(1, -1))[0,0], x)

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

        # Update memory with clustering
        if self.memory_X is None:
            self.memory_X = new_X
            self.memory_y = new_y
        else:
            all_X = np.vstack((self.memory_X, new_X))
            all_y = np.vstack((self.memory_y, new_y))

            # Cluster the data
            n_clusters = min(self.memory_size, len(all_X))
            kmeans = KMeans(n_clusters=n_clusters, n_init=10)
            kmeans.fit(all_X)

            # Select the best point from each cluster
            cluster_ids = kmeans.labels_
            self.memory_X = np.zeros((n_clusters, self.dim))
            self.memory_y = np.zeros((n_clusters, 1))

            for i in range(n_clusters):
                cluster_points_x = all_X[cluster_ids == i]
                cluster_points_y = all_y[cluster_ids == i]
                best_index = np.argmin(cluster_points_y)
                self.memory_X[i] = cluster_points_x[best_index]
                self.memory_y[i] = cluster_points_y[best_index]

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
  File "<EHBBO_AFLS_v6>", line 233, in __call__
 233->             self._update_eval_points(X_next, y_next)
  File "<EHBBO_AFLS_v6>", line 207, in _update_eval_points
 205 |                 cluster_points_x = all_X[cluster_ids == i]
 206 |                 cluster_points_y = all_y[cluster_ids == i]
 207->                 best_index = np.argmin(cluster_points_y)
 208 |                 self.memory_X[i] = cluster_points_x[best_index]
 209 |                 self.memory_y[i] = cluster_points_y[best_index]
  File "<__array_function__ internals>", line 200, in argmin
ValueError: attempt to get argmin of an empty sequence
