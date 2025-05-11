# Description
**EHBBO_AFLS_v4: Enhanced Hybrid Bayesian Optimization with Adaptive Acquisition, Focused Local Search, and Dynamic Temperature Scaling with Memory and Adaptive Kernel.** This algorithm improves EHBBO_AFLS_v3 by incorporating an adaptive kernel selection mechanism for the Gaussian Process model. It dynamically switches between a Matern kernel and an RBF kernel based on the optimization progress, specifically the rate of improvement in the best function value. The algorithm also refines the temperature scaling in Thompson Sampling and adjusts the local search radius based on the function's landscape characteristics, estimated by the GP's uncertainty.

# Justification
The key improvements and their justifications are:

1.  **Adaptive Kernel Selection:** The choice of kernel significantly impacts the GP's performance. Matern kernels are suitable for non-smooth functions, while RBF kernels are better for smooth functions. By adaptively switching between them based on the optimization progress (rate of improvement), the algorithm can better model different function landscapes. If the improvement rate slows down, it suggests the function might be less smooth, prompting a switch to the Matern kernel, and vice versa.

2.  **Refined Temperature Scaling:** The temperature in Thompson Sampling controls the exploration-exploitation balance. The original temperature scaling was based on the best function value and the fraction of evaluations. The refined scaling incorporates the GP's predicted uncertainty (sigma) to adjust the temperature. Higher uncertainty leads to higher temperature, promoting exploration in less-explored regions.

3.  **Adaptive Local Search Radius:** The local search radius determines the extent of the focused local search around the best point. The original radius decayed linearly with the remaining budget. The refined radius is now inversely proportional to the average GP uncertainty in the neighborhood of the best point. High uncertainty implies a rough landscape, requiring a smaller radius to avoid getting stuck in local optima.

4. **Memory Management Improvement:** Instead of just keeping the best points, we now keep a diverse set of points by using k-means clustering on the memory and keeping the best point from each cluster. This helps to avoid the memory being dominated by points from a single region and promotes exploration.

These changes enhance the algorithm's ability to adapt to different function characteristics, leading to improved performance across the BBOB test suite.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize
from sklearn.cluster import KMeans

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
        self.kernel_type = 'Matern' # Initial kernel type
        self.improvement_history = []

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
        if len(self.improvement_history) > 5:
            improvement_rate = np.mean(np.diff(self.improvement_history[-5:]))
            if improvement_rate > 0.01 and self.kernel_type == 'Matern':
                self.kernel_type = 'RBF'
                print("Switching to RBF kernel")
            elif improvement_rate <= 0.01 and self.kernel_type == 'RBF':
                self.kernel_type = 'Matern'
                print("Switching to Matern kernel")

        if self.kernel_type == 'Matern':
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, nu=2.5)  # Matern kernel
        else:
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0)  # RBF kernel
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
        temperature = (1.0 / (1.0 + np.exp(- (self.best_y + 1e-9)))) * (1 - fraction_evaluated) + 0.01 + np.mean(sigma) # Dynamic temperature + uncertainty
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
        _, sigma = self.gp.predict(self.best_x.reshape(1, -1), return_std=True)
        local_search_radius = 0.15 * remaining_budget_fraction / (np.mean(sigma) + 1e-9) # Adaptive radius, depends on uncertainty

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
            self.improvement_history.append(self.best_y)

        # Update memory
        if self.memory_X is None:
            self.memory_X = new_X
            self.memory_y = new_y
        else:
            self.memory_X = np.vstack((self.memory_X, new_X))
            self.memory_y = np.vstack((self.memory_y, new_y))

        # Memory management: keep a diverse set of points
        if len(self.memory_X) > self.memory_size:
            kmeans = KMeans(n_clusters=self.memory_size // 5, random_state=0, n_init=5)  # Reduce number of clusters
            cluster_labels = kmeans.fit_predict(self.memory_X)
            indices_to_keep = []
            for i in range(kmeans.n_clusters):
                cluster_indices = np.where(cluster_labels == i)[0]
                best_index_in_cluster = cluster_indices[np.argmin(self.memory_y[cluster_indices])]
                indices_to_keep.append(best_index_in_cluster)

            # Add some random points to the memory to maintain diversity
            num_random_points = min(self.memory_size - len(indices_to_keep), self.memory_size // 10)
            random_indices = np.random.choice(len(self.memory_X), size=num_random_points, replace=False)
            indices_to_keep.extend(random_indices)

            self.memory_X = self.memory_X[indices_to_keep]
            self.memory_y = self.memory_y[indices_to_keep]

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
 The algorithm EHBBO_AFLS_v4 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1602 with standard deviation 0.1086.

took 197.61 seconds to run.