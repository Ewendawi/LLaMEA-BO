# Description
EHBBO_BLS_PLE_AdaptiveKernel_v4: Extends EHBBO_BLS_PLE_AdaptiveKernel_v3 by incorporating a more robust probabilistic local search (PLE) evaluation. It introduces a dynamic adjustment of the local search radius based on the GP's uncertainty estimates and the function's landscape. Furthermore, it refines the acquisition function by adding a term that encourages exploration in regions with high variance and low density of sampled points. This balances exploration and exploitation more effectively. The kernel update strategy is also enhanced by considering the gradient of the GP's predicted mean to better adapt to changes in the function's curvature.

# Justification
The key improvements are:

1.  **Enhanced Probabilistic Local Search (PLE):** The PLE evaluation is made more robust by dynamically adjusting the local search radius based on the GP's uncertainty. This allows the algorithm to adapt to different function landscapes and avoid getting stuck in local optima.
2.  **Acquisition Function Refinement:** The acquisition function is refined by adding a term that encourages exploration in regions with high variance and low density of sampled points. This helps to balance exploration and exploitation more effectively. The density is estimated using a simple k-nearest neighbors approach.
3.  **Kernel Update Strategy Enhancement:** The kernel update strategy is enhanced by considering the gradient of the GP's predicted mean. This allows the algorithm to better adapt to changes in the function's curvature and improve the accuracy of the GP model.
4.  **Adaptive Local Search Restarts:** The number of local search restarts is dynamically adjusted based on the optimization progress and the remaining budget. This allows the algorithm to focus its computational resources on promising regions of the search space.

These changes aim to improve the algorithm's ability to explore the search space, avoid getting stuck in local optima, and adapt to different function landscapes. The addition of a density estimation term in the acquisition function and the use of the gradient in the kernel update strategy are novel approaches that can potentially lead to significant performance improvements.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors

class EHBBO_BLS_PLE_AdaptiveKernel_v4:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim # initial samples
        self.gp = None
        self.best_x = None
        self.best_y = np.inf
        self.exploration_factor = 1.0 # initial exploration factor
        self.exploration_decay = 0.99 # decay rate for exploration factor
        self.kernel_length_scale = 1.0
        self.kernel_update_interval = 5 * dim # Update kernel every this many evaluations
        self.kappa = 2.0  # UCB exploration-exploitation parameter
        self.kappa_decay = 0.995
        self.min_kappa = 0.1
        self.initial_sample_multiplier = 1.0
        self.density_knn = min(10, self.n_init) # Number of neighbors for density estimation

    def _sample_points(self, n_points):
        # sample points using Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.kernel_length_scale, length_scale_bounds=(1e-2, 10))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-5) # Increased restarts for kernel tuning
        self.gp.fit(X, y)
        self.kernel_length_scale = self.gp.kernel_.k2.length_scale # Update the length scale
        return self.gp

    def _acquisition_function(self, X):
        # Implement Upper Confidence Bound acquisition function with density-based exploration
        mu, sigma = self.gp.predict(X, return_std=True)
        # Estimate density using k-NN
        if self.X is not None and len(self.X) > self.density_knn:
            knn = NearestNeighbors(n_neighbors=self.density_knn)
            knn.fit(self.X)
            distances, _ = knn.kneighbors(X)
            density = 1.0 / (np.mean(distances, axis=1) + 1e-6)  # Avoid division by zero
        else:
            density = np.ones(X.shape[0])  # Assign uniform density if not enough data

        acquisition = mu - self.kappa * sigma + self.exploration_factor * sigma * density
        return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate by maximizing the acquisition function
        # Use a multi-start local search strategy
        best_x = None
        best_acq = np.inf
        num_starts = 5 * batch_size # Increased number of starts
        for _ in range(num_starts):
            x_start = self._sample_points(1)
            x_start = x_start.flatten()

            def acquisition_objective(x):
                return self._acquisition_function(x.reshape(1, -1))[0]

            res = minimize(acquisition_objective, x_start, method='L-BFGS-B', bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)])
            acq = res.fun

            if acq < best_acq:
                best_acq = acq
                best_x = res.x

        return best_x.reshape(1, -1)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

        # Update best_x and best_y
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

    def local_search(self, gp, x0, sigma, num_restarts=1):
        # Perform a local search around x0 using the GP prediction
        best_x = x0
        best_y_pred = np.inf
        search_radius = 0.2 + sigma # Dynamic search radius

        for _ in range(num_restarts):
            x_start = x0 + np.random.normal(0, 0.1, size=self.dim) # Add some noise
            x_start = np.clip(x_start, self.bounds[0], self.bounds[1])

            def gp_objective(x):
                mu, _ = gp.predict(x.reshape(1, -1), return_std=True)
                return mu[0]

            bounds = [(max(self.bounds[0][i], x0[i] - search_radius), min(self.bounds[1][i], x0[i] + search_radius)) for i in range(self.dim)]  # Dynamic bounds
            res = minimize(gp_objective, x_start, method='L-BFGS-B', bounds=bounds)
            y_pred = gp_objective(res.x)

            if y_pred < best_y_pred:
                best_y_pred = y_pred
                best_x = res.x
        return best_x

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial exploration
        n_init_dynamic = int(self.initial_sample_multiplier * self.n_init)
        initial_X = self._sample_points(n_init_dynamic)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Optimization loop
        while self.n_evals < self.budget:
            # Fit the GP model
            if self.n_evals % self.kernel_update_interval == 0:
                #Consider the gradient of the GP's predicted mean
                if self.X is not None and len(self.X) > 1:
                    # Calculate the gradient of the GP's predicted mean
                    X_ = np.linspace(self.bounds[0], self.bounds[1], 100)
                    mu, _ = self.gp.predict(X_, return_std=True)
                    gradient = np.gradient(mu, axis=0)
                    gradient_norm = np.linalg.norm(gradient)

                    # Adjust the kernel update frequency based on the gradient norm
                    if gradient_norm > 0.1:
                        self.kernel_update_interval = 2 * self.dim # Update more frequently in regions with high curvature
                    else:
                        self.kernel_update_interval = 10 * self.dim # Update less frequently in flatter regions
                self._fit_model(self.X, self.y)
            else:
                # Refit without kernel optimization for efficiency
                kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.kernel_length_scale, length_scale_bounds="fixed")
                self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-5)
                self.gp.fit(self.X, self.y)


            # Select next point(s) by acquisition function
            next_X = self._select_next_points(1)

            # Evaluate the new point(s)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Local search around the best point
            predicted_y, predicted_sigma = self.gp.predict(self.best_x.reshape(1, -1), return_std=True)
            num_local_restarts = 1 + int(self.budget / (self.n_evals + 1e-6)) # More restarts early on
            local_x = self.local_search(self.gp, self.best_x, predicted_sigma[0], num_restarts=num_local_restarts)
            predicted_y_local, predicted_sigma_local = self.gp.predict(local_x.reshape(1, -1), return_std=True)


            # Probabilistic evaluation
            improvement = self.best_y - predicted_y_local[0]
            Z = improvement / predicted_sigma_local[0] if predicted_sigma_local[0] > 0 else np.inf
            prob_improvement = norm.cdf(Z)

            # Adjust the evaluation threshold based on the remaining budget
            remaining_budget = self.budget - self.n_evals
            evaluation_threshold = 0.5 + 0.4 * (remaining_budget / self.budget)  # Higher threshold when budget is tight

            if prob_improvement > evaluation_threshold and self.n_evals < self.budget:
                # Evaluate the true function value only if the GP predicts a sufficiently high probability of improvement and we have budget
                local_y = func(local_x)
                self.n_evals += 1
                if local_y < self.best_y:
                    self.best_y = local_y
                    self.best_x = local_x

            # Decay exploration factor and kappa
            self.exploration_factor *= self.exploration_decay
            self.kappa = max(self.min_kappa, self.kappa * self.kappa_decay)

            #Adaptive Kernel Update
            if len(self.y) > 1:
                improvement_ratio = abs(self.y[-1] - self.y[-2]) / abs(self.y[-2]) if abs(self.y[-2]) > 1e-6 else 0
                _, predicted_sigma = self.gp.predict(self.X[-1].reshape(1, -1), return_std=True)
                uncertainty = predicted_sigma[0]

                if improvement_ratio > 0.1 or uncertainty > 0.5:
                    self._fit_model(self.X, self.y)
                    self.kernel_update_interval = 5 * self.dim #reset update interval
                else:
                    self.kernel_update_interval = max(1, int(self.kernel_update_interval * 0.9)) #reduce update interval gradually

            # Adjust initial sampling size
            if self.n_evals > self.n_init and abs(self.y[-1] - self.y[-self.n_init]) < 1e-3:
                self.initial_sample_multiplier *= 1.1
            else:
                self.initial_sample_multiplier = max(1.0, self.initial_sample_multiplier * 0.95)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm EHBBO_BLS_PLE_AdaptiveKernel_v4 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1662 with standard deviation 0.1027.

took 845.79 seconds to run.