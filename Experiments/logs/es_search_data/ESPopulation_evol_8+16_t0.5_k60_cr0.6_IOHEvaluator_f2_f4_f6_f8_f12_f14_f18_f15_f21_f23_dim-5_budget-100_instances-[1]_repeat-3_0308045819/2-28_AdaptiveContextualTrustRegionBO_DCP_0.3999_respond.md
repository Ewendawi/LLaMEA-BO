# Description
**Adaptive Contextual Trust Region Bayesian Optimization with Dynamic Context Penalty (ACTRBO-DCP):** This algorithm enhances Adaptive Contextual Trust Region Bayesian Optimization (ACTRBO) by introducing a dynamic context penalty. The context penalty, which penalizes points close to existing points, is adaptively adjusted based on the optimization progress and the trust region's success. This dynamic adjustment helps to balance exploration and exploitation more effectively. Additionally, the kernel is optimized using L-BFGS-B.

# Justification
The key improvement is the dynamic context penalty. In the original ACTRBO, the `context_penalty` was fixed. However, the optimal penalty might change during the optimization process. By dynamically adjusting the penalty, we can encourage exploration in the early stages and exploitation in later stages, or vice versa, depending on the success of the trust region. The adjustment is based on the `success_ratio` of the trust region. A high success ratio indicates that the trust region is in a promising area, and the penalty should be increased to encourage exploitation. A low success ratio indicates that the trust region is not promising, and the penalty should be decreased to encourage exploration. The kernel optimization allows the GP to better capture the underlying function's characteristics.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize

class AdaptiveContextualTrustRegionBO_DCP:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(5*dim, self.budget//10)

        self.gp = None
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0
        self.radius_decay = 0.95
        self.radius_increase = 1.1
        self.min_radius = 0.1
        self.success_ratio = 0.0
        self.random_restart_prob = 0.05
        self.knn = NearestNeighbors(n_neighbors=5)
        self.context_penalty = 0.1
        self.initial_context_penalty = 0.1
        self.context_penalty_decay = 0.95
        self.context_penalty_increase = 1.05
        self.min_context_penalty = 0.01
        self.max_context_penalty = 1.0

    def _sample_points(self, n_points):
        if self.best_x is None or np.random.rand() < self.random_restart_prob:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            points = []
            while len(points) < n_points:
                sample = np.random.uniform(-1, 1, size=(self.dim,))
                sample = self.best_x + self.trust_region_radius * sample
                sample = np.clip(sample, self.bounds[0], self.bounds[1])
                points.append(sample)
            return np.array(points)

    def _fit_model(self, X, y):
        # Kernel optimization
        def neg_log_likelihood(theta):
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=theta, length_scale_bounds="fixed")
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
            gp.fit(X, y)
            return -gp.log_marginal_likelihood()

        # Initial guess for kernel parameters
        initial_length_scale = 1.0

        # Optimize kernel parameters using L-BFGS-B
        result = minimize(neg_log_likelihood, initial_length_scale, method='L-BFGS-B', bounds=[(1e-5, 10.0)])
        optimized_length_scale = result.x[0]

        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=optimized_length_scale, length_scale_bounds="fixed")
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        self.gp.fit(X, y)
        self.knn.fit(X)
        return self.gp

    def _acquisition_function(self, X):
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            mu, sigma = self.gp.predict(X, return_std=True)
            sigma = np.clip(sigma, 1e-9, np.inf)
            imp = self.best_y - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

            distances, _ = self.knn.kneighbors(X)
            context_penalty = np.mean(distances, axis=1).reshape(-1, 1)
            acquisition = ei - self.context_penalty * sigma * context_penalty
            return acquisition.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        n_candidates = min(100 * self.dim, self.budget - self.n_evals)
        candidates = self._sample_points(n_candidates)
        acquisition_values = self._acquisition_function(candidates)
        
        # Corrected point selection to avoid IndexError
        if n_candidates > 0:
            best_indices = np.argsort(acquisition_values.flatten())[-batch_size:]
            best_indices = best_indices[best_indices < n_candidates]  # Ensure indices are within bounds
            if len(best_indices) > 0:
                return candidates[best_indices]
            else:
                # If no valid indices are found, return a random candidate
                return candidates[np.random.randint(0, n_candidates, size=1)]
        else:
            # If no candidates are generated, return a random point within bounds
            return np.random.uniform(self.bounds[0], self.bounds[1], size=(1, self.dim))

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]
            self.success_ratio = 1.0
        else:
            self.success_ratio *= 0.75

    def _adjust_trust_region(self):
        if self.success_ratio > 0.5:
            self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)

    def _adjust_context_penalty(self):
        if self.success_ratio > 0.5:
            self.context_penalty = min(self.context_penalty * self.context_penalty_increase, self.max_context_penalty)
        else:
            self.context_penalty = max(self.context_penalty * self.context_penalty_decay, self.min_context_penalty)

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        batch_size = min(1, self.dim)
        while self.n_evals < self.budget:
            self._fit_model(self.X, self.y)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self._adjust_trust_region()
            self._adjust_context_penalty()

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AdaptiveContextualTrustRegionBO_DCP got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1758 with standard deviation 0.1008.

took 102.92 seconds to run.