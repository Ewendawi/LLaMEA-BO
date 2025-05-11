# Description
**Adaptive Trust Region with Stochastic Patch and Dynamic Kappa-Radius Adjustment Bayesian Optimization (ATSPDKRBO)** combines the strengths of ATRBO_DKRA and ATSPBO, incorporating dynamic trust region adjustments, stochastic patches, and adaptive kappa. The core idea is to use stochastic patches for efficient exploration, especially in high dimensions, while leveraging dynamic trust region adjustments based on success/failure rates to balance exploration and exploitation. This involves training the GP on stochastically selected patches and dynamically adapting the kappa parameter (exploration-exploitation trade-off) and trust region radius based on the optimization's recent performance. A minimum number of evaluations before adjusting the trust region is maintained to avoid premature convergence. The trust region center is adapted after each iteration. A stochastic factor is added to the trust region radius.

# Justification
The algorithm builds upon ATRBO_DKRA's adaptive trust region and dynamic kappa/radius adjustments, which have demonstrated improved performance and runtime. The incorporation of stochastic patches from ATSPBO tackles high dimensionality more effectively by focusing the Gaussian process model on a subset of dimensions, reducing computational cost and improving model accuracy. The main changes are:

1.  **Stochastic Patches**: The Gaussian process is trained and the acquisition function is evaluated on a randomly selected subset of dimensions (the "patch"). This addresses the curse of dimensionality and accelerates computation. The patch size is dynamically adjusted.
2.  **Dynamic Kappa and Radius Adaptation**: The kappa (exploration-exploitation) and trust region radius are adjusted based on the recent success/failure rate, as in ATRBO_DKRA.
3.  **Trust Region Center Adaptation**: The trust region center is updated with better points.
4.  **Stochastic Trust Region Radius**: Added stochasticity when updating the trust region radius to avoid premature convergence.
The combination allows for more effective exploration in high-dimensional spaces while maintaining a focused search within promising regions, leading to a more robust and efficient optimization process.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ATSPDKRBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(10 * dim, self.budget // 5)
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.5  # Initial trust region radius
        self.kappa = 2.0  # Exploration-exploitation trade-off for LCB
        self.success_count = 0
        self.failure_count = 0
        self.min_evals_for_adjust = 5 * dim
        self.kappa = 2.0 + np.log(dim)  # Adaptive Initial Kappa
        self.rho = 0.95 # Shrinking factor

    def _sample_points(self, n_points, center=None, radius=None):
        if center is None:
            center = (self.bounds[1] + self.bounds[0]) / 2
        if radius is None:
            radius = np.max(self.bounds[1] - self.bounds[0]) / 2

        sampler = qmc.Sobol(d=self.dim, scramble=True)
        points = sampler.random(n=n_points)
        points = qmc.scale(points, -1, 1)  # Scale to [-1, 1]

        lengths = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / lengths * np.random.uniform(0, 1, size=lengths.shape) ** (1/self.dim)

        points = points * radius + center
        points = np.clip(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y, patch_indices):
        # Fit the model on the stochastic patch
        X_patched = X[:, patch_indices]
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X_patched, y)
        return gp

    def _acquisition_function(self, X, gp, patch_indices):
        # LCB within the patch
        X_patched = X[:, patch_indices]
        mu, sigma = gp.predict(X_patched, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        return mu - self.kappa * sigma

    def _select_next_point(self, gp, patch_indices):
        n_samples = 100 * self.dim
        samples = self._sample_points(n_samples, center=self.best_x, radius=self.trust_region_radius)
        acq_values = self._acquisition_function(samples, gp, patch_indices)
        best_index = np.argmin(acq_values)
        return samples[best_index]

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

        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        self.best_x = self.X[np.argmin(self.y)].copy()

        while self.n_evals < self.budget:
            # Dynamic patch size
            remaining_evals = self.budget - self.n_evals
            patch_size = max(1, min(self.dim, int(self.dim * remaining_evals / self.budget) + 1))
            patch_indices = np.random.choice(self.dim, patch_size, replace=False)

            # Fit model
            gp = self._fit_model(self.X, self.y, patch_indices)

            # Select next point
            next_x = self._select_next_point(gp, patch_indices)
            next_y = self._evaluate_points(func, next_x.reshape(1, -1))
            self._update_eval_points(next_x.reshape(1, -1), next_y)

            # Adjust trust region radius and kappa
            if self.n_evals > self.n_init + self.min_evals_for_adjust:
                if next_y < self.best_y:
                    self.success_count += 1
                    self.failure_count = 0
                    success_ratio = self.success_count / (self.success_count + self.failure_count + 1e-9)
                    self.trust_region_radius /= (0.9 + 0.09 * success_ratio)  # Expand faster with higher success
                    self.kappa *= (0.9 + 0.09 * success_ratio) # Less exploration
                    self.best_x = next_x.copy()
                else:
                    self.failure_count += 1
                    self.success_count = 0
                    failure_ratio = self.failure_count / (self.success_count + self.failure_count + 1e-9)
                    self.trust_region_radius *= (0.9 + 0.09 * failure_ratio)  # Shrink faster with higher failure
                    self.kappa /= (0.9 + 0.09 * failure_ratio) # More exploration

                # Stochastic trust region radius
                self.trust_region_radius *= np.random.uniform(0.95, 1.05)
                self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)
                self.kappa = np.clip(self.kappa, 0.1, 10.0)

        return self.best_y, self.best_x
```