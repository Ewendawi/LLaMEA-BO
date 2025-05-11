# Description
**Adaptive Trust Region with Dynamic Kappa, Adaptive Radius and Stochastic Patch Bayesian Optimization (ATRBO_DKARSP)** combines the adaptive trust region (ATR) framework with dynamic kappa and adaptive radius from ATRBO_DKAI and ATRBO_DKAR, further enhancing it with a stochastic patch approach inspired by ATSPBO. It uses a dynamic adjustment of the exploration-exploitation trade-off parameter (kappa) and an adaptive strategy for adjusting the trust region radius (rho). To handle high dimensionality, it incorporates a stochastic patch, where a subset of dimensions is randomly selected for GP training and acquisition function evaluation. The trust region center is adapted after each iteration. If the new evaluation point improves the best objective, the trust region center is set as the evaluation point. This is beneficial when the next evaluation point is far from the current best location and has better performance. Also adds a stochasticity when updating the trust region radius to avoid premature convergence. The initial number of samples is made adaptive to the dimension of the search space. The kappa is adjusted dynamically based on the success of previous iterations. If the new point improves the best-seen value, kappa is decreased to promote exploitation. Conversely, if the new point does not improve the best-seen value, kappa is increased to encourage exploration.

# Justification
The combination of these strategies is justified as follows:

*   **Adaptive Trust Region (ATR):** Enables focused search within promising regions, enhancing convergence speed.
*   **Dynamic Kappa (DK):** Provides a flexible exploration-exploitation balance tailored to the optimization progress. When GP variance is high, exploration is encouraged, and vice versa.
*   **Adaptive Radius (AR):** Dynamically adjusts the trust region size based on the success rate, allowing for finer exploitation or broader exploration as needed.
*   **Stochastic Patch (SP):** Addresses the curse of dimensionality by reducing the computational cost of GP training and acquisition function evaluation, enabling efficient exploration in high-dimensional spaces. Project both the sampled candidate points *and* the training data to the stochastic patch for GP training and acquisition function evaluation to avoid errors.
*   **Trust Region Center Adaptation:** Adapt the trust region center after each iteration to improve the performance.
*   **Stochastic Trust Region Radius:** Add a stochasticity when updating the trust region radius to avoid premature convergence.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class ATRBO_DKARSP:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(20 * dim, self.budget // 5)
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.5
        self.rho = 0.95
        self.kappa = 2.0
        self.success_history = []

    def _sample_points(self, n_points, center=None, radius=None):
        if center is None:
            center = (self.bounds[1] + self.bounds[0]) / 2
        if radius is None:
            radius = np.max(self.bounds[1] - self.bounds[0]) / 2

        sampler = qmc.Sobol(d=self.dim, scramble=True)
        points = sampler.random(n=n_points)
        points = qmc.scale(points, -1, 1)
        lengths = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / lengths * np.random.uniform(0, 1, size=lengths.shape) ** (1 / self.dim)
        points = points * radius + center
        points = np.clip(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y, patch_indices):
        X_patched = X[:, patch_indices]
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X_patched, y)
        return gp

    def _acquisition_function(self, X, gp, patch_indices):
        X_patched = X[:, patch_indices]
        mu, sigma = gp.predict(X_patched, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        return mu - self.kappa * sigma

    def _select_next_point(self, gp, patch_indices):
        n_samples = 100 * self.dim
        samples = self._sample_points(n_samples, center=self.best_x, radius=self.trust_region_radius)
        acq_values = self._acquisition_function(samples[:, patch_indices], gp, patch_indices)
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
            self.best_x = self.X[idx].copy()

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial exploration
        n_uniform = self.n_init // 2
        n_around_best = self.n_init - n_uniform

        initial_X_uniform = self._sample_points(n_uniform)
        initial_X_best = self._sample_points(n_around_best, center=self.bounds[1]/2, radius=np.max(self.bounds[1] - self.bounds[0]) / 4) # Sampling around the middle of the search space as initial guess

        initial_X = np.vstack((initial_X_uniform, initial_X_best))
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        self.best_x = self.X[np.argmin(self.y)].copy()

        while self.n_evals < self.budget:
            # Stochastic Patch Selection
            remaining_evals = self.budget - self.n_evals
            patch_size = max(1, min(self.dim, int(self.dim * remaining_evals / self.budget) + 1))
            patch_indices = np.random.choice(self.dim, patch_size, replace=False)

            # Model Fitting
            gp = self._fit_model(self.X, self.y, patch_indices)

            # Next Point Selection
            next_x = self._select_next_point(gp, patch_indices)
            next_y = self._evaluate_points(func, next_x.reshape(1, -1))
            self._update_eval_points(next_x.reshape(1, -1), next_y)

            # Adjust trust region center
            if next_y < self.best_y:
                self.best_x = next_x.copy()

            # Adjust trust region radius and kappa
            if next_y < self.best_y:
                self.trust_region_radius /= self.rho
                self.kappa *= self.rho * 0.9
                self.success_history.append(True)
            else:
                self.trust_region_radius *= self.rho
                self.kappa /= (self.rho * 0.9)
                self.success_history.append(False)

            # Stochastic trust region update
            self.trust_region_radius *= np.random.uniform(0.9, 1.1) # Add stochasticity

            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)
            self.kappa = np.clip(self.kappa, 0.1, 10.0)

            # Adjust rho based on success history
            if len(self.success_history) > 10:
                success_rate = np.mean(self.success_history[-10:])
                self.rho = 0.9 + 0.09 * success_rate

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<ATRBO_DKARSP>", line 102, in __call__
 102->             next_x = self._select_next_point(gp, patch_indices)
  File "<ATRBO_DKARSP>", line 56, in _select_next_point
  56->         acq_values = self._acquisition_function(samples[:, patch_indices], gp, patch_indices)
  File "<ATRBO_DKARSP>", line 47, in _acquisition_function
  45 | 
  46 |     def _acquisition_function(self, X, gp, patch_indices):
  47->         X_patched = X[:, patch_indices]
  48 |         mu, sigma = gp.predict(X_patched, return_std=True)
  49 |         mu = mu.reshape(-1, 1)
IndexError: index 4 is out of bounds for axis 1 with size 4
