# Description
**Adaptive Trust Region with Dynamic Kappa, Adaptive Radius and Stochastic Patch Bayesian Optimization (ATRBO_DKARSP)**. This algorithm combines the strengths of ATRBO_DKAI and ATSPBO, aiming to achieve a balance between global exploration and local exploitation, while also addressing the computational cost associated with high dimensionality. It incorporates adaptive trust region management (radius and center), dynamic adjustment of the exploration-exploitation trade-off (kappa), a dynamic stochastic patch strategy for efficient high-dimensional exploration and utilizes success-rate based dynamic adjustment of the trust region shrinking factor `rho`.

# Justification
*   **Trust Region and Adaptive Radius:** ATRBO effectively focuses the search in promising regions. Adaptive adjustment of the trust region radius (`rho`) based on the success rate of previous steps allows for efficient exploration and exploitation. The stochastic patch strategy allows the algorithm to explore more efficiently in high-dimensional spaces by only considering a subset of dimensions for GP training and acquisition function evaluation in each iteration.
*   **Dynamic Kappa:** Adaptive adjustment of `kappa` balances exploration and exploitation. The tuning of kappa is essential for adapting to different stages of the optimization process.
*   **Stochastic Patches (borrowed from ATSPBO):** This reduces the computational burden of GP training and acquisition function evaluation in high dimensions by operating on a random subset of dimensions. The patch size is dynamically adjusted based on remaining budget.
*   **Adaptive Trust Region Center:** The trust region center adapts after each iteration. If the new evaluation point improves the best objective, the trust region center is set as the evaluation point.
*   **Computational efficiency:** Projecting the data to a random subspace and using LCB as the acquisition function allows for faster computation of the next point.
*   **Initial Exploration Strategy:** Combining uniform sampling with sampling around the center to encourage faster initial convergence.

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
            self.best_x = self.X[idx].copy()

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:

        # Initial exploration: Combine uniform sampling with sampling around the best seen point
        n_uniform = self.n_init // 2
        n_around_best = self.n_init - n_uniform

        initial_X_uniform = self._sample_points(n_uniform)
        initial_X_best = self._sample_points(n_around_best, center=self.bounds[1] / 2,
                                              radius=np.max(self.bounds[1] - self.bounds[0]) / 4)  # Sampling around the middle of the search space as initial guess

        initial_X = np.vstack((initial_X_uniform, initial_X_best))
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Initialize best_x to a random initial point
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
            if next_y < self.best_y:
                self.trust_region_radius /= self.rho  # Expand
                self.kappa *= self.rho * 0.9  # Reduced kappa decrease.
                self.success_history.append(True)
                self.best_x = next_x.copy() # Adapt trust region center
            else:
                self.trust_region_radius *= self.rho  # Shrink
                self.kappa /= (self.rho * 0.9)  # increase kappa more when unsuccessful
                self.success_history.append(False)

            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2,
                                                np.max(self.bounds[1] - self.bounds[0]) / 2)
            self.kappa = np.clip(self.kappa, 0.1, 10.0)

            # Adjust rho based on success history
            if len(self.success_history) > 10:
                success_rate = np.mean(self.success_history[-10:])
                self.rho = 0.9 + 0.09 * success_rate  # adaptive rho. Higher success rate leads to higher rho, and thus slower shrinking.

        return self.best_y, self.best_x
```
## Error
 [Errno Evaluation timed out (%d)] 1800