# Description
Adaptive Trust Region Stochastic Patch Bayesian Optimization (ATSPBO) combines the strengths of ATRBO and SPBO, addressing their weaknesses. It utilizes a trust region approach from ATRBO to focus the search around promising regions while incorporating stochastic patches from SPBO for efficient exploration in high-dimensional spaces. A key modification is that instead of training the GP on the full space and evaluating the acquisition function only on the patch (which is what SPBO was doing and produced an error), ATSPBO projects both the sampled candidate points *and* the training data to the stochastic patch for GP training and acquisition function evaluation. This resolves the dimensionality mismatch error. The trust region radius and patch size are dynamically adjusted based on the optimization progress and remaining budget. The acquisition function is Lower Confidence Bound (LCB) applied within the selected patch, allowing for balanced exploration and exploitation.

# Justification
- **Trust Region for Focused Search:** The trust region, borrowed from ATRBO, helps to narrow down the search space to regions that are likely to contain the optimum. This prevents the algorithm from wasting function evaluations in unpromising areas. This is particularly important as the dimensionality increases.
- **Stochastic Patches for Dimensionality Reduction:** SPBO's stochastic patches provide a computationally efficient way to handle high-dimensional problems. By focusing on random subsets of dimensions, ATSPBO can explore different subspaces without requiring an excessive number of function evaluations. The patch size is dynamically adjusted so that we use a large patch size for exploration at the beginning and a small patch size for exploitation at the end.
- **Dimensionality matching fix:** The training data is projected to the random patch before training the GP. And when it is time to evaluate the acquisition function, the candidate points are also projected to the random patch. By doing this, the dimensionality mismatch error is avoided.
- **Dynamic Adaptation of Trust Region and Patch Size:** The trust region radius and the patch size are dynamically adjusted throughout the optimization process. Initially the trust region is big and the patch size is also big, this gives room for exploration. As the budget runs out, the trust region shrinks and the patch size reduces.
- **LCB for Exploration-Exploitation Balance:** Lower Confidence Bound encourages both exploration (by considering the uncertainty) and exploitation (by considering the predicted mean).

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ATSPBO:
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
        self.rho = 0.95  # Shrinking factor
        self.kappa = 2.0  # Exploration-exploitation trade-off for LCB

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

            # Adjust trust region radius
            if next_y < self.best_y:
                self.trust_region_radius /= self.rho  # Expand
                self.kappa *= self.rho
            else:
                self.trust_region_radius *= self.rho  # Shrink
                self.kappa /= self.rho  # More exploration

            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)
            self.kappa = np.clip(self.kappa, 0.1, 10.0)

        return self.best_y, self.best_x
```