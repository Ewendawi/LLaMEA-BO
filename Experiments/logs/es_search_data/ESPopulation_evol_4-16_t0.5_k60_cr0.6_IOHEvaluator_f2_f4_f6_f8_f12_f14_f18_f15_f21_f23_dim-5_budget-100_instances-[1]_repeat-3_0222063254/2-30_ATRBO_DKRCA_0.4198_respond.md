# Description
**Adaptive Trust Region Bayesian Optimization with Dynamic Kappa, Radius, and Center Adjustment (ATRBO_DKRCA)**

This algorithm integrates successful components from ATRBO_DKAI and ATRBO_DKRA to enhance the Adaptive Trust Region Bayesian Optimization framework. Key features include:

1.  **Dynamic Kappa:** Adapts the exploration-exploitation trade-off (`kappa`) based on both the success history of recent iterations and the GP's uncertainty (variance) within the trust region, similar to ATRBO_DKAR.
2.  **Dynamic Radius Adjustment:** Adapts the trust region radius based on a success/failure ratio, as in ATRBO_DKRA, providing responsive adaptation to the local landscape.
3.  **Trust Region Center Adaptation:** Shifts the trust region center to the new evaluation point if it improves the best objective value, enhancing exploitation of promising regions, as in ATRBO_DKAR.
4.  **Adaptive Initial Exploration:** Combines uniform sampling with sampling around a central point to encourage faster initial convergence, inspired by ATRBO_DKAI.
5.  **Minimum Evaluations for Adjustment:** Ensures a minimum number of evaluations before adjusting the trust region radius and `kappa` to prevent premature convergence, similar to ATRBO_DKRA.

By combining these strategies, ATRBO_DKRCA aims to achieve a better balance between exploration and exploitation, leading to more efficient optimization.

# Justification
The combination of elements from ATRBO_DKAI and ATRBO_DKRA is motivated by the following observations:

*   ATRBO_DKAI's adaptive initial exploration helps to quickly identify promising regions, while its dynamic `kappa` adjustment based on success history facilitates the exploration-exploitation balance. However, the fixed shrinking factor `rho` can limit its adaptability.
*   ATRBO_DKRA's dynamic radius adjustment provides a more responsive adaptation to the local landscape. However, it lacks the uncertainty-aware `kappa` adjustment and the adaptive initial exploration of ATRBO_DKAI.
*   ATRBO_DKAR's Trust Region Center Adaptation is beneficial when the next evaluation point is far from the current best location and has better performance, preventing premature convergence in some functions

By incorporating these elements, ATRBO_DKRCA aims to combine the strengths of both algorithms and overcome their weaknesses, leading to a more robust and efficient optimization strategy. Additionally, we introduce a dynamic `kappa` adjustment that considers the GP's uncertainty to further enhance exploration-exploitation balance. The minimum evaluations before adjusting parameters also prevents premature convergence.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ATRBO_DKRCA:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0  # the number of function evaluations
        self.n_init = min(20 * dim, self.budget // 5)  # increased samples for initial exploration
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.5  # Initial trust region radius
        self.kappa = 2.0  # Exploration-exploitation trade-off for LCB
        self.success_count = 0
        self.failure_count = 0
        self.min_evals_for_adjust = 5 * dim
        self.success_history = []

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, center=None, radius=None):
        # sample points within the trust region
        if center is None:
            center = (self.bounds[1] + self.bounds[0]) / 2
        if radius is None:
            radius = np.max(self.bounds[1] - self.bounds[0]) / 2

        sampler = qmc.Sobol(d=self.dim, scramble=True)
        points = sampler.random(n=n_points)
        points = qmc.scale(points, -1, 1)  # Scale to [-1, 1]

        # Project points to a hypersphere
        lengths = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / lengths * np.random.uniform(0, 1, size=lengths.shape) ** (1 / self.dim)

        points = points * radius + center

        # Clip to the bounds
        points = np.clip(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp):
        # Implement Lower Confidence Bound acquisition function
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        # Dynamic Kappa based on GP variance
        kappa = self.kappa + np.mean(sigma)  # Increased exploration with high variance
        return mu - kappa * sigma

    def _select_next_point(self, gp):
        # Select the next point to evaluate within the trust region
        n_samples = 100 * self.dim
        samples = self._sample_points(n_samples, center=self.best_x, radius=self.trust_region_radius)
        acq_values = self._acquisition_function(samples, gp)
        best_index = np.argmin(acq_values)
        return samples[best_index]

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop

        # Initial exploration: Combine uniform sampling with sampling around the best seen point
        n_uniform = self.n_init // 2
        n_around_best = self.n_init - n_uniform

        initial_X_uniform = self._sample_points(n_uniform)
        initial_X_best = self._sample_points(n_around_best, center=self.bounds[1]/2, radius=np.max(self.bounds[1] - self.bounds[0]) / 4)  # Sampling around the middle

        initial_X = np.vstack((initial_X_uniform, initial_X_best))
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Initialize best_x to a random initial point
        self.best_x = self.X[np.argmin(self.y)].copy()

        while self.n_evals < self.budget:
            # Optimization
            gp = self._fit_model(self.X, self.y)

            # Select next point
            next_x = self._select_next_point(gp)
            next_y = self._evaluate_points(func, next_x.reshape(1, -1))
            self._update_eval_points(next_x.reshape(1, -1), next_y)

            # Adjust trust region radius and kappa
            if self.n_evals > self.n_init + self.min_evals_for_adjust:
                if next_y < self.best_y:
                    self.success_count += 1
                    self.failure_count = 0
                    success_ratio = self.success_count / (self.success_count + self.failure_count + 1e-9)
                    self.trust_region_radius /= (0.9 + 0.09 * success_ratio)  # Expand faster with higher success
                    self.success_history.append(True)
                else:
                    self.failure_count += 1
                    self.success_count = 0
                    failure_ratio = self.failure_count / (self.success_count + self.failure_count + 1e-9)
                    self.trust_region_radius *= (0.9 + 0.09 * failure_ratio)  # Shrink faster with higher failure
                    self.success_history.append(False)

                self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)

            # Adjust kappa based on success history
            if len(self.success_history) > 10:
                success_rate = np.mean(self.success_history[-10:])
                self.kappa = 2.0 - 1.9 * success_rate #adaptive kappa. Higher success rate leads to lower kappa, and thus less exploration.

            self.kappa = np.clip(self.kappa, 0.1, 10.0)

            # Trust Region Center Adaptation
            if next_y < self.best_y:
                self.best_x = next_x  # Move trust region center

        return self.best_y, self.best_x
```