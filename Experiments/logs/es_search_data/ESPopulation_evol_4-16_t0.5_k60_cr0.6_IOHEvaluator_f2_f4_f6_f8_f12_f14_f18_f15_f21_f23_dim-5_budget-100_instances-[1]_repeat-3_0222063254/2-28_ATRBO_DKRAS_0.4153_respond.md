# Description
**ATRBO-DKRAS**: Adaptive Trust Region Bayesian Optimization with Dynamic Kappa, Radius Adjustment, and Stochastic Center Refinement. This algorithm builds upon the strengths of ATRBO-DKRA and ATRBO-DKAR by combining dynamic adjustment of the exploration-exploitation trade-off (kappa), adaptive trust region radius adjustment based on success/failure, and incorporates a novel stochastic refinement of the trust region center to escape local optima. Specifically, the algorithm blends the success/failure based radius and kappa adaptation from DKRA with the dynamic rho of DKAR, and adds a new mechanism to randomly relocate the trust region center when stagnation is detected, facilitating broader exploration.

# Justification
The algorithm incorporates the following key improvements:

1.  **Dynamic Kappa and Radius Adjustment (DKRA Component):** As in ATRBO-DKRA, the exploration-exploitation trade-off (kappa) and trust region radius are dynamically adjusted based on the recent history of successful and unsuccessful steps. This allows for a responsive adaptation to the local landscape.
2. **Adaptive Rho (DKAR Component):** The shrinking factor `rho` for the trust region is adapted based on the success rate within a window of recent iterations. This allows finer control over the trust region size based on how fruitful the region has been.
3.  **Stochastic Center Refinement (SCR):** When the optimization stagnates (i.e., a lack of improvement in the best objective value over a certain number of iterations), the trust region center is stochastically relocated within the bounds. This allows the algorithm to escape local optima and explore new regions of the search space. The magnitude of the relocation is proportional to the trust region radius and diminishes over time. The stagnation is identified based on the moving average of the objective function value and best_y.
4. **Combined Success/Failure Adaptation:** The algorithm combines the success/failure adaptation approach from DKRA with the adaptive rho from DKAR.

The motivation for these changes is to improve the algorithm's ability to balance exploration and exploitation and escape local optima, leading to better overall performance across the BBOB test suite.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class ATRBO_DKRAS:
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
        self.trust_region_radius = 2.5
        self.kappa = 2.0
        self.success_count = 0
        self.failure_count = 0
        self.min_evals_for_adjust = 5 * dim
        self.stagnation_counter = 0
        self.max_stagnation = 10 * dim
        self.rho = 0.95  # Initial rho for radius adjustment
        self.success_history = []
        self.success_window = 5
        self.rng = np.random.RandomState(42)
        self.obj_history = [] # Store objective function history
        self.history_length = 10

    def _sample_points(self, n_points, center=None, radius=None):
        if center is None:
            center = (self.bounds[1] + self.bounds[0]) / 2
        if radius is None:
            radius = np.max(self.bounds[1] - self.bounds[0]) / 2

        sampler = qmc.Sobol(d=self.dim, scramble=True)
        points = sampler.random(n=n_points)
        points = qmc.scale(points, -1, 1)

        lengths = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / lengths * self.rng.uniform(0, 1, size=lengths.shape) ** (1 / self.dim)

        points = points * radius + center
        points = np.clip(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp):
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        return mu - self.kappa * sigma

    def _select_next_point(self, gp):
        n_samples = 100 * self.dim
        samples = self._sample_points(n_samples, center=self.best_x, radius=self.trust_region_radius)
        acq_values = self._acquisition_function(samples, gp)
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

    def _stochastic_center_refinement(self):
        # Stochastically relocate the trust region center
        move = self.rng.uniform(-1, 1, size=self.dim) * self.trust_region_radius * 0.5 # Reduced factor. Original 1.0
        new_center = self.best_x + move
        new_center = np.clip(new_center, self.bounds[0], self.bounds[1])
        return new_center

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        self.best_x = self.X[np.argmin(self.y)].copy()

        while self.n_evals < self.budget:
            # Optimization
            gp = self._fit_model(self.X, self.y)
            next_x = self._select_next_point(gp)
            next_y = self._evaluate_points(func, next_x.reshape(1, -1))
            self._update_eval_points(next_x.reshape(1, -1), next_y)

            # Update success history
            success = next_y < self.best_y
            self.success_history.append(success[0])
            if len(self.success_history) > self.success_window:
                self.success_history = self.success_history[-self.success_window:]

            # Adapt kappa based on GP variance within trust region
            mu, sigma = gp.predict(self._sample_points(100, center=self.best_x, radius=self.trust_region_radius), return_std=True)
            avg_sigma = np.mean(sigma)
            self.kappa = np.clip(self.kappa * (1 + avg_sigma), 0.1, 10.0)

            # Adapt rho based on success rate
            success_rate = np.mean(self.success_history) if self.success_history else 0.5
            self.rho = np.clip(0.9 + (0.5 - success_rate) / 5, 0.7, 0.99)

            # Adjust trust region radius and kappa
            if next_y < self.best_y:
                self.success_count += 1
                self.failure_count = 0
                success_ratio = self.success_count / (self.success_count + self.failure_count + 1e-9)
                self.trust_region_radius /= (self.rho + self.rng.normal(0, 0.01))  # Success expands, modified with rho
                self.kappa *= (0.9 + 0.09 * success_ratio) # Less exploration
                self.best_x = next_x.copy()  #Move the trust region if there is success
                self.stagnation_counter = 0
            else:
                self.failure_count += 1
                self.success_count = 0
                failure_ratio = self.failure_count / (self.success_count + self.failure_count + 1e-9)
                self.trust_region_radius *= (self.rho + self.rng.normal(0, 0.01))  # Failure shrinks, modified with rho
                self.kappa /= (0.9 + 0.09 * failure_ratio)  # More exploration
                self.stagnation_counter += 1

            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)
            self.kappa = np.clip(self.kappa, 0.1, 10.0)

            # Stochastic Center Refinement
            self.obj_history.append(next_y[0][0])
            if len(self.obj_history) > self.history_length:
                self.obj_history.pop(0)
                
            if len(self.obj_history) == self.history_length:
                moving_avg = np.mean(self.obj_history)
                if abs(moving_avg - self.best_y) < 1e-5: # Check for small objective function value changes. Increased the tolerance
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0

            if self.stagnation_counter > self.max_stagnation:
                self.best_x = self._stochastic_center_refinement()
                self.stagnation_counter = 0  # Reset counter after refinement

        return self.best_y, self.best_x
```