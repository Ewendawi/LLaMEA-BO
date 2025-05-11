# Description
**ATRBO-HDKRAES-EI: Adaptive Trust Region Bayesian Optimization with Hybrid Kappa, Dynamic Kappa Radius Adjustment, Enhanced Exploration, Success-rate-based Radius Scaling, and Expected Improvement Acquisition.** This algorithm builds upon ATRBO-HDKRAES by replacing the Lower Confidence Bound (LCB) acquisition function with Expected Improvement (EI). Additionally, we introduce a more sophisticated mechanism for adjusting the trust region radius based on the magnitude of improvement observed, and a dynamic adjustment of the exploration probability.

# Justification
*   **Expected Improvement (EI):** EI is generally less sensitive to the scaling of the GP variance and can provide a better balance between exploration and exploitation compared to LCB, especially in non-convex and multi-modal landscapes. It explicitly considers the potential for improvement over the current best value.
*   **Magnitude-Aware Radius Adjustment:** Instead of simply increasing/decreasing the radius based on success/failure, the radius adjustment is now proportional to the amount of improvement observed. Larger improvements lead to larger expansions, while smaller improvements lead to smaller expansions. This allows the algorithm to more effectively adapt to the local landscape.
*   **Dynamic Exploration Probability:** The exploration probability is now dynamically adjusted based on the success rate within the trust region. If the success rate is low, the exploration probability is increased to encourage exploration of new regions. If the success rate is high, the exploration probability is decreased to focus on exploitation within the current trust region.
*   **Clipping of Radius Adjustment:** To prevent overly aggressive expansion or shrinkage of the trust region, the radius adjustment factor is clipped to a reasonable range.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ATRBO_HDKRAES_EI:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0  # the number of function evaluations
        self.n_init = min(20 * dim, self.budget // 5)
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.5  # Initial trust region radius
        self.initial_trust_region_radius = 2.5 # Store initial radius
        self.min_trust_region_radius = 0.1 * self.initial_trust_region_radius # Minimum radius
        self.kappa = 2.0 + np.log(dim)  # Exploration-exploitation trade-off for LCB, adaptive to dimension
        self.success_count = 0
        self.failure_count = 0
        self.min_evals_for_adjust = 5 * dim
        self.rho = 0.95  # Shrinking factor for trust region
        self.kappa_success_failure_weight = 0.5  # Weight for combining success/failure and variance based kappa adjustment
        self.exploration_probability = 0.1  # Probability of sampling from a wider region
        self.initial_exploration_probability = 0.1
        self.success_history = []  # Keep track of successful moves
        self.success_rate_threshold = 0.7  # Threshold for aggressive expansion

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, center=None, radius=None, wider_region=False):
        # sample points within the trust region
        if center is None:
            center = (self.bounds[1] + self.bounds[0]) / 2
        if radius is None:
            radius = np.max(self.bounds[1] - self.bounds[0]) / 2

        if wider_region:
            radius = np.max(self.bounds[1] - self.bounds[0]) / 2  # Wider region for exploration

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

    def _sample_initial_points(self, n_points):
        # Use Latin Hypercube Sampling for initial exploration
        sampler = qmc.LatinHypercube(d=self.dim, seed=42)
        points = sampler.random(n=n_points)
        points = qmc.scale(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp):
        # Implement Expected Improvement acquisition function
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        improvement = self.best_y - mu
        Z = improvement / (sigma + 1e-9)  # Avoid division by zero
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        return -ei  # We want to minimize the negative EI

    def _select_next_point(self, gp):
        # Select the next point to evaluate within the trust region
        if np.random.rand() < self.exploration_probability:
            # Sample from a wider region
            n_samples = 100 * self.dim
            samples = self._sample_points(n_samples, center=self.best_x, radius=None, wider_region=True)
            acq_values = self._acquisition_function(samples, gp)
            best_index = np.argmin(acq_values)
            return samples[best_index]
        else:
            # Sample within the trust region
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
            self.best_x = self.X[idx].copy()

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop

        # Initial exploration
        initial_X = self._sample_initial_points(self.n_init)
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
                # Radius Adjustment (DKRA component)
                improvement = self.best_y - next_y[0][0]  # Calculate improvement
                if next_y < self.best_y:
                    self.success_count += 1
                    self.failure_count = 0
                    success_ratio = self.success_count / (self.success_count + self.failure_count + 1e-9)
                    radius_adj_factor = 1 + 0.2 * np.clip(improvement / (np.abs(self.best_y) + 1e-9), 0, 1)  # Clip for stability
                    self.trust_region_radius /= radius_adj_factor  # Expand
                    self.success_history.append(True)
                else:
                    self.failure_count += 1
                    self.success_count = 0
                    failure_ratio = self.failure_count / (self.success_count + self.failure_count + 1e-9)
                    radius_adj_factor = 1 + 0.2 * np.clip(np.abs(improvement) / (np.abs(self.best_y) + 1e-9), 0, 1) # Clip for stability
                    self.trust_region_radius *= radius_adj_factor  # Shrink
                    self.success_history.append(False)

                # Kappa Adjustment (Hybrid: DKRA + GP variance)
                mu, sigma = gp.predict(self.X, return_std=True)
                avg_sigma = np.mean(sigma)
                kappa_gp_component = 1.0 + np.log(1 + avg_sigma)  # Example GP variance component, can be tuned

                success_ratio = self.success_count / (self.success_count + self.failure_count + 1e-9)
                kappa_success_failure_component = (0.9 + 0.09 * success_ratio)

                self.kappa = (self.kappa_success_failure_weight * kappa_success_failure_component +
                               (1 - self.kappa_success_failure_weight) * kappa_gp_component)

                self.kappa = np.clip(self.kappa, 0.1, 10.0)
                
                # Adjust rho based on success history
                if len(self.success_history) > 10:
                    success_rate = np.mean(self.success_history[-10:])
                    self.rho = 0.9 + 0.09 * success_rate #adaptive rho. Higher success rate leads to higher rho, and thus slower shrinking.

                    # Aggressive expansion if success rate is high
                    if success_rate > self.success_rate_threshold:
                        self.trust_region_radius /= (self.rho * (1 + (success_rate - self.success_rate_threshold)))

                # Dynamic Exploration Probability Adjustment
                success_rate = np.mean(self.success_history[-min(10, len(self.success_history)):] if self.success_history else 0)
                self.exploration_probability = self.initial_exploration_probability * (1 - success_rate)  # Increase exploration if success is low
                self.exploration_probability = np.clip(self.exploration_probability, 0.01, 0.5)  # Clip for stability

            # Trust Region Center Update
            if next_y < self.best_y:
                self.best_x = next_x.copy()  # Adapt trust region center
                self.best_y = next_y[0][0]
                
            self.trust_region_radius = np.clip(self.trust_region_radius, self.min_trust_region_radius, np.max(self.bounds[1] - self.bounds[0]) / 2)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ATRBO_HDKRAES_EI got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1858 with standard deviation 0.1136.

took 179.98 seconds to run.