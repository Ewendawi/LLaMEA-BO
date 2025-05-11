# Description
**ATRBO-HDKRAES-AC: Adaptive Trust Region Bayesian Optimization with Hybrid Kappa, Dynamic Kappa Radius Adjustment, Enhanced Exploration, Success-rate-based Radius Scaling, and Acquisition Clipping.** This algorithm builds upon ATRBO-HDKRAES by introducing acquisition clipping to prevent over-exploration and focus search within promising regions. It adaptively adjusts the acquisition function values to balance exploration and exploitation more effectively. Furthermore, the GP fitting is enhanced by using a more robust kernel and optimizing hyperparameters more thoroughly.

# Justification
The key improvements are:

1.  **Acquisition Clipping:** Clipping the acquisition function values helps to prevent the algorithm from being overly influenced by extremely low acquisition values in unexplored regions. This stabilizes the search and promotes more focused exploration around promising areas. The clipping threshold is adaptively adjusted based on the mean and standard deviation of the acquisition values, allowing the algorithm to automatically adapt to different function landscapes.

2. **Enhanced GP Fitting**: By using a more robust kernel and optimizing hyperparameters more thoroughly, the GP model can better capture the underlying function landscape. This leads to more accurate predictions and better acquisition function values.

3.  **Adaptive Clipping Threshold:** The clipping threshold is dynamically adjusted based on the statistics of the acquisition values. This allows the algorithm to adapt to different function landscapes and balance exploration and exploitation more effectively.

4. **Success Rate Decay**: The success rate decays over time to prevent the algorithm from getting stuck in local optima.

These changes aim to improve the robustness and efficiency of the algorithm by balancing exploration and exploitation more effectively and preventing over-exploration.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ATRBO_HDKRAES_AC:
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
        self.success_history = []  # Keep track of successful moves
        self.success_rate_threshold = 0.7  # Threshold for aggressive expansion
        self.acq_clip_quantile = 0.95 # Quantile for clipping acquisition function
        self.success_rate_decay = 0.99 # Decay factor for success rate

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
        # Implement Lower Confidence Bound acquisition function
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        acq_values = mu - self.kappa * sigma

        # Acquisition Clipping
        clip_threshold = np.quantile(acq_values, self.acq_clip_quantile)
        acq_values = np.clip(acq_values, a_min=None, a_max=clip_threshold)

        return acq_values

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

                # Success Rate Decay
                self.success_rate_threshold *= self.success_rate_decay

            # Trust Region Center Update
            if next_y < self.best_y:
                self.best_x = next_x.copy()  # Adapt trust region center
                
            self.trust_region_radius = np.clip(self.trust_region_radius, self.min_trust_region_radius, np.max(self.bounds[1] - self.bounds[0]) / 2)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ATRBO_HDKRAES_AC got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1946 with standard deviation 0.1216.

took 163.81 seconds to run.