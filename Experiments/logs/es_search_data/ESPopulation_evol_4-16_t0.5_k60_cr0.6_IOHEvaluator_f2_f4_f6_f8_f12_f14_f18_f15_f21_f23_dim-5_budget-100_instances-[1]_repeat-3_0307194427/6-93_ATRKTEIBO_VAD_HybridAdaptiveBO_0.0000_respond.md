# Description
**ATRKTEIBO_VAD_HybridAdaptiveBO**: This algorithm builds upon the strengths of ATRKTEIBO_VAD_Hybrid and ATRKTEIBO_VAD_AdaptiveBatching, incorporating adaptive trust region management, variance-aware radius control, EI-based diversity, dynamic EI scaling, and adaptive batch sizing. It introduces a more refined trust region adaptation strategy based on the success ratio of evaluations within the trust region and the EI variance and GP uncertainty. The batch size is dynamically adjusted based on both GP uncertainty and EI variance. The diversity mechanism combines distance-based and EI-based selection, with the diversity threshold adapted based on the trust region radius and iteration. Kernel optimization is performed periodically using SLSQP. A key addition is the incorporation of a success ratio based trust region adaptation, which helps in balancing exploration and exploitation more effectively.

# Justification
The algorithm combines several successful strategies:
1.  **Hybrid Diversity:** Combines distance-based and EI-based diversity to balance exploration and exploitation.
2.  **Adaptive Trust Region:** Adjusts the trust region radius based on EI, EI variance, GP uncertainty, and a success ratio of evaluations within the trust region. A success ratio is computed based on whether the new points added to the dataset improved the current best value.
3.  **Adaptive Batch Sizing:** Dynamically adjusts the batch size based on GP uncertainty (sigma) and EI variance.
4.  **Dynamic EI Scaling:** Adjusts the EI scaling factor during the optimization process.
5.  **Kernel Optimization:** Periodically optimizes the kernel hyperparameters using SLSQP to improve the GP model's accuracy.
6.  **Success Ratio Based Trust Region Adaptation:** This allows for a more aggressive reduction of the trust region if the evaluations within the trust region consistently lead to improvements, and a slower reduction or expansion if improvements are rare.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.optimize import minimize, Bounds
from scipy.spatial.distance import cdist, pdist, squareform

class ATRKTEIBO_VAD_HybridAdaptiveBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * (dim + 1)
        self.trust_region_center = np.zeros(dim)
        self.trust_region_radius = 2.5
        self.min_radius = 0.1
        self.radius_decay_base = 0.95
        self.radius_grow_base = 1.1
        self.gp = None
        self.ei_scaling_init = 0.2  # Initial EI scaling
        self.ei_scaling_final = 0.01 # Final EI scaling
        self.ei_variance_scaling = 0.05
        self.recentering_threshold = 0.5
        self.length_scale = 1.0
        self.kernel_optim_interval = 5
        self.diversity_threshold_base = 0.05 # Base diversity threshold
        self.diversity_weight = 0.5  # Weight for distance in diversity calculation
        self.min_decay = 0.8
        self.success_ratio = 0.0
        self.success_history = []
        self.success_window = 10

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        sample = sampler.random(n=n_points)
        scaled_sample = qmc.scale(sample, -1.0, 1.0)
        points = self.trust_region_center + scaled_sample * self.trust_region_radius
        points = np.clip(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * \
                 RBF(length_scale=self.length_scale, length_scale_bounds=(1e-3, 1e3)) + \
                 WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-7, 1e-3))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        self.gp.fit(X, y)
        return self.gp

    def _optimize_kernel(self):
        # Initialize length_scale with median distance
        distances = pdist(self.X)
        if len(distances) > 0:
            self.length_scale = np.median(distances)
        else:
            self.length_scale = 1.0

        def obj(length_scale):
            kernel = ConstantKernel(constant_value=1.0, constant_value_bounds="fixed") * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 10))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-5)
            gp.fit(self.X, self.y)
            return -gp.log_marginal_likelihood()

        bounds = Bounds(1e-2, 10)
        res = minimize(obj, x0=self.length_scale, method='SLSQP', bounds=bounds)
        self.length_scale = res.x[0]

    def _acquisition_function(self, X):
        if self.gp is None or self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)
        y_best = np.min(self.y)
        gamma = (y_best - mu) / sigma
        ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(100 * self.dim)
        ei = self._acquisition_function(candidates)

        if self.X is not None:
            distances = cdist(candidates, self.X)
            min_distances = np.min(distances, axis=1)
            normalized_ei = (ei - np.min(ei)) / (np.max(ei) - np.min(ei) + 1e-9)  # Normalize EI
            normalized_distances = min_distances / self.trust_region_radius  # Normalize distances

            # Hybrid diversity metric
            diversity_metric = self.diversity_weight * normalized_distances + (1 - self.diversity_weight) * normalized_ei.flatten()

            # Select top candidates based on diversity metric
            selected_indices = np.argsort(diversity_metric)[-batch_size:]
            selected_points = candidates[selected_indices]

            if len(selected_points) < batch_size:
                remaining_needed = batch_size - len(selected_points)
                additional_indices = np.argsort(ei.flatten())[:-batch_size-1:-1][:remaining_needed]
                additional_points = candidates[additional_indices]
                selected_points = np.concatenate([selected_points, additional_points], axis=0)

        else:
            selected_indices = np.argsort(ei.flatten())[-batch_size:]
            selected_points = candidates[selected_indices]

        return selected_points[:batch_size]

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        iteration = 0
        while self.n_evals < self.budget:
            # Kernel Optimization
            if iteration % self.kernel_optim_interval == 0:
                self._optimize_kernel()

            # Fit GP model
            gp = self._fit_model(self.X, self.y)

            # Dynamic batch size
            mu, sigma = gp.predict(self.X, return_std=True)
            avg_sigma = np.mean(sigma)
            ei_values = self._acquisition_function(self.X)
            ei_variance = np.var(ei_values)

            batch_size = min(self.budget - self.n_evals, int(np.ceil(5 * (avg_sigma / np.std(self.bounds))))) if np.std(self.bounds) > 0 else min(self.budget - self.n_evals, 5)
            # Adjust batch size based on EI variance
            batch_size = int(batch_size * (1 + self.ei_variance_scaling * ei_variance))
            batch_size = max(1, batch_size)

            # Adaptive EI scaling
            ei_scaling = self.ei_scaling_init + (self.ei_scaling_final - self.ei_scaling_init) * (iteration / self.budget)

            # Select next points
            next_X = self._select_next_points(batch_size)

            # Evaluate points
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Update best solution
            best_idx = np.argmin(self.y)
            current_best_y = self.y[best_idx][0]
            current_best_x = self.X[best_idx]

            # Calculate EI values and statistics
            ei_values = self._acquisition_function(next_X)
            avg_ei = np.mean(ei_values)
            ei_variance = np.var(ei_values)

            # Update success history
            if current_best_y < best_y:
                self.success_history.append(1)
            else:
                self.success_history.append(0)

            if len(self.success_history) > self.success_window:
                self.success_history.pop(0)

            self.success_ratio = np.mean(self.success_history) if self.success_history else 0.0

            # Adjust trust region radius based on EI and its variance and GP uncertainty and success ratio
            if current_best_y < best_y:
                # Improvement: shrink radius, considering EI, variance and success ratio
                decay_rate = self.radius_decay_base + ei_scaling * avg_ei - self.ei_variance_scaling * ei_variance - ei_scaling * avg_sigma + 0.1 * self.success_ratio
                decay_rate = max(decay_rate, self.min_decay)  # Ensure a minimum decay
                self.trust_region_radius *= decay_rate
                self.trust_region_center = current_best_x
                best_y = current_best_y
                best_x = current_best_x
            else:
                # No improvement: expand radius, but slower if EI is high
                self.trust_region_radius *= (self.radius_grow_base - ei_scaling * avg_ei + self.ei_variance_scaling * ei_variance + ei_scaling * avg_sigma - 0.1 * self.success_ratio)
                self.trust_region_radius = min(self.trust_region_radius, 2.5)  # Limit to initial radius

            self.trust_region_radius = max(self.trust_region_radius, self.min_radius)

            # Re-center trust region if current center is far from best point
            distance = np.linalg.norm(self.trust_region_center - best_x)
            if distance > self.recentering_threshold * self.trust_region_radius:
                self.trust_region_center = best_x

            iteration += 1

        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<ATRKTEIBO_VAD_HybridAdaptiveBO>", line 161, in __call__
 161->             next_y = self._evaluate_points(func, next_X)
  File "<ATRKTEIBO_VAD_HybridAdaptiveBO>", line 113, in _evaluate_points
 113->         y = np.array([func(x) for x in X])
  File "<ATRKTEIBO_VAD_HybridAdaptiveBO>", line 113, in <listcomp>
 111 | 
 112 |     def _evaluate_points(self, func, X):
 113->         y = np.array([func(x) for x in X])
 114 |         self.n_evals += len(X)
 115 |         return y.reshape(-1, 1)
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
