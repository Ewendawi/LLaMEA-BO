# Description
**ATRKTEIBO-VarianceAware**: This algorithm refines the ATRKTEIBO algorithm by incorporating variance awareness into the trust region radius adjustment and batch size selection. Specifically, it considers the variance of the Expected Improvement (EI) values in addition to the average EI when adjusting the trust region radius. A high EI variance suggests potential for exploration, leading to a slower radius decay or faster radius growth. The batch size is also adjusted based on EI variance, increasing the batch size when the EI variance is high to promote exploration and decreasing it when the EI variance is low to focus on exploitation. Furthermore, the kernel optimization is enhanced using a more robust optimization algorithm (SLSQP) and a better initialization strategy based on the median distance between points.

# Justification
The key improvements are:

1.  **Variance-Aware Trust Region Adjustment:** The original ATRKTEIBO uses only the average EI for adjusting the trust region radius. Incorporating the variance of EI provides a more nuanced approach to balancing exploration and exploitation. High EI variance suggests that the GP model is uncertain about the potential improvements in the trust region, warranting a slower radius decay (or faster growth) to encourage exploration.

2.  **Variance-Aware Batch Size Selection:** Similar to the trust region radius adjustment, the batch size is now adjusted based on the EI variance. This allows for more efficient use of the evaluation budget. When the EI variance is high, a larger batch size is used to explore more of the trust region. When the EI variance is low, a smaller batch size is used to focus on exploiting the most promising areas.

3.  **Improved Kernel Optimization:** The original kernel optimization uses L-BFGS-B. Switching to SLSQP can potentially lead to better solutions, especially when the objective function is complex. Furthermore, initializing the length scale with the median distance between data points provides a better starting point for the optimization, potentially leading to faster convergence and better kernel parameter estimates.

4. **Enhanced Diversity:** A more aggressive diversity mechanism is added, using the EI values to filter points that are too close to existing data points, further promoting exploration.

These changes aim to improve the algorithm's ability to balance exploration and exploitation, adapt to the characteristics of the objective function, and efficiently utilize the evaluation budget.

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

class ATRKTEIBO_VarianceAware:
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
        self.ei_scaling = 0.1
        self.recentering_threshold = 0.5
        self.length_scale = 1.0
        self.kernel_optim_interval = 5
        self.diversity_threshold = 0.1

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

        # Enhanced diversity mechanism using EI
        if self.X is not None:
            distances = cdist(candidates, self.X)
            min_distances = np.min(distances, axis=1)
            # Filter candidates based on both distance and EI
            diversity_mask = min_distances > self.diversity_threshold
            ei_threshold = np.quantile(ei, 0.25)  # Consider only top 75% EI values
            ei_mask = ei.flatten() >= ei_threshold
            selected_candidates = candidates[diversity_mask & ei_mask]
            selected_ei = ei[diversity_mask & ei_mask]

            if len(selected_candidates) < batch_size:
                # If not enough diverse candidates, add some more based on EI only
                remaining_needed = batch_size - len(selected_candidates)
                additional_indices = np.argsort(ei.flatten())[::-1][:remaining_needed]
                additional_candidates = candidates[additional_indices]
                selected_candidates = np.concatenate([selected_candidates, additional_candidates], axis=0)
                selected_ei = np.concatenate([selected_ei, ei[additional_indices]], axis=0)
            candidates = selected_candidates
            ei = selected_ei

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
            ei_values = self._acquisition_function(self.X)
            ei_variance = np.var(ei_values)
            avg_sigma = np.mean(sigma)
            batch_size = min(self.budget - self.n_evals, int(np.ceil(5 * (avg_sigma / np.std(self.bounds))))) if np.std(self.bounds) > 0 else min(self.budget - self.n_evals, 5)
            # Adjust batch size based on EI variance
            batch_size = int(batch_size * (1 + self.ei_scaling * ei_variance))
            batch_size = max(1, min(batch_size, self.budget - self.n_evals))

            # Select next points
            next_X = self._select_next_points(batch_size)

            # Evaluate points
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Update best solution
            best_idx = np.argmin(self.y)
            current_best_y = self.y[best_idx][0]
            current_best_x = self.X[best_idx]

            # Calculate average EI and variance of evaluated points
            ei_values = self._acquisition_function(next_X)
            avg_ei = np.mean(ei_values)
            ei_variance = np.var(ei_values)

            # Adjust trust region radius based on EI and EI variance
            if current_best_y < best_y:
                # Improvement: shrink radius based on EI, slower if EI variance is high
                self.trust_region_radius *= (self.radius_decay_base + self.ei_scaling * avg_ei) * (1 - self.ei_scaling * ei_variance)
                self.trust_region_center = current_best_x
                best_y = current_best_y
                best_x = current_best_x
            else:
                # No improvement: expand radius, but slower if EI is high, faster if EI variance is high
                self.trust_region_radius *= (self.radius_grow_base - self.ei_scaling * avg_ei) * (1 + self.ei_scaling * ei_variance)
                self.trust_region_radius = min(self.trust_region_radius, 2.5)  # Limit to initial radius

            self.trust_region_radius = max(self.trust_region_radius, self.min_radius)

            # Re-center trust region if current center is far from best point
            distance = np.linalg.norm(self.trust_region_center - best_x)
            if distance > self.recentering_threshold * self.trust_region_radius:
                self.trust_region_center = best_x

            iteration += 1

        return best_y, best_x
```
## Feedback
 The algorithm ATRKTEIBO_VarianceAware got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1897 with standard deviation 0.1135.

took 305.43 seconds to run.