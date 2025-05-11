# Description
**ATREKVD_BO_EnhancedDiversity**: This algorithm builds upon ATREKVD_BO by introducing an enhanced diversity mechanism during point selection. It uses a dynamic diversity threshold that adapts not only to the trust region radius but also to the distribution of points within the trust region. Specifically, it calculates the average distance between points in the current trust region and uses this value, scaled by a factor, to adjust the diversity threshold. This helps to maintain a better balance between exploration and exploitation by preventing premature convergence in crowded regions of the search space. Additionally, the kernel optimization is enhanced by using a better initialization strategy based on the median distance between points.

# Justification
The key improvements are:

1.  **Enhanced Dynamic Diversity Threshold:** The diversity threshold is now dynamically adjusted based on both the trust region radius and the average distance between points within the trust region. This allows for a more adaptive diversity control, preventing premature convergence in crowded regions and promoting exploration in sparse regions. This addresses a potential weakness of the original algorithm where a fixed scaling of the diversity threshold based on the trust region radius might not be sufficient to maintain diversity in all situations.

2.  **Improved Kernel Initialization:** The kernel lengthscale is now initialized based on the median distance between points in the initial design. This provides a better starting point for kernel optimization, potentially leading to a more accurate GP model and improved performance.

These changes aim to improve the exploration-exploitation balance and the accuracy of the GP model, leading to better overall optimization performance.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

class ATREKVD_BO_EnhancedDiversity:
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
        self.ei_variance_scaling = 0.05
        self.recentering_threshold = 0.5
        self.length_scale = 1.0
        self.kernel_optim_interval = 5
        self.initial_diversity_threshold = 0.1
        self.diversity_threshold = self.initial_diversity_threshold
        self.min_decay = 0.8
        self.diversity_scaling = 0.5 # Scale average distance by this factor

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
        def obj(length_scale):
            kernel = ConstantKernel(constant_value=1.0, constant_value_bounds="fixed") * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 10))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-5)
            gp.fit(self.X, self.y)
            return -gp.log_marginal_likelihood()

        res = minimize(obj, x0=self.length_scale, method='L-BFGS-B', bounds=[(1e-2, 10)])
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

        selected_indices = np.argsort(ei.flatten())[-batch_size:]
        selected_points = candidates[selected_indices]

        if self.X is not None:
            # Calculate average distance between points in the trust region
            distances_within_tr = cdist(self.X, self.X)
            avg_distance = np.mean(distances_within_tr[np.triu_indices_from(distances_within_tr, k=1)])

            # Adapt diversity threshold
            self.diversity_threshold = self.initial_diversity_threshold * (self.trust_region_radius / 2.5) + self.diversity_scaling * avg_distance

            distances = cdist(selected_points, self.X)
            min_distances = np.min(distances, axis=1)
            selected_points = selected_points[min_distances > self.diversity_threshold]
            if len(selected_points) < batch_size:
                remaining_needed = batch_size - len(selected_points)
                additional_indices = np.argsort(ei.flatten())[:-batch_size-1:-1][:remaining_needed]
                additional_points = candidates[additional_indices]
                selected_points = np.concatenate([selected_points, additional_points], axis=0)

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

        # Initialize length_scale based on median distance
        distances = cdist(initial_X, initial_X)
        self.length_scale = np.median(distances[np.triu_indices_from(distances, k=1)])

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
            _, sigma = gp.predict(self.X, return_std=True)
            avg_sigma = np.mean(sigma)
            batch_size = min(self.budget - self.n_evals, int(np.ceil(5 * (avg_sigma / np.std(self.bounds))))) if np.std(self.bounds) > 0 else min(self.budget - self.n_evals, 5)
            batch_size = max(1, batch_size)

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

            # Adjust trust region radius based on EI and its variance
            if current_best_y < best_y:
                # Improvement: shrink radius, considering EI and variance
                decay_rate = self.radius_decay_base + self.ei_scaling * avg_ei - self.ei_variance_scaling * ei_variance
                decay_rate = max(decay_rate, self.min_decay)
                self.trust_region_radius *= decay_rate
                self.trust_region_center = current_best_x
                best_y = current_best_y
                best_x = current_best_x
            else:
                # No improvement: expand radius, but slower if EI is high
                self.trust_region_radius *= (self.radius_grow_base - self.ei_scaling * avg_ei + self.ei_variance_scaling * ei_variance)
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
 The algorithm ATREKVD_BO_EnhancedDiversity got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1888 with standard deviation 0.1192.

took 1153.25 seconds to run.