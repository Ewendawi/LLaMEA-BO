# Description
**Adaptive Trust Region with Variance-Informed Exploration and Lengthscale Adaptation (ATREK_VIOLA-BO)**: This algorithm builds upon ATREKBO_VEB and ATREKVDIBO by refining the trust region management and point selection strategies. It introduces a more sophisticated mechanism for adjusting the trust region radius based on both EI and EI variance, employing separate scaling factors for exploration and exploitation. Furthermore, it incorporates a dynamic diversity threshold that adapts to the trust region radius and the distribution of EI values, preventing premature convergence while maintaining exploration. A new lengthscale adaptation strategy is introduced, which uses the variance of the EI values to guide the adjustment of the kernel lengthscale, promoting faster learning and improved performance.

# Justification
The algorithm combines the strengths of ATREKBO_VEB and ATREKVDIBO, particularly their adaptive trust region management, kernel tuning, and dynamic batch sizing. The key improvements are:

1.  **Refined Trust Region Radius Adjustment:** The trust region radius adjustment is modified to use separate scaling factors for EI and EI variance when growing and shrinking the radius. This allows for a more nuanced control over exploration and exploitation.

2.  **Dynamic Diversity Threshold:** The diversity threshold is dynamically adjusted based on the trust region radius and the distribution of EI values. This helps to prevent premature convergence in small trust regions while maintaining exploration in larger regions.

3. **Variance-Informed Lengthscale Adaptation:** The kernel lengthscale is adapted based on the variance of the EI values. High EI variance suggests that the GP model is uncertain about the function landscape, and the lengthscale is increased to promote exploration. Low EI variance suggests that the GP model is confident about the function landscape, and the lengthscale is decreased to promote exploitation. This allows for faster learning and improved performance.

These changes are designed to address the limitations of the previous algorithms and improve their ability to handle a wide range of black box optimization problems.

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

class ATREK_VIOLABO:
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
        self.ei_scaling_decay = 0.1
        self.ei_scaling_grow = 0.05
        self.ei_variance_scaling_decay = 0.05
        self.ei_variance_scaling_grow = 0.01
        self.recentering_threshold = 0.5
        self.length_scale = 1.0
        self.kernel_optim_interval = 5
        self.diversity_threshold_base = 0.1
        self.min_decay = 0.8
        self.lengthscale_adapt_rate = 0.1

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

        # Dynamic diversity threshold
        diversity_threshold = self.diversity_threshold_base * self.trust_region_radius * (1 + np.std(ei))
        selected_indices = np.argsort(ei.flatten())[-batch_size:]
        selected_points = candidates[selected_indices]

        if self.X is not None:
            distances = cdist(selected_points, self.X)
            min_distances = np.min(distances, axis=1)
            selected_points = selected_points[min_distances > diversity_threshold]
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
                decay_rate = self.radius_decay_base + self.ei_scaling_decay * avg_ei - self.ei_variance_scaling_decay * ei_variance
                decay_rate = max(decay_rate, self.min_decay)  # Ensure a minimum decay
                self.trust_region_radius *= decay_rate
                self.trust_region_center = current_best_x
                best_y = current_best_y
                best_x = current_best_x
            else:
                # No improvement: expand radius, but slower if EI is high
                self.trust_region_radius *= (self.radius_grow_base - self.ei_scaling_grow * avg_ei + self.ei_variance_scaling_grow * ei_variance)
                self.trust_region_radius = min(self.trust_region_radius, 2.5)  # Limit to initial radius

            self.trust_region_radius = max(self.trust_region_radius, self.min_radius)

            # Re-center trust region if current center is far from best point
            distance = np.linalg.norm(self.trust_region_center - best_x)
            if distance > self.recentering_threshold * self.trust_region_radius:
                self.trust_region_center = best_x

            # Adapt lengthscale based on EI variance
            self.length_scale *= np.exp(self.lengthscale_adapt_rate * (ei_variance - 0.1))  # Adjust towards a target variance of 0.1
            self.length_scale = np.clip(self.length_scale, 1e-2, 10)

            iteration += 1

        return best_y, best_x
```
## Feedback
 The algorithm ATREK_VIOLABO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1819 with standard deviation 0.1062.

took 548.65 seconds to run.