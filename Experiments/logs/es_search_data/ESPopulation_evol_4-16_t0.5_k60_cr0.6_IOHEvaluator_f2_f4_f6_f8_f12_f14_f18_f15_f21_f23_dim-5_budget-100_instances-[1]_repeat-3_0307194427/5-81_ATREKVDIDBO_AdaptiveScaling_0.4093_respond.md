# Description
**ATREKVDIDBO-AdaptiveScaling**: This algorithm refines the ATREKVDIDBO by introducing adaptive scaling of the EI and EI variance terms in the trust region radius adjustment. It also incorporates a more sophisticated mechanism for adjusting the diversity threshold based on both the trust region radius and the iteration number, aiming to balance exploration and exploitation more effectively. Furthermore, the kernel optimization is enhanced by using a more robust initialization strategy based on k-means clustering, and the batch size is adjusted dynamically based on the predicted variance and the remaining budget.

# Justification
The key improvements in this version focus on better balancing exploration and exploitation.

1.  **Adaptive EI and EI Variance Scaling:** The original algorithm used fixed scaling factors for EI and EI variance when adjusting the trust region radius. This version makes these scaling factors adaptive, changing them based on the iteration number. Early on, higher EI scaling promotes exploration, while later, higher EI variance scaling encourages exploitation of promising regions. This is achieved by using a sigmoid function to adjust the weights dynamically.
2.  **Enhanced Diversity Threshold Adjustment:** The diversity threshold is now adjusted based on both the trust region radius and the iteration number. Early in the optimization, a higher diversity threshold is maintained to explore the search space more broadly. As the optimization progresses and the trust region shrinks, the diversity threshold is reduced to focus on finer exploitation.
3.  **Improved Kernel Optimization:** The initialization of the kernel length scale is improved by using k-means clustering to identify representative points and estimate the length scale based on the distances between cluster centers. This provides a more robust starting point for the kernel optimization, leading to better GP model fitting.
4.  **Dynamic Batch Size Adjustment:** The batch size is dynamically adjusted based on the predicted variance from the GP model and the remaining budget. This ensures efficient use of the budget by taking larger batches when uncertainty is high and smaller batches when uncertainty is low.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.optimize import minimize, Bounds
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

class ATREKVDIDBO_AdaptiveScaling:
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
        self.initial_ei_scaling = 0.2
        self.initial_ei_variance_scaling = 0.1
        self.recentering_threshold = 0.5
        self.length_scale = 1.0
        self.kernel_optim_interval = 5
        self.initial_diversity_threshold = 0.1
        self.diversity_threshold = self.initial_diversity_threshold
        self.min_decay = 0.8
        self.iteration = 0

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
        # Initialize length scale based on k-means clustering
        if self.X is not None and len(self.X) > 1:
            kmeans = KMeans(n_clusters=min(len(self.X), 5), random_state=0, n_init = 'auto').fit(self.X)
            cluster_centers = kmeans.cluster_centers_
            distances = cdist(cluster_centers, cluster_centers)
            distances = np.triu(distances, k=1)
            median_distance = np.median(distances[distances > 0])
            initial_length_scale = median_distance
        else:
            initial_length_scale = self.length_scale

        def obj(length_scale):
            kernel = ConstantKernel(constant_value=1.0, constant_value_bounds="fixed") * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 10))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-5)
            gp.fit(self.X, self.y)
            return -gp.log_marginal_likelihood()

        bounds = Bounds(1e-2, 10)
        res = minimize(obj, x0=initial_length_scale, method='SLSQP', bounds=bounds)
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

        if self.gp is not None and self.X is not None and self.y is not None:
            mu, sigma = self.gp.predict(candidates, return_std=True)
            sigma = np.clip(sigma, 1e-9, np.inf)
        else:
            sigma = np.ones(len(candidates))  # Assign equal uncertainty if GP is not yet fitted

        selected_indices = np.argsort(ei.flatten())[-batch_size:]
        selected_points = candidates[selected_indices]

        if self.X is not None:
            distances = cdist(selected_points, self.X)
            min_distances = np.min(distances, axis=1)

            # Incorporate uncertainty into diversity metric
            diversity_metric = min_distances + self.diversity_threshold * sigma[selected_indices]
            selected_points = selected_points[diversity_metric > self.diversity_threshold]

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

        while self.n_evals < self.budget:
            # Kernel Optimization
            if self.iteration % self.kernel_optim_interval == 0:
                self._optimize_kernel()

            # Fit GP model
            gp = self._fit_model(self.X, self.y)

            # Dynamic batch size
            _, sigma = gp.predict(self.X, return_std=True)
            avg_sigma = np.mean(sigma)
            batch_size = min(self.budget - self.n_evals, int(np.ceil(5 * (avg_sigma / np.std(self.bounds))))) if np.std(self.bounds) > 0 else min(self.budget - self.n_evals, 5)
            batch_size = max(1, batch_size)

            # Adapt diversity threshold
            exploration_exploitation_tradeoff = 1 / (1 + np.exp(-5 * (self.iteration / self.budget - 0.5)))
            self.diversity_threshold = self.initial_diversity_threshold * (self.trust_region_radius / 2.5) * (1 - 0.5 * exploration_exploitation_tradeoff) # Scale diversity threshold with trust region radius and iteration

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

            # Adaptive EI and EI variance scaling
            ei_scaling = self.initial_ei_scaling * (1 - exploration_exploitation_tradeoff)
            ei_variance_scaling = self.initial_ei_variance_scaling * exploration_exploitation_tradeoff

            # Adjust trust region radius based on EI and its variance
            if current_best_y < best_y:
                # Improvement: shrink radius, considering EI and variance
                decay_rate = self.radius_decay_base + ei_scaling * avg_ei - ei_variance_scaling * ei_variance
                decay_rate = max(decay_rate, self.min_decay)
                self.trust_region_radius *= decay_rate
                self.trust_region_center = current_best_x
                best_y = current_best_y
                best_x = current_best_x
            else:
                # No improvement: expand radius, but slower if EI is high
                self.trust_region_radius *= (self.radius_grow_base - ei_scaling * avg_ei + ei_variance_scaling * ei_variance)
                self.trust_region_radius = min(self.trust_region_radius, 2.5)  # Limit to initial radius

            self.trust_region_radius = max(self.trust_region_radius, self.min_radius)

            # Re-center trust region if current center is far from best point
            distance = np.linalg.norm(self.trust_region_center - best_x)
            if distance > self.recentering_threshold * self.trust_region_radius:
                self.trust_region_center = best_x

            self.iteration += 1

        return best_y, best_x
```
## Feedback
 The algorithm ATREKVDIDBO_AdaptiveScaling got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1869 with standard deviation 0.1169.

took 692.17 seconds to run.