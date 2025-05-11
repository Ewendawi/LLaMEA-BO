# Description
**ATRKTEIBO_HybridV2: Adaptive Trust Region with Kernel Tuning, EI-Variance Aware Dynamics, Distance-Uncertainty Diversity, and Adaptive EI-Sigma Balancing.** This algorithm builds upon ATRKTEIBO_VAD_Hybrid and ATRKTEIBO_VADID_BO, integrating adaptive trust region management, kernel tuning, EI-variance aware dynamics, and a diversity mechanism that considers both distance and uncertainty. A key enhancement is the adaptive balancing of EI and GP uncertainty (sigma) in the point selection process. It dynamically adjusts the weight given to EI and sigma based on the iteration number and the current trust region radius, promoting exploration in early iterations and exploitation in later iterations. The kernel optimization uses SLSQP and initializes the length scale based on the median distance between points. Dynamic batch sizing based on GP uncertainty is also included.

# Justification
This algorithm combines the strengths of the two parent algorithms, ATRKTEIBO_VAD_Hybrid and ATRKTEIBO_VADID_BO, while addressing their potential weaknesses.

*   **Adaptive EI-Sigma Balancing:** The balance between exploration and exploitation is crucial in BO. By adaptively weighting EI and sigma, the algorithm can explore the search space more effectively in early iterations (when sigma is high) and exploit promising regions in later iterations (when EI is more reliable). This adaptive balancing is controlled by `ei_sigma_tradeoff`, which is dynamically adjusted based on the iteration number and trust region radius.
*   **Hybrid Diversity with Uncertainty:** The diversity mechanism considers both the distance to existing points and the predicted uncertainty (sigma) from the GP model. This helps to select points that are not only diverse but also located in regions of high uncertainty, promoting exploration.
*   **Adaptive Trust Region:** The trust region radius is adjusted based on the success of previous evaluations and the EI variance. This helps to focus the search on promising regions while avoiding premature convergence.
*   **Kernel Tuning:** Periodic kernel optimization helps to maintain the accuracy of the GP model, which is essential for effective BO. Initializing the length scale with the median distance between points provides a good starting point for the optimization.
*   **Dynamic Batch Sizing:** Adjusting the batch size based on GP uncertainty allows the algorithm to make efficient use of the available budget.

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

class ATRKTEIBO_HybridV2:
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
        self.initial_diversity_threshold = 0.1
        self.diversity_threshold = self.initial_diversity_threshold
        self.min_decay = 0.8
        self.ei_sigma_tradeoff = 0.5 # Initial tradeoff between EI and Sigma

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
        # Initialize length scale based on median distance
        if self.X is not None and len(self.X) > 1:
            distances = cdist(self.X, self.X)
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

    def _select_next_points(self, batch_size, iteration):
        candidates = self._sample_points(100 * self.dim)
        ei = self._acquisition_function(candidates)

        if self.gp is not None and self.X is not None and self.y is not None:
            mu, sigma = self.gp.predict(candidates, return_std=True)
            sigma = np.clip(sigma, 1e-9, np.inf)

            # Normalize EI and Sigma
            normalized_ei = (ei - np.min(ei)) / (np.max(ei) - np.min(ei) + 1e-9)
            normalized_sigma = sigma / np.max(sigma)

            # Adaptive EI-Sigma Tradeoff
            self.ei_sigma_tradeoff = 0.2 + 0.8 * (1 - (iteration / self.budget)) * (self.trust_region_radius / 2.5) # favor sigma early on, and EI later

            # Combined metric
            combined_metric = self.ei_sigma_tradeoff * normalized_ei.flatten() + (1 - self.ei_sigma_tradeoff) * normalized_sigma

            # Incorporate diversity
            if self.X is not None:
                distances = cdist(candidates, self.X)
                min_distances = np.min(distances, axis=1)
                diversity_metric = min_distances + self.diversity_threshold * sigma
                combined_metric += 0.1 * diversity_metric # Add a small weight to diversity

            selected_indices = np.argsort(combined_metric)[-batch_size:]
            selected_points = candidates[selected_indices]
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
            _, sigma = gp.predict(self.X, return_std=True)
            avg_sigma = np.mean(sigma)
            batch_size = min(self.budget - self.n_evals, int(np.ceil(5 * (avg_sigma / np.std(self.bounds))))) if np.std(self.bounds) > 0 else min(self.budget - self.n_evals, 5)
            batch_size = max(1, batch_size)

            # Adapt diversity threshold
            self.diversity_threshold = self.initial_diversity_threshold * (self.trust_region_radius / 2.5) # Scale diversity threshold with trust region radius

            # Adaptive EI scaling
            ei_scaling = self.ei_scaling_init + (self.ei_scaling_final - self.ei_scaling_init) * (iteration / self.budget)

            # Select next points
            next_X = self._select_next_points(batch_size, iteration)

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

            # Adjust trust region radius based on EI and EI variance
            ei_weight = 0.7
            variance_weight = 0.3
            if current_best_y < best_y:
                # Improvement: shrink radius based on EI, slower if EI variance is high
                self.trust_region_radius *= (self.radius_decay_base + ei_scaling * avg_ei * ei_weight) * (1 - ei_scaling * ei_variance * variance_weight)
                self.trust_region_center = current_best_x
                best_y = current_best_y
                best_x = current_best_x
            else:
                # No improvement: expand radius, but slower if EI is high, faster if EI variance is high
                self.trust_region_radius *= (self.radius_grow_base - ei_scaling * avg_ei * ei_weight) * (1 + ei_scaling * ei_variance * variance_weight)
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
 The algorithm ATRKTEIBO_HybridV2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1858 with standard deviation 0.1147.

took 1321.09 seconds to run.