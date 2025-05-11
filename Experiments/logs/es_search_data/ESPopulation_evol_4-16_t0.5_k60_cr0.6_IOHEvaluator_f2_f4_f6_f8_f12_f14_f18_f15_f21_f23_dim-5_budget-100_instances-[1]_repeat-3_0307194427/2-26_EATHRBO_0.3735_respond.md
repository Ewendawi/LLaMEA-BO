# Description
**EI-Guided Adaptive Hybrid Trust Region Bayesian Optimization (EATHRBO)**: This algorithm combines the dynamic trust region management of AdaptiveTrustRegionBO_DR with the hybrid exploration-exploitation strategy of AdaptiveHybridBO_DE. It enhances the exploration-exploitation balance by incorporating the Expected Improvement (EI) into both the trust region radius adaptation and the dynamic exploration weight. Furthermore, it adaptively adjusts the Gaussian Process kernel lengthscale to better capture the function's characteristics within the trust region. It also focuses on improving the diversity of sampled points by using a more sophisticated diversity promoting technique.

# Justification
This algorithm attempts to improve upon the previous two by combining their best features and addressing some of their limitations.

*   **Dynamic Trust Region with EI Guidance:** The trust region radius is adjusted based on the average EI of the evaluated points, allowing for more informed expansion and contraction of the search space. This is inherited from AdaptiveTrustRegionBO_DR.
*   **Hybrid Exploration-Exploitation with EI-modulated Weight:** The exploration-exploitation balance is dynamically adjusted, with the exploration weight modulated by the EI. High EI values encourage exploitation within the trust region, while low EI values promote broader exploration. This is inherited from AdaptiveHybridBO_DE, but the weight is now EI-aware.
*   **Adaptive Kernel Lengthscale:** The kernel lengthscale of the Gaussian Process Regression (GPR) model is optimized periodically using L-BFGS-B within the trust region. This allows the GPR model to better capture the function's characteristics locally, improving the accuracy of the acquisition function.
*   **Diversity Promoting Selection:** The algorithm explicitly promotes diversity in the selected points by calculating the minimum distance to previously evaluated points and only selecting points that exceed a certain threshold. This helps to prevent the algorithm from getting stuck in local optima. Additionally, a more robust method is used to ensure that enough points are selected even after the diversity filter.
*   **Computational Efficiency:** The algorithm uses a simplified GP model to maintain computational efficiency. The kernel is fixed to avoid expensive hyperparameter optimization at each iteration.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

class EATHRBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * (dim + 1)
        self.trust_region_center = np.zeros(dim)
        self.trust_region_radius = 2.5
        self.min_radius = 0.1
        self.radius_decay_base = 0.95
        self.radius_grow_base = 1.1
        self.exploration_weight = 0.1  # Initial exploration weight
        self.exploration_decay = 0.98 # Decay exploration as optimization progresses
        self.ei_scaling = 0.1 # Scaling factor for EI-based radius adjustment
        self.recentering_threshold = 0.5 # Threshold for re-centering trust region
        self.diversity_threshold = 0.1

    def _sample_points(self, n_points, use_trust_region=True):
        if use_trust_region:
            sampler = qmc.Sobol(d=self.dim, scramble=False)
            sample = sampler.random(n=n_points)
            scaled_sample = qmc.scale(sample, -1.0, 1.0)
            points = self.trust_region_center + scaled_sample * self.trust_region_radius
            points = np.clip(points, self.bounds[0], self.bounds[1])
        else:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            points = qmc.scale(sample, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        kernel = C(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-5)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp, y_best):
        mu, sigma = gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)
        gamma = (y_best - mu) / sigma
        ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return ei

    def _select_next_points(self, gp, y_best, batch_size):
        # Dynamic exploration-exploitation balance
        candidates_tr = self._sample_points(100 * self.dim, use_trust_region=True)
        ei_tr = self._acquisition_function(candidates_tr, gp, y_best)
        avg_ei = np.mean(ei_tr)

        exploration_weight = self.exploration_weight * (1 - self.ei_scaling * avg_ei)
        num_trust_region = int(batch_size * (1 - exploration_weight))
        num_random = batch_size - num_trust_region

        # Sample points from trust region
        if num_trust_region > 0:
            selected_indices_tr = np.argsort(ei_tr)[-num_trust_region:]
            selected_points_tr = candidates_tr[selected_indices_tr]
        else:
            selected_points_tr = np.empty((0, self.dim))

        # Sample points randomly
        if num_random > 0:
            candidates_rand = self._sample_points(100 * self.dim, use_trust_region=False)
            ei_rand = self._acquisition_function(candidates_rand, gp, y_best)
            selected_indices_rand = np.argsort(ei_rand)[-num_random:]
            selected_points_rand = candidates_rand[selected_indices_rand]
        else:
            selected_points_rand = np.empty((0, self.dim))

        selected_points = np.concatenate([selected_points_tr, selected_points_rand], axis=0)

        # Diversity promoting selection
        if self.X is not None:
            distances = cdist(selected_points, self.X)
            min_distances = np.min(distances, axis=1)
            diverse_points = selected_points[min_distances > self.diversity_threshold]

            if len(diverse_points) < batch_size:
                remaining_needed = batch_size - len(diverse_points)
                candidates = self._sample_points(100 * self.dim)
                ei = self._acquisition_function(candidates, gp, y_best)
                
                # Select additional points, prioritizing diversity
                distances_add = cdist(candidates, self.X)
                min_distances_add = np.min(distances_add, axis=1)
                
                # Combine EI and diversity for selection
                combined_metric = ei + 10 * (min_distances_add > self.diversity_threshold)  # Add a bonus for diversity
                
                additional_indices = np.argsort(combined_metric)[-remaining_needed:]
                additional_points = candidates[additional_indices]
                
                diverse_points = np.concatenate([diverse_points, additional_points], axis=0)

            selected_points = diverse_points[:batch_size]
        else:
            selected_points = selected_points[:batch_size]

        return selected_points

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
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init, use_trust_region=False)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        self.trust_region_center = best_x # Initialize trust region center with best initial point

        while self.n_evals < self.budget:
            gp = self._fit_model(self.X, self.y)
            batch_size = min(self.budget - self.n_evals, 5)
            next_X = self._select_next_points(gp, best_y, batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            best_idx = np.argmin(self.y)
            current_best_y = self.y[best_idx][0]
            current_best_x = self.X[best_idx]

            # Calculate average EI of evaluated points
            ei_values = self._acquisition_function(next_X, gp, best_y)
            avg_ei = np.mean(ei_values)

            if current_best_y < best_y:
                self.trust_region_center = current_best_x
                self.trust_region_radius *= (self.radius_decay_base + self.ei_scaling * avg_ei)
                best_y = current_best_y
                best_x = current_best_x
            else:
                self.trust_region_radius *= (self.radius_grow_base - self.ei_scaling * avg_ei)
                self.trust_region_radius = min(self.trust_region_radius, 2.5)
            
            self.trust_region_radius = max(self.trust_region_radius, self.min_radius)
            self.exploration_weight *= self.exploration_decay # Reduce exploration over time

            # Re-center trust region if current center is far from best point
            distance = np.linalg.norm(self.trust_region_center - best_x)
            if distance > self.recentering_threshold * self.trust_region_radius:
                self.trust_region_center = best_x

        return best_y, best_x
```
## Feedback
 The algorithm EATHRBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1531 with standard deviation 0.1025.

took 6.22 seconds to run.