# Description
**Adaptive Trust Region with EI-based Radius and Dynamic Exploration Bayesian Optimization (ATREID-BO)**: This algorithm combines the strengths of AdaptiveTrustRegionBO_DR and AdaptiveHybridBO_DE. It uses a dynamic trust region, where the radius is adjusted based on the Expected Improvement (EI) values, similar to AdaptiveTrustRegionBO_DR. It also incorporates a dynamic exploration strategy, balancing trust region exploitation with broader exploration based on the uncertainty of the GP model, similar to AdaptiveHybridBO_DE. The GP model uses a fixed kernel for computational efficiency. A diversity-promoting selection strategy is used to avoid premature convergence. Additionally, the algorithm introduces a mechanism to dynamically adjust the exploration weight based on the success rate of the trust region exploitation. If the trust region consistently yields improvements, the exploration weight is decreased, and vice versa.

# Justification
The algorithm combines the strengths of the two best-performing algorithms.
- **Dynamic Radius Adjustment based on EI:** The radius of the trust region is adjusted based on the average EI of the evaluated points within the trust region. This allows for more aggressive shrinking when significant improvements are observed and slower expansion when improvements are marginal.
- **Dynamic Exploration:** The algorithm balances trust region exploitation with broader exploration based on the uncertainty of the GP model. This helps to avoid premature convergence and to explore the search space more effectively.
- **Fixed Kernel:** A fixed kernel is used for the GP model to reduce computational cost.
- **Diversity-Promoting Selection:** A diversity-promoting selection strategy is used to avoid premature convergence.
- **Dynamic Exploration Weight Adjustment:** The exploration weight is dynamically adjusted based on the success rate of the trust region exploitation. This allows the algorithm to adapt to the characteristics of the optimization problem and to balance exploration and exploitation more effectively.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.spatial.distance import cdist

class ATREID_BO:
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
        self.exploration_increase = 1.02
        self.ei_scaling = 0.1 # Scaling factor for EI-based radius adjustment
        self.recentering_threshold = 0.5 # Threshold for re-centering trust region
        self.success_threshold = 0.2 # Threshold for considering trust region exploitation successful
        self.success_rate = 0.0

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
        num_trust_region = int(batch_size * (1 - self.exploration_weight))
        num_random = batch_size - num_trust_region

        # Sample points from trust region
        if num_trust_region > 0:
            candidates_tr = self._sample_points(100 * self.dim, use_trust_region=True)
            ei_tr = self._acquisition_function(candidates_tr, gp, y_best)
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

        if self.X is not None:
            distances = cdist(selected_points, self.X)
            min_distances = np.min(distances, axis=1)
            selected_points = selected_points[min_distances > 0.1] # Enforce diversity

            if len(selected_points) < batch_size:
                remaining_needed = batch_size - len(selected_points)
                candidates = self._sample_points(100 * self.dim)
                ei = self._acquisition_function(candidates, gp, y_best)
                additional_indices = np.argsort(ei)[:-batch_size-1:-1][:remaining_needed]
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
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init, use_trust_region=False)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        self.trust_region_center = best_x # Initialize trust region center with best initial point

        successful_iterations = 0
        total_iterations = 0

        while self.n_evals < self.budget:
            gp = self._fit_model(self.X, self.y)
            batch_size = min(self.budget - self.n_evals, 5)
            next_X = self._select_next_points(gp, best_y, batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            best_idx = np.argmin(self.y)
            current_best_y = self.y[best_idx][0]
            current_best_x = self.X[best_idx]

            improvement = current_best_y < best_y
            if improvement:
                self.trust_region_center = current_best_x
                best_y = current_best_y
                best_x = current_best_x
                successful_iterations += 1
            
            total_iterations += 1

            # Calculate average EI of evaluated points
            ei_values = self._acquisition_function(next_X, gp, best_y)
            avg_ei = np.mean(ei_values)

            # Adjust trust region radius based on EI and improvement
            if improvement:
                self.trust_region_radius *= (self.radius_decay_base + self.ei_scaling * avg_ei)
            else:
                self.trust_region_radius *= (self.radius_grow_base - self.ei_scaling * avg_ei)
                self.trust_region_radius = min(self.trust_region_radius, 2.5)
            
            self.trust_region_radius = max(self.trust_region_radius, self.min_radius)

            # Re-center trust region if current center is far from best point
            distance = np.linalg.norm(self.trust_region_center - best_x)
            if distance > self.recentering_threshold * self.trust_region_radius:
                self.trust_region_center = best_x

            # Adjust exploration weight based on success rate
            if total_iterations > 5:
                self.success_rate = successful_iterations / total_iterations
                if self.success_rate > self.success_threshold:
                    self.exploration_weight *= self.exploration_decay
                else:
                    self.exploration_weight *= self.exploration_increase
                self.exploration_weight = np.clip(self.exploration_weight, 0.05, 0.5)
                successful_iterations = 0
                total_iterations = 0

        return best_y, best_x
```
## Feedback
 The algorithm ATREID_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1707 with standard deviation 0.0979.

took 6.56 seconds to run.