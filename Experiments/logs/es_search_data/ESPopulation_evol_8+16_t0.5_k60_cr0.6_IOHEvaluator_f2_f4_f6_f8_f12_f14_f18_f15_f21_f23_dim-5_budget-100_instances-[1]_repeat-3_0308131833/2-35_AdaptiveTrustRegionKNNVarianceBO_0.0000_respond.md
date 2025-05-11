# Description
**Adaptive Trust Region with KNN-guided Variance Exploration (ATRKVO):** This algorithm combines the strengths of AdaptiveTrustRegionBO and HybridTrustRegionKNNBO. It leverages the adaptive scaling of Expected Improvement (EI) based on predictive variance from AdaptiveTrustRegionBO, the KNN-based exploration from HybridTrustRegionKNNBO, and integrates them within a trust region framework. Additionally, it incorporates a more sophisticated mechanism to balance exploration and exploitation using a dynamic weighting scheme that considers both the GP's predictive variance and the KNN distance. The local search radius is also adaptively adjusted based on the success rate.

# Justification
This algorithm builds upon the existing AdaptiveTrustRegionBO and HybridTrustRegionKNNBO by integrating their key features and refining the exploration-exploitation balance.

*   **Adaptive Variance Scaling:** The EI is scaled by the GP's predictive variance, focusing exploration on uncertain regions.
*   **KNN-based Exploration:** KNN provides a surrogate-model-free exploration term, especially useful when the GP is inaccurate.
*   **Trust Region:** The trust region limits the search space, ensuring that the GP model remains a reasonable approximation of the true function.
*   **Dynamic Weighting:** The weights for EI and KNN exploration are dynamically adjusted based on the GP's predictive variance and the KNN distance to the nearest neighbors. High variance and large KNN distance imply greater uncertainty, leading to increased KNN exploration.
*   **Adaptive Local Search:** The local search radius is dynamically adjusted based on the success rate of previous local searches, enabling finer exploitation or broader exploration.

This combination aims to provide a robust and efficient optimization algorithm that can effectively handle a wide range of black-box optimization problems.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors

class AdaptiveTrustRegionKNNVarianceBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.gp = None
        self.trust_region_radius = 2.0
        self.trust_region_shrink = 0.5
        self.trust_region_expand = 1.5
        self.success_threshold = 0.75
        self.failure_threshold = 0.25
        self.trust_region_center = np.zeros(dim)
        self.best_x = None
        self.best_y = np.inf
        self.local_search_radius = 0.1
        self.local_search_success_rate = 0.0
        self.local_search_success_memory = []
        self.local_search_success_window = 5
        self.knn = NearestNeighbors(n_neighbors=5)
        self.ei_weight = 0.5
        self.knn_weight = 0.5

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        scaled_sample = qmc.scale(sample, -self.trust_region_radius, self.trust_region_radius)
        return scaled_sample + self.trust_region_center

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        if self.gp is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.gp.predict(X, return_std=True)
        improvement = self.best_y - mu
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0
        ei = ei.reshape(-1, 1)
        ei = ei * sigma  # Adaptive variance scaling

        # KNN-based exploration
        distances, indices = self.knn.kneighbors(X)
        min_distances = distances[:, 0].reshape(-1, 1)
        knn_exploration = min_distances

        # Dynamic weighting based on variance and KNN distance
        variance_scale = np.mean(sigma)
        knn_distance_scale = np.mean(min_distances)

        # Normalize scales to be between 0 and 1
        variance_scale = variance_scale / (variance_scale + 1e-6) #Avoid division by zero
        knn_distance_scale = knn_distance_scale / (knn_distance_scale + 1e-6) #Avoid division by zero


        # Adjust weights based on the scales
        self.ei_weight = 0.5 + 0.5 * (1 - variance_scale)
        self.knn_weight = 0.5 + 0.5 * (1 - knn_distance_scale)


        # Combined acquisition function
        acquisition = self.ei_weight * ei + self.knn_weight * knn_exploration
        return acquisition

    def _select_next_points(self, batch_size):
        x_tries = self._sample_points(batch_size * 10)
        x_tries = np.clip(x_tries, self.bounds[0], self.bounds[1])
        acq_values = self._acquisition_function(x_tries)
        indices = np.argsort(acq_values.flatten())[::-1][:batch_size]
        return x_tries[indices]

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y
    
    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))
        
        self.knn.fit(self.X) # Update KNN model

        best_index = np.argmin(self.y)
        if self.y[best_index, 0] < self.best_y:
            self.best_x = self.X[best_index]
            self.best_y = self.y[best_index, 0]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_X = np.clip(initial_X, self.bounds[0], self.bounds[1])
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        while self.n_evals < self.budget:
            # Fit GP model
            self.gp = self._fit_model(self.X, self.y)

            # Select points by acquisition function
            batch_size = min(10, self.budget - self.n_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Calculate the ratio of improvement
            predicted_improvement = self.best_y - self.gp.predict(self.best_x.reshape(1, -1))[0]
            actual_improvement = self.best_y - min(self.y)[0]

            if abs(predicted_improvement) < 1e-9:
                ratio = 0.0
            else:
                ratio = actual_improvement / predicted_improvement

            # Adjust trust region size
            if ratio > self.success_threshold:
                self.trust_region_radius *= self.trust_region_expand
            elif ratio < self.failure_threshold:
                self.trust_region_radius *= self.trust_region_shrink
                self.trust_region_radius = max(self.trust_region_radius, 0.1)

            # Update trust region center
            self.trust_region_center = self.best_x

            # Local search around the best solution
            if self.best_x is not None and self.n_evals < self.budget:
                local_X = self._sample_points(1) * self.local_search_radius + self.best_x
                local_X = np.clip(local_X, self.bounds[0], self.bounds[1])
                local_y = self._evaluate_points(func, local_X)
                self._update_eval_points(local_X, local_y)

                # Update local search success rate
                if local_y[0, 0] < self.best_y:
                    self.local_search_success_memory.append(1)
                else:
                    self.local_search_success_memory.append(0)

                if len(self.local_search_success_memory) > self.local_search_success_window:
                    self.local_search_success_memory.pop(0)

                self.local_search_success_rate = np.mean(self.local_search_success_memory)

                # Adjust local search radius
                if self.local_search_success_rate > 0.5:
                    self.local_search_radius *= 0.9  # Reduce radius if successful
                else:
                    self.local_search_radius *= 1.1  # Increase radius if unsuccessful
                self.local_search_radius = np.clip(self.local_search_radius, 0.01, 1.0)

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<AdaptiveTrustRegionKNNVarianceBO>", line 121, in __call__
 121->             next_X = self._select_next_points(batch_size)
  File "<AdaptiveTrustRegionKNNVarianceBO>", line 87, in _select_next_points
  85 |         acq_values = self._acquisition_function(x_tries)
  86 |         indices = np.argsort(acq_values.flatten())[::-1][:batch_size]
  87->         return x_tries[indices]
  88 | 
  89 |     def _evaluate_points(self, func, X):
IndexError: index 3769 is out of bounds for axis 0 with size 100
