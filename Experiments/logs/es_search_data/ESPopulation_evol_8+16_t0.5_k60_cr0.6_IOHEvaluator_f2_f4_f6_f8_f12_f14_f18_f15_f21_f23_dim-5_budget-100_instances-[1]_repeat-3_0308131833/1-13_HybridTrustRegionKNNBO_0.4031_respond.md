# Description
**HybridTrustRegionKNNBO (HTRKBO):** This algorithm combines the strengths of Trust Region Bayesian Optimization (TRBO) and Surrogate Model Free Bayesian Optimization (SMFBO) using a KNN approach. It employs a Gaussian Process (GP) surrogate model within a trust region framework, similar to TRBO, but adaptively adjusts the acquisition function by incorporating information from the nearest neighbors, inspired by SMFBO. Specifically, the acquisition function is a weighted combination of Expected Improvement (EI) from the GP model and a distance-based exploration term from KNN. The weights are dynamically adjusted based on the optimization progress. The trust region size is adaptively adjusted based on the agreement between the GP model's predictions and the actual function evaluations. A local search is performed around the best-observed solution to refine the search.

# Justification
The HTRKBO algorithm is designed to improve upon TRBO and SMFBO by leveraging their complementary strengths. TRBO can be effective in exploiting promising regions of the search space but may sometimes get stuck in local optima. SMFBO, on the other hand, provides a way to explore the search space without relying heavily on a surrogate model, which can be useful when the model is inaccurate.

The key components of HTRKBO are:

*   **Trust Region Framework:** This helps to focus the search on promising regions while ensuring sufficient exploration.
*   **Gaussian Process Surrogate Model:** This provides a probabilistic model of the objective function, which is used to guide the search.
*   **KNN-based Exploration:** This encourages exploration in regions that are far from existing points, which can help to escape local optima.
*   **Adaptive Acquisition Function:** The weights of the EI and distance-based exploration terms are dynamically adjusted based on the optimization progress. This allows the algorithm to adapt to the characteristics of the objective function.
*   **Local Search:** This helps to refine the search around the best-observed solution.

By combining these components, HTRKBO aims to achieve a better balance between exploration and exploitation, leading to improved performance on a wide range of optimization problems.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors

class HybridTrustRegionKNNBO:
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
        self.knn = NearestNeighbors(n_neighbors=5)
        self.exploration_weight = 0.1
        self.ei_weight = 0.5 # Initial weight for Expected Improvement
        self.knn_weight = 0.5 # Initial weight for KNN exploration

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        scaled_sample = qmc.scale(sample, -self.trust_region_radius, self.trust_region_radius)
        return scaled_sample + self.trust_region_center

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
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

        # KNN-based exploration
        distances, indices = self.knn.kneighbors(X)
        min_distances = distances[:, 0].reshape(-1, 1)
        knn_exploration = min_distances

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
            
            # Adaptively adjust weights
            if ratio > 0.5:  # If the GP model is doing well
                self.ei_weight = min(1.0, self.ei_weight + 0.05)
                self.knn_weight = max(0.0, self.knn_weight - 0.05)
            else:  # If the GP model is not doing well
                self.ei_weight = max(0.0, self.ei_weight - 0.05)
                self.knn_weight = min(1.0, self.knn_weight + 0.05)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm HybridTrustRegionKNNBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1794 with standard deviation 0.1030.

took 1.82 seconds to run.