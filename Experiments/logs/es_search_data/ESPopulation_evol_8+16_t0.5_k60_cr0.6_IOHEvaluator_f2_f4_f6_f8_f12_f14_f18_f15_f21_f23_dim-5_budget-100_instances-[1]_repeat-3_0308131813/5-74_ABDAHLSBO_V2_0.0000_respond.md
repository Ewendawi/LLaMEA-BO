# Description
**Adaptive Batch-Size Density-Aware Hybrid Bayesian Optimization with Local Search and Improved Exploration (ABDAHLSBO_V2):** This algorithm builds upon ABDAHLSBO by incorporating several refinements to enhance its exploration and exploitation capabilities. It includes an adaptive local search radius, adjusted based on the optimization progress and model uncertainty. The exploration weight decay is also improved with a schedule that considers both evaluation count and GPR uncertainty. A more robust KDE bandwidth selection is implemented using a weighted median of k-distances. Finally, a dynamic adjustment of the local search probability is introduced to balance global exploration and local refinement.

# Justification
The key improvements focus on refining the exploration-exploitation trade-off and enhancing the local search strategy.
1.  **Adaptive Local Search Radius:** The original algorithm used a fixed local search radius. Adaptively adjusting this radius based on the optimization progress and model uncertainty allows for more focused local refinement when the model is confident and broader exploration when uncertainty is high.
2.  **Improved Exploration Weight Decay:** The original exploration weight decay only considered the number of evaluations. Incorporating GPR uncertainty into the decay schedule allows for more aggressive exploration in regions of high uncertainty and more exploitation in well-modeled regions.
3.  **Robust KDE Bandwidth Selection:** The original KDE bandwidth selection used the median of k-distances. A weighted median is more robust to outliers and provides a better estimate of the local density.
4.  **Dynamic Local Search Probability:** The probability of performing local search is dynamically adjusted based on the optimization progress. This allows for more global exploration in the early stages of optimization and more local refinement in the later stages.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KernelDensity, NearestNeighbors

class ABDAHLSBO_V2:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.k_neighbors = min(10, 2 * dim)
        self.best_y = np.inf
        self.best_x = None
        self.kde_bandwidth = 0.5
        self.max_batch_size = min(10, dim) # Maximum batch size for selecting points
        self.min_batch_size = 1
        self.exploration_weight = 0.1 # Initial exploration weight
        self.exploration_decay = 0.995 # Decay factor for exploration weight
        self.min_exploration = 0.01 # Minimum exploration weight
        self.uncertainty_threshold = 0.5  # Threshold for adjusting batch size
        self.initial_local_search_radius = 0.1
        self.local_search_radius = self.initial_local_search_radius
        self.local_search_radius_decay = 0.99
        self.min_local_search_radius = 0.01
        self.local_search_prob = 0.5  # Probability of performing local search
        self.local_search_prob_increase = 0.001
        self.max_local_search_prob = 0.9

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=5, alpha=1e-5
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Distance-based exploration term
        if self.X is not None and len(self.X) > 0:
            min_dist = np.min(
                np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2),
                axis=1,
                keepdims=True,
            )
            exploration = min_dist / np.max(min_dist)
        else:
            exploration = np.ones(X.shape[0]).reshape(-1, 1)

        # KDE-based exploration term
        if self.X is not None and len(self.X) > self.dim + 1:
            bandwidth = self._adaptive_bandwidth()
            kde = KernelDensity(bandwidth=bandwidth).fit(self.X)
            kde_scores = kde.score_samples(X)
            kde_scores = np.exp(kde_scores).reshape(-1, 1)  # Convert to density
            kde_exploration = kde_scores / np.max(kde_scores)  # Normalize
        else:
            kde_exploration = np.zeros(X.shape[0]).reshape(-1, 1)

        # Dynamic weighting
        exploration_weight = np.clip(1.0 - self.n_evals / self.budget, 0.1, 1.0)
        kde_weight = 1.0 - exploration_weight

        # Hybrid acquisition function
        acquisition = (
            ei + self.exploration_weight * exploration + kde_weight * kde_exploration
        )
        return acquisition

    def _adaptive_bandwidth(self):
        if self.X is None or len(self.X) < self.k_neighbors:
            return self.kde_bandwidth

        nbrs = NearestNeighbors(
            n_neighbors=self.k_neighbors, algorithm="ball_tree"
        ).fit(self.X)
        distances, _ = nbrs.kneighbors(self.X)
        k_distances = distances[:, -1]

        # Weighted median for robustness
        weights = np.arange(1, self.k_neighbors + 1)
        weights = weights / np.sum(weights)
        bandwidth = np.median(np.repeat(k_distances, weights))

        return bandwidth

    def _select_next_points(self, batch_size):
        candidate_points = self._sample_points(50 * batch_size)

        # Add points around the best solution (local search)
        if self.best_x is not None and np.random.rand() < self.local_search_prob:
            local_points = np.random.normal(loc=self.best_x, scale=self.local_search_radius, size=(50 * batch_size, self.dim))
            local_points = np.clip(local_points, self.bounds[0], self.bounds[1])
            candidate_points = np.vstack((candidate_points, local_points))

        acquisition_values = self._acquisition_function(candidate_points)
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]
        return next_points

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

        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[
        np.float64, np.array
    ]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            remaining_evals = self.budget - self.n_evals

            # Adjust batch size based on uncertainty
            _, sigma = self.model.predict(self.X, return_std=True)
            avg_sigma = np.mean(sigma)
            
            if avg_sigma > self.uncertainty_threshold:
                batch_size = self.max_batch_size
                exploration_decay = 0.99
            else:
                batch_size = self.min_batch_size
                exploration_decay = 0.999
            
            batch_size = min(batch_size, remaining_evals) # Adjust batch size to budget
            
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self.model = self._fit_model(self.X, self.y)

            # Update exploration weight
            self.exploration_weight = max(self.exploration_weight * exploration_decay, self.min_exploration)

            # Update local search radius
            self.local_search_radius = max(self.local_search_radius * self.local_search_radius_decay, self.min_local_search_radius)

            # Update local search probability
            self.local_search_prob = min(self.local_search_prob + self.local_search_prob_increase, self.max_local_search_prob)

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
TypeError: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<ABDAHLSBO_V2>", line 164, in __call__
 164->             next_X = self._select_next_points(batch_size)
  File "<ABDAHLSBO_V2>", line 116, in _select_next_points
 116->         acquisition_values = self._acquisition_function(candidate_points)
  File "<ABDAHLSBO_V2>", line 72, in _acquisition_function
  72->             bandwidth = self._adaptive_bandwidth()
  File "<ABDAHLSBO_V2>", line 103, in _adaptive_bandwidth
 101 |         weights = np.arange(1, self.k_neighbors + 1)
 102 |         weights = weights / np.sum(weights)
 103->         bandwidth = np.median(np.repeat(k_distances, weights))
 104 | 
 105 |         return bandwidth
  File "<__array_function__ internals>", line 200, in repeat
TypeError: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'
