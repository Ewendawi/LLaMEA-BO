# Description
**Efficient Density-Adaptive Hybrid Bayesian Optimization with Improved Exploration (EDAHBO-IE):** This algorithm enhances the original EDAHBO by refining the exploration strategy. It introduces a more sophisticated distance-based exploration term that considers the average distance to the k-nearest neighbors instead of just the minimum distance. This provides a more robust measure of isolation and encourages exploration in less-explored regions. Additionally, the algorithm incorporates a dynamic adjustment of the KDE bandwidth based on the optimization progress, shrinking the bandwidth over time to focus on promising regions. Finally, a local search is added around the best point found so far, to refine the solution.

# Justification
The key enhancements are designed to improve the exploration-exploitation balance.
1.  **Average Distance to k-Nearest Neighbors:** Using the average distance to the k-nearest neighbors provides a more robust measure of isolation compared to just the minimum distance. This helps to avoid getting stuck in local optima and encourages exploration in less-explored regions.
2.  **Dynamic KDE Bandwidth Adjustment:** Dynamically adjusting the KDE bandwidth based on the optimization progress allows the algorithm to initially explore a wider region and then gradually focus on promising areas. This can improve the efficiency of the search and lead to better solutions.
3.  **Local Search:** The addition of a local search refines the solution by searching around the best point found so far. This can help to escape local optima and improve the accuracy of the solution.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KernelDensity, NearestNeighbors
from scipy.optimize import minimize

class EDAHBO_IE:
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
        self.batch_size = min(10, dim)  # Batch size for selecting points
        self.local_search_iter = 3

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

        # Distance-based exploration term (Average distance to k-NN)
        if self.X is not None and len(self.X) > 0:
            nbrs = NearestNeighbors(
                n_neighbors=self.k_neighbors, algorithm="ball_tree"
            ).fit(self.X)
            distances, _ = nbrs.kneighbors(X)
            avg_dist = np.mean(distances, axis=1, keepdims=True)
            exploration = avg_dist / np.max(avg_dist)
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
            ei + exploration_weight * exploration + kde_weight * kde_exploration
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
        bandwidth = np.median(k_distances) * (0.95**(self.n_evals/self.budget)) # Shrink bandwidth over time
        return bandwidth

    def _select_next_points(self, batch_size):
        candidate_points = self._sample_points(100 * batch_size)
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

    def _local_search(self, func):
        if self.best_x is not None:
            def obj_func(x):
                return func(x)

            res = minimize(obj_func, self.best_x, bounds=[(-5, 5)] * self.dim, method='L-BFGS-B', options={'maxiter': self.local_search_iter})
            if res.fun < self.best_y:
                self.best_y = res.fun
                self.best_x = res.x
                self._update_eval_points(res.x.reshape(1,-1), np.array([[res.fun]])) # Update history

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[
        np.float64, np.array
    ]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            remaining_evals = self.budget - self.n_evals
            batch_size = min(self.batch_size, remaining_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self.model = self._fit_model(self.X, self.y)

            self._local_search(func)

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<EDAHBO_IE>", line 145, in __call__
 145->             self._local_search(func)
  File "<EDAHBO_IE>", line 122, in _local_search
 122->             res = minimize(obj_func, self.best_x, bounds=[(-5, 5)] * self.dim, method='L-BFGS-B', options={'maxiter': self.local_search_iter})
  File "<EDAHBO_IE>", line 120, in obj_func
 118 |         if self.best_x is not None:
 119 |             def obj_func(x):
 120->                 return func(x)
 121 | 
 122 |             res = minimize(obj_func, self.best_x, bounds=[(-5, 5)] * self.dim, method='L-BFGS-B', options={'maxiter': self.local_search_iter})
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
