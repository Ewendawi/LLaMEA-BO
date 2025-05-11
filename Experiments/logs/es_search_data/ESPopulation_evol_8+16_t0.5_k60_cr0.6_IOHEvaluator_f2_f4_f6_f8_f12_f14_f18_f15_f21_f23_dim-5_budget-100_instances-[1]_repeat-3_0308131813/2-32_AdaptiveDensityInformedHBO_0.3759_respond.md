# Description
**Adaptive Density-Informed Hybrid Bayesian Optimization (ADIHBO):** This algorithm combines the strengths of AHBBO and AdaptiveBandwidthDensiTreeBO (ABDTBO) to achieve a more robust and efficient Bayesian optimization. It uses a Gaussian Process Regression (GPR) model with a hybrid acquisition function (Expected Improvement + Adaptive Distance-based Exploration) from AHBBO for balancing exploration and exploitation. It integrates a Kernel Density Estimation (KDE) with adaptive bandwidth from ABDTBO to focus the search on high-density regions of promising solutions. The exploration weight in the acquisition function is dynamically adjusted based on the optimization progress, and a lower bound is introduced to prevent premature convergence. The KDE bandwidth is adaptively adjusted based on the local density of evaluated points. Finally, a refinement step is added where the best point among the top KDE scoring points is selected using the acquisition function.

# Justification
The ADIHBO algorithm leverages the strengths of both AHBBO and AdaptiveBandwidthDensiTreeBO. AHBBO provides a good balance of exploration and exploitation through its hybrid acquisition function and adaptive exploration weight. AdaptiveBandwidthDensiTreeBO focuses the search on promising high-density regions using KDE with adaptive bandwidth. Combining these two approaches allows the algorithm to efficiently explore the search space while focusing on the most promising regions.

The adaptive bandwidth selection for KDE allows the algorithm to adjust to the local density of the evaluated points, which is particularly useful for multimodal functions. The dynamic adjustment of the exploration weight in the acquisition function helps to prevent premature convergence. The refinement step, where the best point among the top KDE scoring points is selected using the acquisition function, further improves the exploitation of promising regions.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KernelDensity, NearestNeighbors


class AdaptiveDensityInformedHBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim

        self.best_y = np.inf
        self.best_x = None

        self.batch_size = min(10, dim)
        self.exploration_weight = 0.1
        self.exploration_decay = 0.995
        self.min_exploration = 0.01
        self.k_neighbors = min(10, 2 * dim)  # Number of neighbors for bandwidth estimation
        self.kde_bandwidth = 0.5  # Initial bandwidth for KDE, can be tuned

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

    def _adaptive_bandwidth(self):
        # Estimate bandwidth based on data density
        if self.X is None or len(self.X) < self.k_neighbors:
            return self.kde_bandwidth  # Return default bandwidth if not enough data

        # Calculate distances to the k-th nearest neighbor
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors, algorithm='ball_tree').fit(self.X)
        distances, _ = nbrs.kneighbors(self.X)
        k_distances = distances[:, -1]

        # Use the median of the k-th nearest neighbor distances as the bandwidth
        bandwidth = np.median(k_distances)
        return bandwidth

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
            exploration = np.ones_like(ei)

        # Hybrid acquisition function
        acquisition = ei + self.exploration_weight * exploration
        return acquisition

    def _select_next_points(self, batch_size):
        if self.X is None or len(self.X) < self.dim + 1:
            # Not enough data for KDE, return random samples
            return self._sample_points(batch_size)

        # Adapt bandwidth
        bandwidth = self._adaptive_bandwidth()

        # Fit KDE to the evaluated points
        kde = KernelDensity(bandwidth=bandwidth).fit(self.X)

        # Generate candidate points
        candidate_points = self._sample_points(100 * batch_size)

        # Calculate KDE scores for candidate points
        kde_scores = kde.score_samples(candidate_points)

        # Select top candidate points based on KDE scores
        top_indices = np.argsort(kde_scores)[-batch_size:]
        next_points = candidate_points[top_indices]

        # Refine selection using acquisition function
        acquisition_values = self._acquisition_function(next_points)
        best_index = np.argmax(acquisition_values)

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
            batch_size = min(self.batch_size, remaining_evals)

            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self.model = self._fit_model(self.X, self.y)

            self.exploration_weight = max(
                self.exploration_weight * self.exploration_decay,
                self.min_exploration,
            )

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AdaptiveDensityInformedHBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1517 with standard deviation 0.1014.

took 44.16 seconds to run.