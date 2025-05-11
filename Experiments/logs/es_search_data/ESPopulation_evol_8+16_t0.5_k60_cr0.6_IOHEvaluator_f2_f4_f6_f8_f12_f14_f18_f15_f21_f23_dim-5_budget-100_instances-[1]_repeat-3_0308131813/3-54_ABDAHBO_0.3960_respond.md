# Description
**Adaptive Batch-Size Density-Aware Hybrid Bayesian Optimization (ABDAHBO):** This algorithm combines the adaptive batch size strategy from AHBBO_ABS with the density-aware exploration from EDAHBO. It dynamically adjusts the batch size based on the uncertainty estimates from the Gaussian Process Regression (GPR) model. It also incorporates a Kernel Density Estimation (KDE) to focus the search on high-density regions of promising solutions, with the bandwidth dynamically adjusted based on the local density of evaluated points. The acquisition function is a hybrid of Expected Improvement (EI), distance-based exploration, and KDE-based exploration, with dynamically adjusted weights. A novel aspect is the adaptive adjustment of the KDE bandwidth based on the local distribution of the evaluated points, improving the accuracy of density estimation. Finally, the exploration weight decay is also adjusted based on the batch size, ensuring a more robust exploration-exploitation trade-off.

# Justification
This algorithm aims to improve upon AHBBO_ABS and EDAHBO by combining their strengths. AHBBO_ABS effectively manages exploration and exploitation by adjusting the batch size based on GPR uncertainty. EDAHBO leverages KDE to focus the search in promising regions. Combining these approaches allows for more efficient exploration and exploitation of the search space.

Specifically:
- **Adaptive Batch Size:** The batch size is adjusted based on the average uncertainty (sigma) of the GPR predictions. High uncertainty leads to larger batch sizes (more exploration), while low uncertainty leads to smaller batch sizes (more exploitation). This is inherited from AHBBO_ABS.
- **Density-Aware Exploration:** A KDE is used to estimate the density of evaluated points. The KDE score is incorporated into the acquisition function to encourage exploration in high-density regions. This is from EDAHBO.
- **Dynamic KDE Bandwidth:** The KDE bandwidth is dynamically adjusted based on the median distance to the k-th nearest neighbor, allowing the KDE to adapt to the local density of the search space. This improves the accuracy of density estimation, especially in high-dimensional spaces.
- **Hybrid Acquisition Function:** The acquisition function combines EI, distance-based exploration, and KDE-based exploration. The weights of these terms are dynamically adjusted based on the optimization progress.
- **Exploration Weight Decay:** The exploration weight is decayed over time, shifting the focus from exploration to exploitation. The decay rate is also adjusted based on the batch size.

This combination aims to address the limitations of each individual algorithm. AHBBO_ABS may struggle in multimodal landscapes where exploration is crucial, while EDAHBO's performance depends on the accuracy of the KDE. By combining these techniques and dynamically adjusting their influence, the algorithm can adapt to a wider range of optimization problems.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KernelDensity, NearestNeighbors

class ABDAHBO:
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
        bandwidth = np.median(k_distances)
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

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ABDAHBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1711 with standard deviation 0.1012.

took 208.94 seconds to run.