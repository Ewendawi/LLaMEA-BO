# Description
**Adaptive Pareto Contextual Trust Region Bayesian Optimization (APCTRBO):** This algorithm synergistically combines the strengths of Adaptive Pareto Trust Region Bayesian Optimization (APTRBO) and Adaptive Contextual Trust Region Bayesian Optimization (ACTRBO). It employs a trust region approach with adaptive radius adjustments, a Pareto-based multi-objective selection balancing expected improvement and diversity, and a context-aware acquisition function that penalizes points in crowded regions to promote exploration. Furthermore, it dynamically adjusts the weights of the expected improvement, diversity, and context penalty based on the optimization progress and success ratio, facilitating a more robust exploration-exploitation trade-off.

# Justification
The algorithm builds upon the existing APTRBO and ACTRBO by integrating their core components:
- **Trust Region:** The trust region approach helps to focus the search on promising areas, improving efficiency. The adaptive adjustment of the radius ensures that the search space is appropriately sized based on the success of previous iterations.
- **Pareto-based Selection:** Using a Pareto front to balance expected improvement and diversity helps to avoid premature convergence and explore the search space more effectively.
- **Contextual Penalty:** The context-aware acquisition function penalizes points that are too close to existing points, encouraging exploration in less-visited regions.
- **Dynamic Weight Adjustment:** The weights for expected improvement, diversity, and the contextual penalty are dynamically adjusted based on the success ratio of the trust region. This allows the algorithm to adapt its exploration-exploitation balance as the optimization progresses. If the success ratio is high, the algorithm increases the weight of exploitation (expected improvement). If the success ratio is low, the algorithm increases the weight of exploration (diversity and contextual penalty). This dynamic adjustment enhances the algorithm's robustness and adaptability to different problem characteristics.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

class AdaptiveParetoContextualTrustRegionBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(5 * dim, self.budget // 10)
        self.gp = None
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0
        self.radius_decay = 0.95
        self.radius_increase = 1.1
        self.min_radius = 0.1
        self.success_ratio = 0.0
        self.random_restart_prob = 0.05
        self.knn = NearestNeighbors(n_neighbors=5)
        self.context_penalty_weight = 0.1
        self.ei_weight = 0.5
        self.diversity_weight = 0.5

    def _sample_points(self, n_points):
        if self.best_x is None or np.random.rand() < self.random_restart_prob:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            points = []
            while len(points) < n_points:
                sample = np.random.uniform(-1, 1, size=(self.dim,))
                sample = self.best_x + self.trust_region_radius * sample
                sample = np.clip(sample, self.bounds[0], self.bounds[1])
                points.append(sample)
            return np.array(points)

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        self.gp.fit(X, y)
        self.knn.fit(X)
        return self.gp

    def _expected_improvement(self, X):
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            mu, sigma = self.gp.predict(X, return_std=True)
            sigma = np.clip(sigma, 1e-9, np.inf)
            imp = self.best_y - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            return ei.reshape(-1, 1)

    def _diversity_metric(self, X):
        if self.X is None:
            return np.ones((len(X), 1))
        else:
            distances = np.min(cdist(X, self.X), axis=1)
            return distances.reshape(-1, 1)

    def _context_penalty(self, X):
        if self.X is None:
            return np.zeros((len(X), 1))
        else:
            distances, _ = self.knn.kneighbors(X)
            context_penalty = np.mean(distances, axis=1).reshape(-1, 1)
            return context_penalty

    def _select_next_points(self, batch_size):
        n_candidates = min(100 * self.dim, self.budget - self.n_evals)
        candidates = self._sample_points(n_candidates)

        ei = self._expected_improvement(candidates)
        diversity = self._diversity_metric(candidates)
        context = self._context_penalty(candidates)

        ei_normalized = (ei - np.min(ei)) / (np.max(ei) - np.min(ei)) if np.max(ei) != np.min(ei) else np.zeros_like(ei)
        diversity_normalized = (diversity - np.min(diversity)) / (np.max(diversity) - np.min(diversity)) if np.max(diversity) != np.min(diversity) else np.zeros_like(diversity)
        context_normalized = (context - np.min(context)) / (np.max(context) - np.min(context)) if np.max(context) != np.min(context) else np.zeros_like(context)

        F = np.hstack([self.ei_weight * ei_normalized, self.diversity_weight * diversity_normalized, self.context_penalty_weight * context_normalized])

        is_efficient = np.ones(F.shape[0], dtype=bool)
        for i, c in enumerate(F):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(F[is_efficient] >= c, axis=1)
                is_efficient[i] = True

        pareto_front = candidates[is_efficient]

        if self.gp is not None and len(pareto_front) > 0:
            _, sigma = self.gp.predict(pareto_front, return_std=True)
            next_point = pareto_front[np.argmax(sigma)].reshape(1, -1)
        elif len(pareto_front) > 0:
            next_point = pareto_front[np.random.choice(len(pareto_front))].reshape(1, -1)
        else:
            # If Pareto front is empty, sample randomly from trust region
            next_point = self._sample_points(1)

        return next_point

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

        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]
            self.success_ratio = 1.0
        else:
            self.success_ratio *= 0.75

    def _adjust_trust_region(self):
        if self.success_ratio > 0.5:
            self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
            self.ei_weight = min(self.ei_weight + 0.05, 1.0)
            self.diversity_weight = max(self.diversity_weight - 0.025, 0.0)
            self.context_penalty_weight = max(self.context_penalty_weight - 0.025, 0.0)

        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)
            self.ei_weight = max(self.ei_weight - 0.05, 0.0)
            self.diversity_weight = min(self.diversity_weight + 0.025, 1.0)
            self.context_penalty_weight = min(self.context_penalty_weight + 0.025, 1.0)
        
        # Normalize weights to sum to 1
        total_weight = self.ei_weight + self.diversity_weight + self.context_penalty_weight
        self.ei_weight /= total_weight
        self.diversity_weight /= total_weight
        self.context_penalty_weight /= total_weight

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        batch_size = min(1, self.dim)
        while self.n_evals < self.budget:
            self._fit_model(self.X, self.y)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self._adjust_trust_region()

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AdaptiveParetoContextualTrustRegionBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1732 with standard deviation 0.1004.

took 16.37 seconds to run.