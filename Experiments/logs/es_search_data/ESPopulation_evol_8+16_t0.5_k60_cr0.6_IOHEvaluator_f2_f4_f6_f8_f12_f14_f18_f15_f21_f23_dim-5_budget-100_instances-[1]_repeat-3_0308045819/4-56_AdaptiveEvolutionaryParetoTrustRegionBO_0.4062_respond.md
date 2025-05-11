# Description
**AdaptiveEvolutionaryParetoTrustRegionBO (AEPTRBO):** This algorithm combines the strengths of AdaptiveEvolutionaryTrustRegionBO_DKAE (AETRBO-DKAE) and AdaptiveParetoLCBTrustRegionBO (APLCB-TRBO) to achieve a more robust and efficient Bayesian Optimization. It uses a trust region framework with adaptive radius adjustment, similar to AETRBO-DKAE. Within the trust region, it employs a Pareto-based approach, inspired by APLCB-TRBO, to balance exploration (diversity) and exploitation (LCB). However, instead of sampling candidate points and then selecting the Pareto front, it uses differential evolution (DE), like AETRBO-DKAE, to directly optimize the Pareto front, considering both LCB and diversity as objectives. A dynamic weighting scheme is introduced to adjust the importance of LCB and diversity based on the optimization progress and the trust region's success. This dynamic weighting allows the algorithm to adapt its exploration-exploitation balance more effectively.

# Justification
The algorithm leverages the strengths of both AETRBO-DKAE and APLCB-TRBO.

*   **Trust Region with Adaptive Radius:** The trust region approach focuses the search on promising regions, and the adaptive radius adjustment helps to balance exploration and exploitation. The radius is adapted based on the success ratio and GP error within the trust region, as in AETRBO-DKAE.
*   **Pareto-Based Acquisition with LCB and Diversity:** Balancing exploration and exploitation is crucial for efficient optimization. The Pareto-based approach, using LCB and a diversity metric, allows the algorithm to consider both the predicted value and the unexplored space.
*   **Differential Evolution for Pareto Optimization:** Instead of sampling and then selecting the Pareto front, DE is used to directly optimize the Pareto front within the trust region. This allows for a more efficient search for points that balance LCB and diversity.
*   **Dynamic Weighting of LCB and Diversity:** The weights for LCB and diversity are dynamically adjusted based on the optimization progress. If the algorithm is making good progress (high success ratio), more weight is given to LCB (exploitation). If the algorithm is stagnating or the GP model is inaccurate, more weight is given to diversity (exploration). This dynamic weighting allows the algorithm to adapt its exploration-exploitation balance more effectively.
*   **Dynamic Kappa:** Similar to AETRBO-DKAE, the kappa parameter in LCB is dynamically adjusted to further control the exploration-exploitation balance.
*   **Computational Efficiency:** By using DE to directly optimize the Pareto front, the algorithm avoids the need to sample a large number of candidate points, which can be computationally expensive.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist

class AdaptiveEvolutionaryParetoTrustRegionBO:
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
        self.trust_region_radius = 2.0 * np.sqrt(dim)
        self.radius_decay = 0.95
        self.radius_increase = 1.1
        self.min_radius = 0.1
        self.success_ratio = 0.0
        self.random_restart_prob = 0.05
        self.de_pop_size = 10
        self.lcb_kappa = 2.0
        self.kappa_decay = 0.99
        self.min_kappa = 0.1
        self.gp_error_threshold = 0.1
        self.noise_estimate = 1e-4
        self.lcb_weight = 0.5  # Initial weight for LCB
        self.diversity_weight = 0.5  # Initial weight for diversity
        self.weight_decay = 0.95
        self.weight_increase = 1.05

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
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
        self.gp.fit(X, y)
        return self.gp

    def _lower_confidence_bound(self, X):
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            mu, sigma = self.gp.predict(X, return_std=True)
            lcb = mu - self.lcb_kappa * sigma
            return lcb.reshape(-1, 1)

    def _diversity_metric(self, X):
        if self.X is None:
            return np.ones((len(X), 1))
        else:
            distances = np.min(cdist(X, self.X), axis=1)
            return distances.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        if self.best_x is None:
            return self._sample_points(batch_size)

        def de_objective(x):
            x = x.reshape(1, -1)
            lcb = self._lower_confidence_bound(x)
            diversity = self._diversity_metric(x)

            # Normalize LCB and diversity
            lcb_normalized = (lcb - np.min(lcb)) / (np.max(lcb) - np.min(lcb)) if np.max(lcb) != np.min(lcb) else np.zeros_like(lcb)
            diversity_normalized = (diversity - np.min(diversity)) / (np.max(diversity) - np.min(diversity)) if np.max(diversity) != np.min(diversity) else np.zeros_like(diversity)

            # Weighted combination of LCB and diversity
            acquisition = self.lcb_weight * lcb_normalized - self.diversity_weight * diversity_normalized  # Minimize LCB, maximize diversity
            return acquisition[0, 0]

        de_bounds = [(max(self.bounds[0][i], self.best_x[i] - self.trust_region_radius),
                      min(self.bounds[1][i], self.best_x[i] + self.trust_region_radius)) for i in range(self.dim)]

        remaining_evals = self.budget - self.n_evals
        maxiter = max(1, int(remaining_evals / (self.de_pop_size * self.dim * 2)))
        maxiter = min(maxiter, 100)

        result = differential_evolution(de_objective, de_bounds, popsize=self.de_pop_size, maxiter=maxiter, tol=0.01, disp=False)

        return result.x.reshape(1, -1)

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

        self.noise_estimate = np.var(self.y)

    def _adjust_trust_region(self):
        if self.best_x is not None:
            distances = np.linalg.norm(self.X - self.best_x, axis=1)
            within_tr = distances < self.trust_region_radius
            if np.any(within_tr):
                X_tr = self.X[within_tr]
                y_tr = self.y[within_tr]
                mu, _ = self.gp.predict(X_tr, return_std=True)
                gp_error = np.mean(np.abs(mu.reshape(-1, 1) - y_tr))
            else:
                gp_error = 0.0

            if self.success_ratio > 0.5 and gp_error < self.gp_error_threshold:
                self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
            else:
                self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)

    def _adjust_weights(self):
        if self.success_ratio > 0.5:
            self.lcb_weight = min(self.lcb_weight * self.weight_increase, 1.0)
            self.diversity_weight = max(self.diversity_weight * self.weight_decay, 0.0)
        else:
            self.lcb_weight = max(self.lcb_weight * self.weight_decay, 0.0)
            self.diversity_weight = min(self.diversity_weight * self.weight_increase, 1.0)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        batch_size = min(4, self.dim)
        while self.n_evals < self.budget:
            self._fit_model(self.X, self.y)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self._adjust_trust_region()
            self._adjust_weights()
            self.lcb_kappa = max(self.lcb_kappa * self.kappa_decay, self.min_kappa)
            self.min_kappa = 0.1 + np.sqrt(self.noise_estimate)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AdaptiveEvolutionaryParetoTrustRegionBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1814 with standard deviation 0.1015.

took 205.34 seconds to run.