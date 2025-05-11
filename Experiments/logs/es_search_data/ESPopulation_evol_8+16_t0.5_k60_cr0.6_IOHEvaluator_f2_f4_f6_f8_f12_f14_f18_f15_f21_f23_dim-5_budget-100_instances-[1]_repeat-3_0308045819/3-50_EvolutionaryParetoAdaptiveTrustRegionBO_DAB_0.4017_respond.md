# Description
**Evolutionary Pareto Adaptive Trust Region Bayesian Optimization with Dynamic Acquisition Balancing (EPATRBO-DAB):** This algorithm combines the strengths of AdaptiveParetoLCBTrustRegionBO and AdaptiveEvolutionaryTrustRegionBO, while introducing a dynamic balancing mechanism between exploration and exploitation in the acquisition function. It uses a trust region approach with adaptive radius adjustment. Within the trust region, it employs differential evolution (DE) to efficiently search for candidate points. The acquisition function is a Pareto-based combination of Lower Confidence Bound (LCB) and a diversity metric, similar to AdaptiveParetoLCBTrustRegionBO. However, instead of normalizing and equally weighting LCB and diversity, this algorithm dynamically adjusts the weights based on the optimization progress, trust region success, and remaining budget. This dynamic adjustment allows for a more flexible exploration-exploitation trade-off. It also incorporates a mechanism to dynamically adjust the LCB's kappa parameter, annealing it over time to promote exploration initially and exploitation later.

# Justification
The algorithm attempts to improve upon AdaptiveParetoLCBTrustRegionBO and AdaptiveEvolutionaryTrustRegionBO by combining their strengths and addressing their weaknesses.

*   **Combining Strengths:** It incorporates the Pareto-based acquisition function from AdaptiveParetoLCBTrustRegionBO to balance LCB and diversity, and the differential evolution search from AdaptiveEvolutionaryTrustRegionBO for efficient local search within the trust region.
*   **Dynamic Acquisition Balancing:** The core innovation is the dynamic adjustment of weights for LCB and diversity in the Pareto front construction. This allows the algorithm to adapt its exploration-exploitation balance based on the current state of the optimization. If the trust region search is successful (i.e., finds better solutions), the weight for LCB is increased to promote exploitation. If the search is stagnating, the weight for diversity is increased to encourage exploration. The kappa parameter of LCB is also dynamically adjusted to further control exploration and exploitation.
*   **Computational Efficiency:** Differential Evolution is used within the trust region to efficiently find candidate points, but its computational cost is controlled by dynamically adjusting the number of iterations based on the remaining budget, similar to AdaptiveEvolutionaryTrustRegionBO_DKADI. The GP kernel remains fixed to reduce computational overhead.
*   **Addressing Limitations:** AdaptiveParetoLCBTrustRegionBO can sometimes get stuck in local optima due to its reliance on random sampling. AdaptiveEvolutionaryTrustRegionBO can be computationally expensive due to the DE search. This algorithm attempts to mitigate these issues by dynamically balancing the exploration-exploitation trade-off and controlling the DE search cost.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution

class EvolutionaryParetoAdaptiveTrustRegionBO_DAB:
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
        self.de_pop_size = 10
        self.lcb_kappa = 2.0  # Initial kappa value
        self.kappa_decay = 0.995 # Decay rate for kappa
        self.diversity_weight = 0.5 # Initial weight for diversity
        self.diversity_weight_increase = 1.05
        self.diversity_weight_decay = 0.95
        self.min_diversity_weight = 0.1
        self.max_diversity_weight = 0.9

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
        return self.gp

    def _lower_confidence_bound(self, X, kappa=2.0):
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            mu, sigma = self.gp.predict(X, return_std=True)
            lcb = mu - kappa * sigma
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
            X = x.reshape(1, -1)
            lcb = self._lower_confidence_bound(X, self.lcb_kappa)
            diversity = self._diversity_metric(X)

            lcb_normalized = (lcb - np.min(lcb)) / (np.max(lcb) - np.min(lcb)) if np.max(lcb) != np.min(lcb) else np.zeros_like(lcb)
            diversity_normalized = (diversity - np.min(diversity)) / (np.max(diversity) - np.min(diversity)) if np.max(diversity) != np.min(diversity) else np.zeros_like(diversity)

            # Dynamic weighting of LCB and diversity
            acquisition = (1 - self.diversity_weight) * lcb_normalized - self.diversity_weight * diversity_normalized
            return acquisition[0, 0]

        de_bounds = [(max(self.bounds[0][i], self.best_x[i] - self.trust_region_radius),
                      min(self.bounds[1][i], self.best_x[i] + self.trust_region_radius)) for i in range(self.dim)]
        
        maxiter = max(1, self.budget // (self.de_pop_size * self.dim * 2) - self.n_evals//(self.de_pop_size * self.dim * 2))
        result = differential_evolution(lambda x: -de_objective(x), de_bounds, popsize=self.de_pop_size, maxiter=maxiter, tol=0.01, disp=False)

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

    def _adjust_trust_region(self):
        if self.success_ratio > 0.5:
            self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)

    def _adjust_acquisition_params(self):
        #Anneal kappa
        self.lcb_kappa *= self.kappa_decay
        self.lcb_kappa = max(0.1, self.lcb_kappa)

        # Adjust diversity weight based on success ratio
        if self.success_ratio > 0.5:
            self.diversity_weight *= self.diversity_weight_decay
        else:
            self.diversity_weight *= self.diversity_weight_increase
        
        self.diversity_weight = np.clip(self.diversity_weight, self.min_diversity_weight, self.max_diversity_weight)


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
            self._adjust_acquisition_params()

        return self.best_y, self.best_x
```
## Feedback
 The algorithm EvolutionaryParetoAdaptiveTrustRegionBO_DAB got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1725 with standard deviation 0.1044.

took 122.85 seconds to run.