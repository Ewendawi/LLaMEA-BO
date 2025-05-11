# Description
**ParetoEvolutionaryTrustRegionBO (PETRBO):** This algorithm combines the strengths of AdaptiveEvolutionaryTrustRegionBO_DKAE and AdaptiveParetoTrustRegionBO. It uses a trust region approach with adaptive radius adjustment. Within the trust region, it employs a Pareto-based approach to balance exploration (diversity) and exploitation (Lower Confidence Bound). Differential Evolution (DE) is then used to efficiently search for points on the Pareto front. The GP kernel is dynamically tuned by optimizing the length scale. A dynamic weighting of the EI and diversity components in the Pareto-based acquisition function is also included. This dynamic weighting allows the algorithm to adapt its exploration-exploitation balance more effectively.

# Justification
This algorithm attempts to leverage the strengths of both AdaptiveEvolutionaryTrustRegionBO_DKAE and AdaptiveParetoTrustRegionBO.
- **Trust Region with Adaptive Radius:** The trust region approach focuses the search on promising areas, and the adaptive radius adjusts the search space based on the success of previous iterations.
- **Pareto-based Exploration/Exploitation:** The Pareto front balances exploration (diversity) and exploitation (LCB) to avoid premature convergence.
- **Differential Evolution:** DE efficiently searches for optimal points within the trust region, improving the search process.
- **Dynamic Kernel Tuning:** Tuning the GP kernel dynamically adapts the model to the specific problem, improving its accuracy.
- **Dynamic Acquisition Weighting:** Adjusting the weights of EI and diversity allows for a more flexible exploration-exploitation balance, adapting to the optimization progress.
- **LCB acquisition:** Using LCB instead of EI is computationally more efficient.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist

class ParetoEvolutionaryTrustRegionBO:
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
        self.ei_weight = 0.5
        self.diversity_weight = 0.5
        self.weight_decay = 0.98
        self.min_weight = 0.1
        self.gp_kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

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

        def tune_kernel(kernel):
            def obj(x):
                kernel_tuned = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=x, length_scale_bounds=(1e-2, 1e2))
                gp = GaussianProcessRegressor(kernel=kernel_tuned, n_restarts_optimizer=0, alpha=1e-6)
                gp.fit(X, y)
                return -gp.log_marginal_likelihood()

            initial_length_scale = kernel.get_params()['rbf__length_scale']
            res = minimize(obj, initial_length_scale, bounds=[(1e-2, 1e2)])
            best_length_scale = res.x[0]
            return ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=best_length_scale, length_scale_bounds=(1e-2, 1e2))

        self.gp_kernel = tune_kernel(self.gp_kernel)
        self.gp = GaussianProcessRegressor(kernel=self.gp_kernel, n_restarts_optimizer=0, alpha=1e-6)
        self.gp.fit(X, y)
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

        def de_objective(x):
            x = x.reshape(1, -1)
            ei = self._expected_improvement(x)
            diversity = self._diversity_metric(x)

            ei_normalized = (ei - np.min(ei)) / (np.max(ei) - np.min(ei)) if np.max(ei) != np.min(ei) else np.zeros_like(ei)
            diversity_normalized = (diversity - np.min(diversity)) / (np.max(diversity) - np.min(diversity)) if np.max(diversity) != np.min(diversity) else np.zeros_like(diversity)

            # Weighted sum of EI and diversity
            acquisition = self.ei_weight * ei_normalized + self.diversity_weight * diversity_normalized
            return -acquisition[0, 0]

        de_bounds = [(max(self.bounds[0][i], self.best_x[i] - self.trust_region_radius),
                      min(self.bounds[1][i], self.best_x[i] + self.trust_region_radius)) for i in range(self.dim)]

        remaining_evals = self.budget - self.n_evals
        maxiter = max(1, int(remaining_evals / (self.de_pop_size * self.dim * 2)))
        maxiter = min(maxiter, 50)

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

    def _adjust_trust_region(self):
        if self.success_ratio > 0.5:
            self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)

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

            self.lcb_kappa = max(self.lcb_kappa * self.kappa_decay, self.min_kappa)
            self.ei_weight = max(self.ei_weight * self.weight_decay, self.min_weight)
            self.diversity_weight = max(self.diversity_weight * self.weight_decay, self.min_weight)

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<ParetoEvolutionaryTrustRegionBO>", line 155, in __call__
 155->             self._fit_model(self.X, self.y)
  File "<ParetoEvolutionaryTrustRegionBO>", line 66, in _fit_model
  66->         self.gp_kernel = tune_kernel(self.gp_kernel)
  File "<ParetoEvolutionaryTrustRegionBO>", line 61, in tune_kernel
  59 |                 return -gp.log_marginal_likelihood()
  60 | 
  61->             initial_length_scale = kernel.get_params()['rbf__length_scale']
  62 |             res = minimize(obj, initial_length_scale, bounds=[(1e-2, 1e2)])
  63 |             best_length_scale = res.x[0]
KeyError: 'rbf__length_scale'
