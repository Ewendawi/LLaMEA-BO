# Description
**Adaptive Pareto Evolutionary Trust Region Bayesian Optimization (APETRBO):** This algorithm synergistically integrates Adaptive Pareto Trust Region Bayesian Optimization (APTRBO) and Adaptive Bayesian Evolutionary Optimization (ABEO). It employs a trust region framework with adaptive radius adjustment, similar to APTRBO, to focus the search. Within the trust region, it leverages a Pareto-based approach, balancing expected improvement and diversity, as well as differential evolution, inspired by ABEO, to efficiently explore the acquisition function landscape. The GP kernel is adaptively tuned, and the diversity weight is dynamically adjusted, similar to ABEO, to enhance the GP model's accuracy and exploration-exploitation balance. To further improve the search within the trust region, a combination of Pareto-based selection and differential evolution is used to select candidate points.

# Justification
The algorithm combines the strengths of both APTRBO and ABEO. APTRBO provides a robust trust region framework and Pareto-based selection for balancing exploration and exploitation. ABEO enhances the GP model by adaptively tuning the kernel and dynamically adjusting the diversity weight. Combining these two approaches can lead to improved performance and robustness.

Specifically:
1.  **Trust Region:** The trust region approach from APTRBO helps to focus the search around promising regions, improving convergence speed and efficiency.
2.  **Pareto-based Selection:** The Pareto-based selection from APTRBO balances exploration and exploitation by considering both expected improvement and diversity.
3.  **Differential Evolution:** The differential evolution from ABEO efficiently explores the acquisition function landscape within the trust region.
4.  **Adaptive Kernel Tuning:** The adaptive kernel tuning from ABEO enhances the GP model's accuracy and ability to capture the underlying function's characteristics.
5.  **Dynamic Diversity Weight Adjustment:** The dynamic diversity weight adjustment from ABEO further improves the exploration-exploitation balance.

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
from sklearn.preprocessing import StandardScaler


class AdaptiveParetoEvolutionaryTrustRegionBO:
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
        self.de_pop_size = 15  # Population size for differential evolution
        self.diversity_weight = 0.1  # Initial diversity weight

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
        # Normalize y
        scaler = StandardScaler()
        y_normalized = scaler.fit_transform(y)

        # Define the kernel with tunable length_scale
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-6)

        # Optimize the kernel parameters using L-BFGS-B
        self.gp.fit(X, y_normalized)

        return self.gp

    def _expected_improvement(self, X):
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            # Normalize best_y for EI calculation
            scaler = StandardScaler()
            scaler.fit(self.y)  # Fit on self.y to get the correct scale
            best_y_normalized = scaler.transform(np.array([[self.best_y]]))[0][0]

            mu, sigma = self.gp.predict(X, return_std=True)
            sigma = np.clip(sigma, 1e-9, np.inf)
            imp = best_y_normalized - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            return ei.reshape(-1, 1)

    def _diversity_metric(self, X):
        if self.X is None:
            return np.ones((len(X), 1))
        else:
            distances = np.min(cdist(X, self.X), axis=1)
            return distances.reshape(-1, 1)

    def _acquisition_function(self, X):
        # Implement acquisition function: Expected Improvement + Diversity
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            # Normalize best_y for EI calculation
            scaler = StandardScaler()
            scaler.fit(self.y)  # Fit on self.y to get the correct scale
            best_y_normalized = scaler.transform(np.array([[self.best_y]]))[0][0]

            mu, sigma = self.gp.predict(X, return_std=True)
            sigma = np.clip(sigma, 1e-9, np.inf)
            imp = best_y_normalized - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

            # Diversity term: encourage exploration
            if self.X is not None:
                distances = np.min([np.linalg.norm(x - self.X, axis=1) for x in X], axis=0)
                diversity = distances.reshape(-1, 1)
            else:
                diversity = np.ones((len(X), 1))  # No diversity if no points yet

            # Dynamic diversity weight adjustment
            self.diversity_weight = np.std(sigma) / np.mean(ei + 1e-6)
            self.diversity_weight = np.clip(self.diversity_weight, 0.01, 0.5)  # Clip to avoid extreme values

            # Combine EI and diversity
            acquisition = ei + self.diversity_weight * diversity  # Adjust weight for diversity as needed
            return acquisition.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        n_candidates = min(100 * self.dim, self.budget - self.n_evals)
        candidates = self._sample_points(n_candidates)

        ei = self._expected_improvement(candidates)
        diversity = self._diversity_metric(candidates)

        ei_normalized = (ei - np.min(ei)) / (np.max(ei) - np.min(ei)) if np.max(ei) != np.min(ei) else np.zeros_like(ei)
        diversity_normalized = (diversity - np.min(diversity)) / (np.max(diversity) - np.min(diversity)) if np.max(diversity) != np.min(diversity) else np.zeros_like(diversity)

        F = np.hstack([ei_normalized, diversity_normalized])

        is_efficient = np.ones(F.shape[0], dtype=bool)
        for i, c in enumerate(F):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(F[is_efficient] >= c, axis=1)
                is_efficient[i] = True

        pareto_front = candidates[is_efficient]

        # Refine Pareto front with differential evolution
        if len(pareto_front) > 0:
            def de_objective(x):
                return -self._acquisition_function(x.reshape(1, -1))[0, 0]

            de_bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
            result = differential_evolution(de_objective, de_bounds, popsize=self.de_pop_size, maxiter=max(1, self.budget // (batch_size * 5)), tol=0.01, disp=False,
                                          initial_population=np.clip(pareto_front[np.random.choice(len(pareto_front), size=self.de_pop_size, replace=len(pareto_front) < self.de_pop_size)], self.bounds[0], self.bounds[1]))

            next_point = result.x.reshape(1, -1)
        elif self.gp is not None:
            _, sigma = self.gp.predict(candidates, return_std=True)
            next_point = candidates[np.argmax(sigma)].reshape(1, -1)
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

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<AdaptiveParetoEvolutionaryTrustRegionBO>", line 191, in __call__
 191->             next_X = self._select_next_points(batch_size)
  File "<AdaptiveParetoEvolutionaryTrustRegionBO>", line 143, in _select_next_points
 141 | 
 142 |             de_bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
 143->             result = differential_evolution(de_objective, de_bounds, popsize=self.de_pop_size, maxiter=max(1, self.budget // (batch_size * 5)), tol=0.01, disp=False,
 144 |                                           initial_population=np.clip(pareto_front[np.random.choice(len(pareto_front), size=self.de_pop_size, replace=len(pareto_front) < self.de_pop_size)], self.bounds[0], self.bounds[1]))
 145 | 
TypeError: differential_evolution() got an unexpected keyword argument 'initial_population'
