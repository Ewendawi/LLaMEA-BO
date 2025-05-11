# Description
**Adaptive Trust Region with Pareto-based Batch Selection and Dynamic Kernel Tuning (ATRPBO-DKT):** This algorithm combines the strengths of Adaptive Trust Region Bayesian Optimization (ATBO) and Pareto Active Bayesian Optimization (PABO) while incorporating dynamic kernel tuning for the Gaussian Process (GP). It uses a trust region approach to focus the search around promising regions and employs a Pareto-based multi-objective approach to balance exploration (diversity) and exploitation (expected improvement) within the trust region. The GP kernel parameters (length_scale and constant_value) are optimized using a gradient-based method (L-BFGS-B) during the GP fitting stage. This allows the GP to better capture the underlying function's characteristics. It also uses a dynamic kappa parameter in the LCB acquisition function.

# Justification
This algorithm builds upon the strengths of both `AdaptiveParetoTrustRegionBO` and the enhanced `AdaptiveTrustRegionBO`.
1.  **Trust Region:** The trust region approach focuses the search on promising areas, improving sample efficiency.
2.  **Pareto-based Batch Selection:** Using a Pareto front to select a batch of points allows for a better balance between exploration (diversity) and exploitation (expected improvement) compared to selecting a single point at a time.
3.  **Dynamic Kernel Tuning:** Tuning the GP kernel parameters (length_scale and constant_value) allows the GP to adapt to the specific characteristics of the objective function, improving the accuracy of the surrogate model. This is crucial for the overall performance of BO.
4. **Dynamic Kappa:** Adjusting the exploration-exploitation trade-off based on the optimization progress can lead to faster convergence and better final results.
5. **Computational Efficiency:** The kernel tuning is performed during the GP fitting stage, adding minimal overhead. The Pareto front calculation is also relatively efficient.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

class AdaptiveTRPBO_DKT:
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
        self.kappa = 2.0 # Initial kappa value
        self.kappa_decay = 0.995 # Decay factor for kappa

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
        #kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        # Define the kernel with bounds
        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-6)
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
            sigma = np.clip(sigma, 1e-9, np.inf)
            lcb = mu - self.kappa * sigma
            return lcb.reshape(-1, 1)

    def _diversity_metric(self, X):
        if self.X is None:
            return np.ones((len(X), 1))
        else:
            distances = np.min(cdist(X, self.X), axis=1)
            return distances.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        n_candidates = min(100 * self.dim, self.budget - self.n_evals)
        candidates = self._sample_points(n_candidates)

        ei = self._expected_improvement(candidates)
        diversity = self._diversity_metric(candidates)
        lcb = self._lower_confidence_bound(candidates)

        ei_normalized = (ei - np.min(ei)) / (np.max(ei) - np.min(ei)) if np.max(ei) != np.min(ei) else np.zeros_like(ei)
        diversity_normalized = (diversity - np.min(diversity)) / (np.max(diversity) - np.min(diversity)) if np.max(diversity) != np.min(diversity) else np.zeros_like(diversity)
        lcb_normalized = (lcb - np.min(lcb)) / (np.max(lcb) - np.min(lcb)) if np.max(lcb) != np.min(lcb) else np.zeros_like(lcb)


        F = np.hstack([ei_normalized, diversity_normalized])

        is_efficient = np.ones(F.shape[0], dtype=bool)
        for i, c in enumerate(F):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(F[is_efficient] >= c, axis=1)
                is_efficient[i] = True

        pareto_front = candidates[is_efficient]

        if len(pareto_front) > batch_size:
            # Select a diverse subset from the Pareto front
            indices = np.random.choice(len(pareto_front), batch_size, replace=False)
            next_points = pareto_front[indices]
        elif len(pareto_front) > 0:
             next_points = pareto_front
        else:
            # If Pareto front is empty, sample randomly from trust region
            next_points = self._sample_points(batch_size)

        return next_points

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

        batch_size = min(4, self.dim) # Increased batch size
        while self.n_evals < self.budget:
            self._fit_model(self.X, self.y)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self._adjust_trust_region()
            self.kappa *= self.kappa_decay # Decay kappa

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AdaptiveTRPBO_DKT got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1685 with standard deviation 0.1018.

took 59.23 seconds to run.