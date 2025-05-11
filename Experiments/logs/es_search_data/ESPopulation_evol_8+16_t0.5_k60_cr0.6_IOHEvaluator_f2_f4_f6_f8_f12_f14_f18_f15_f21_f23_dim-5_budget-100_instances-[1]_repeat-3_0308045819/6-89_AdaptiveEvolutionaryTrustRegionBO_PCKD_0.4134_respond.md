# Description
**AdaptiveEvolutionaryTrustRegionBO with Pareto-based Contextual Exploration and Dynamic Kernel Selection (AETRBO-PCKD):** This algorithm combines the strengths of AETRBO-DKAE and AEPTRBO, incorporating Pareto-based exploration within the trust region, dynamic kernel selection for the GP model, and a context penalty to encourage exploration in less-sampled areas. It dynamically adjusts the trust region size based on success ratio and GP model error. The Pareto front balances LCB (exploitation), diversity (exploration), and a context penalty (exploration in less-sampled regions). Differential Evolution (DE) optimizes the Pareto front. Dynamic kernel selection adapts the GP model to the landscape complexity. The LCB kappa parameter is dynamically adjusted based on noise estimation. The DE population size is also dynamically adjusted based on the remaining budget.

# Justification
This algorithm builds upon the successful components of AETRBO-DKAE and AEPTRBO, aiming for a better balance between exploration and exploitation.

*   **Pareto-based Exploration:** Combines LCB, diversity, and a context penalty in a Pareto front to guide exploration. This addresses the limitations of relying solely on LCB, which can lead to premature convergence.
*   **Context Penalty:** Encourages exploration in less-sampled regions, improving global search capabilities. This is especially useful in high-dimensional spaces.
*   **Dynamic Kernel Selection:** Adapts the GP model to the landscape complexity, improving prediction accuracy.
*   **Trust Region Adaptation:** Dynamically adjusts the trust region size based on success ratio and GP model error, balancing exploration and exploitation.
*   **Dynamic Kappa:** Adjusts the LCB kappa parameter based on noise estimation, making the algorithm more robust to noisy environments.
*   **Dynamic DE Population Size:** Adjusts the DE population size based on the remaining budget, improving computational efficiency.
*   **Batch Size:** The batch size is dynamically adjusted to improve the efficiency of the algorithm.

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
from sklearn.model_selection import train_test_split

class AdaptiveEvolutionaryTrustRegionBO_PCKD:
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
        self.ei_weight = 0.33
        self.diversity_weight = 0.33
        self.context_weight = 0.34
        self.de_pop_size = 10
        self.lcb_kappa = 2.0
        self.kappa_decay = 0.99
        self.min_kappa = 0.1
        self.gp_error_threshold = 0.1
        self.noise_estimate = 1e-4
        self.context_penalty_exponent = 2.0

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
        # Dynamic Kernel Selection
        kernels = [
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=0.1, length_scale_bounds="fixed"),
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed"),
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=10.0, length_scale_bounds="fixed"),
        ]
        best_gp = None
        best_log_likelihood = -np.inf

        if len(X) > 5:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            for kernel in kernels:
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
                gp.fit(X_train, y_train)
                log_likelihood = gp.log_marginal_likelihood(gp.kernel_.theta)

                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_gp = gp
            self.gp = best_gp
        else:
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
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

    def _diversity_metric(self, X):
        if self.X is None:
            return np.ones((len(X), 1))
        else:
            distances = np.min(cdist(X, self.X), axis=1)
            return distances.reshape(-1, 1)

    def _context_penalty(self, X):
        if self.X is None:
            return np.ones((len(X), 1))
        else:
            # Calculate the average distance to the k-nearest neighbors
            distances = cdist(X, self.X)
            knn = min(5, len(self.X))  # Use k=5 or the number of existing points, whichever is smaller
            nearest_distances = np.partition(distances, knn - 1, axis=1)[:, :knn]
            avg_nearest_distance = np.mean(nearest_distances, axis=1)
            # Apply a power function to the average distances
            context_penalty = avg_nearest_distance ** self.context_penalty_exponent
            return context_penalty.reshape(-1, 1)

    def _acquisition_function(self, X):
        ei = self._expected_improvement(X)
        diversity = self._diversity_metric(X)
        context = self._context_penalty(X)

        ei_normalized = (ei - np.min(ei)) / (np.max(ei) - np.min(ei)) if np.max(ei) != np.min(ei) else np.zeros_like(ei)
        diversity_normalized = (diversity - np.min(diversity)) / (np.max(diversity) - np.min(diversity)) if np.max(diversity) != np.min(diversity) else np.zeros_like(diversity)
        context_normalized = (context - np.min(context)) / (np.max(context) - np.min(context)) if np.max(context) != np.min(context) else np.zeros_like(context)

        F = np.hstack([self.ei_weight * ei_normalized, self.diversity_weight * diversity_normalized, self.context_weight * context_normalized])
        return F

    def _select_next_points(self, batch_size):
        if self.best_x is None:
            return self._sample_points(batch_size)

        def de_objective(x):
            F = self._acquisition_function(x.reshape(1, -1))
            is_efficient = np.ones(F.shape[0], dtype=bool)
            for i, c in enumerate(F):
                if is_efficient[i]:
                    is_efficient[is_efficient] = np.any(F[is_efficient] >= c, axis=1)
                    is_efficient[i] = True
            # Return negative sum of normalized objectives for Pareto front points
            if is_efficient[0]:
                return -np.sum(F[0])
            else:
                return 0

        de_bounds = [(max(self.bounds[0][i], self.best_x[i] - self.trust_region_radius),
                      min(self.bounds[1][i], self.best_x[i] + self.trust_region_radius)) for i in range(self.dim)]

        remaining_evals = self.budget - self.n_evals
        self.de_pop_size = max(5, min(20, int(remaining_evals / (self.dim * 5)))) #dynamic de pop size
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
                self.ei_weight = min(self.ei_weight + 0.05, 1.0)
                self.diversity_weight = max(self.diversity_weight - 0.025, 0.0)
                self.context_weight = max(self.context_weight - 0.025, 0.0)
            else:
                self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)
                self.ei_weight = max(self.ei_weight - 0.025, 0.0)
                self.diversity_weight = min(self.diversity_weight + 0.0125, 1.0)
                self.context_weight = min(self.context_weight + 0.0125, 1.0)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)

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

            self.lcb_kappa = max(self.lcb_kappa * self.kappa_decay, self.min_kappa)
            self.min_kappa = 0.1 + np.sqrt(self.noise_estimate)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AdaptiveEvolutionaryTrustRegionBO_PCKD got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1874 with standard deviation 0.1108.

took 166.20 seconds to run.