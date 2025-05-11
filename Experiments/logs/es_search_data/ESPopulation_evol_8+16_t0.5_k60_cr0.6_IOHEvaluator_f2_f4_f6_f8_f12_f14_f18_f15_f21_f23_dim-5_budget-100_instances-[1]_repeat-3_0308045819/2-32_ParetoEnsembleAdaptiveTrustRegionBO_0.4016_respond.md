# Description
**ParetoEnsembleAdaptiveTrustRegionBO (PEATRBO):** This algorithm combines the strengths of Adaptive Pareto Trust Region Bayesian Optimization (APTRBO) and Trust Region Ensemble Bayesian Optimization (TREBO) while addressing their individual limitations. It uses an ensemble of Gaussian Process (GP) models within a trust region framework, similar to TREBO, to improve robustness and model uncertainty. It also incorporates a Pareto-based approach, similar to APTRBO, to balance exploration (diversity) and exploitation (expected improvement) when selecting candidate points within the trust region. A key innovation is the dynamic adjustment of the Pareto front's influence based on the trust region's success, allowing the algorithm to adapt its exploration-exploitation balance more effectively. Furthermore, instead of Thompson Sampling, it uses Expected Improvement for selecting the next point from the Pareto front, while also considering the variance of the GP ensemble. The trust region is centered around the best point found so far, and its radius is adaptively adjusted based on the success of previous iterations.

# Justification
The algorithm is designed based on the following observations:
1. **Combining Strengths:** APTRBO has a good AOCC score (0.1768) but can be further improved. TREBO leverages an ensemble of GPs which provides robustness. Combining these two approaches can lead to a more robust and efficient algorithm.
2. **Addressing Limitations:** APTRBO uses a fixed kernel and Thompson Sampling. TREBO has a static Pareto front influence. This algorithm addresses these issues by using an ensemble of GPs with different kernels (RBF and Matern) and dynamically adjusting the Pareto front influence.
3. **Dynamic Pareto Influence:** The balance between exploration and exploitation is crucial. The algorithm dynamically adjusts the influence of the Pareto front based on the success of the trust region. If the trust region is successful, the algorithm focuses more on exploitation. Otherwise, it increases the influence of diversity.
4. **Ensemble Variance:** Instead of Thompson Sampling, the algorithm selects the next point from the Pareto front by considering the Expected Improvement and the variance of the GP ensemble. This helps to select points that are both promising and have high uncertainty, leading to better exploration.
5. **Computational Efficiency:** The algorithm uses a limited number of GP models in the ensemble to maintain computational efficiency. The kernel parameters are fixed to avoid expensive optimization. The Pareto front calculation is also optimized to reduce the computational cost.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc, norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

class ParetoEnsembleAdaptiveTrustRegionBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(5 * dim, self.budget // 10)
        self.gp_ensemble = []
        self.ensemble_weights = []
        self.best_x = None
        self.best_y = float('inf')
        self.n_ensemble = 3
        self.trust_region_radius = 2.0
        self.radius_decay = 0.95
        self.radius_increase = 1.1
        self.min_radius = 0.1
        self.success_ratio = 0.0
        self.random_restart_prob = 0.05
        self.pareto_influence = 0.5  # Initial influence of the Pareto front
        self.pareto_decay = 0.95
        self.pareto_increase = 1.05

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
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        if not self.gp_ensemble:
            kernels = [
                ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed"),
                ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds="fixed", nu=0.5),
                ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds="fixed", nu=1.5)
            ]
            for i in range(self.n_ensemble):
                gp = GaussianProcessRegressor(kernel=kernels[i % len(kernels)], n_restarts_optimizer=0, alpha=1e-6)
                gp.fit(X_train, y_train)
                self.gp_ensemble.append(gp)
                self.ensemble_weights.append(1.0 / self.n_ensemble)
        else:
            for gp in self.gp_ensemble:
                gp.fit(X_train, y_train)

        val_errors = []
        for gp in self.gp_ensemble:
            y_pred, _ = gp.predict(X_val, return_std=True)
            error = np.mean((y_pred - y_val.flatten())**2)
            val_errors.append(error)

        val_errors = np.array(val_errors)
        weights = np.exp(-val_errors) / np.sum(np.exp(-val_errors))
        self.ensemble_weights = weights

    def _expected_improvement(self, X):
        ei = np.zeros((len(X), 1))
        for i, gp in enumerate(self.gp_ensemble):
            mu, sigma = gp.predict(X, return_std=True)
            sigma = np.clip(sigma, 1e-9, np.inf)
            imp = self.best_y - mu
            Z = imp / sigma
            ei += self.ensemble_weights[i] * (imp * norm.cdf(Z) + sigma * norm.pdf(Z)).reshape(-1, 1)
        return ei

    def _ensemble_variance(self, X):
        variance = np.zeros((len(X), 1))
        for i, gp in enumerate(self.gp_ensemble):
            _, sigma = gp.predict(X, return_std=True)
            variance += self.ensemble_weights[i] * sigma.reshape(-1, 1)
        return variance

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

        ei_normalized = (ei - np.min(ei)) / (np.max(ei) - np.min(ei)) if np.max(ei) != np.min(ei) else np.zeros_like(ei)
        diversity_normalized = (diversity - np.min(diversity)) / (np.max(diversity) - np.min(diversity)) if np.max(diversity) != np.min(diversity) else np.zeros_like(diversity)

        F = np.hstack([ei_normalized, diversity_normalized])

        is_efficient = np.ones(F.shape[0], dtype=bool)
        for i, c in enumerate(F):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(F[is_efficient] >= c, axis=1)
                is_efficient[i] = True

        pareto_front = candidates[is_efficient]

        if len(pareto_front) > 0:
            ei_pareto = self._expected_improvement(pareto_front)
            variance_pareto = self._ensemble_variance(pareto_front)
            # Select the point with the highest EI and variance
            scores = self.pareto_influence * ei_pareto + (1 - self.pareto_influence) * variance_pareto
            next_point = pareto_front[np.argmax(scores)].reshape(1, -1)
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
            self.pareto_influence = min(self.pareto_influence * self.pareto_increase, 1.0)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)
            self.pareto_influence = max(self.pareto_influence * self.pareto_decay, 0.0)

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
 The algorithm ParetoEnsembleAdaptiveTrustRegionBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1784 with standard deviation 0.1091.

took 37.47 seconds to run.