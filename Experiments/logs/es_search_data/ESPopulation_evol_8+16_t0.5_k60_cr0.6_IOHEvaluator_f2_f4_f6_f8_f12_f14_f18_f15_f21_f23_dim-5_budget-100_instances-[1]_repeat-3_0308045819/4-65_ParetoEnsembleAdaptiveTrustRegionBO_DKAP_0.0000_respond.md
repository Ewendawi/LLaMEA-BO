# Description
**ParetoEnsembleAdaptiveTrustRegionBO with Dynamic Kernel Learning and Adaptive Pareto Influence (PEATRBO-DKAP):** This algorithm refines the ParetoEnsembleAdaptiveTrustRegionBO (PEATRBO) by incorporating dynamic kernel learning for the Gaussian Process (GP) ensemble and an adaptive strategy for adjusting the Pareto front's influence. The GP kernels are dynamically optimized using L-BFGS-B to better fit the data within the trust region. The Pareto influence is adapted based on the trust region's success and the correlation between EI and diversity, allowing for a more intelligent exploration-exploitation balance. The diversity metric is also enhanced by considering the average distance to the k-nearest neighbors instead of just the nearest neighbor, promoting a more robust diversity measure.

# Justification
*   **Dynamic Kernel Learning:** Optimizing the GP kernel parameters (length scale and constant value) within the trust region allows the GP models to better capture the local landscape of the objective function, leading to more accurate predictions and improved acquisition function values. L-BFGS-B is used for efficient kernel optimization.
*   **Adaptive Pareto Influence:** The original algorithm adjusts the Pareto influence based solely on the trust region's success. This version adds a correlation-based adjustment. If EI and diversity are highly correlated (meaning exploring diverse regions also leads to high EI), the Pareto influence is decreased to focus more on exploitation. Conversely, if they are uncorrelated, the influence is increased to encourage exploration.
*   **Enhanced Diversity Metric:** Using the average distance to the k-nearest neighbors provides a more robust diversity measure than just the distance to the nearest neighbor. This helps prevent the algorithm from getting stuck in local optima by encouraging exploration of regions that are not only far from the closest point but also relatively far from other points in the dataset.
*   **Computational Efficiency:** While kernel learning adds computational overhead, it is performed within the trust region, limiting the computational cost. The use of L-BFGS-B and efficient nearest neighbor search helps to maintain a reasonable runtime.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc, norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize

class ParetoEnsembleAdaptiveTrustRegionBO_DKAP:
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
        self.k_nearest = 5 # Number of nearest neighbors for diversity metric

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
                # Dynamic Kernel Learning
                def obj_func(theta):
                    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=theta, length_scale_bounds="fixed")
                    gp.kernel = kernel
                    gp.fit(X_train, y_train)
                    return -gp.log_marginal_likelihood(gp.kernel_.hyperparameters, clone_kernel=False)

                initial_length_scale = gp.kernel_.get_params()['k2__length_scale']
                res = minimize(obj_func, initial_length_scale, method='L-BFGS-B', bounds=[(1e-5, 5.0)])
                best_length_scale = res.x[0]
                gp.kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=best_length_scale, length_scale_bounds="fixed")

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
            knn = NearestNeighbors(n_neighbors=min(self.k_nearest, len(self.X)), algorithm='ball_tree').fit(self.X)
            distances, _ = knn.kneighbors(X)
            avg_distances = np.mean(distances, axis=1)
            return avg_distances.reshape(-1, 1)

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

        # Adaptive Pareto Influence based on correlation
        if self.X is not None and len(self.X) > 5:
            ei_vals = self._expected_improvement(self.X).flatten()
            diversity_vals = self._diversity_metric(self.X).flatten()
            correlation = np.corrcoef(ei_vals, diversity_vals)[0, 1]

            if np.isnan(correlation):
                correlation = 0.0

            if correlation > 0.5:  # High correlation, focus on exploitation
                self.pareto_influence = max(self.pareto_influence * 0.9, 0.1)
            elif correlation < -0.5:  # Negative correlation, focus on exploration
                self.pareto_influence = min(self.pareto_influence * 1.1, 0.9)

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
  File "<ParetoEnsembleAdaptiveTrustRegionBO_DKAP>", line 198, in __call__
 198->             self._fit_model(self.X, self.y)
  File "<ParetoEnsembleAdaptiveTrustRegionBO_DKAP>", line 74, in _fit_model
  74->                 res = minimize(obj_func, initial_length_scale, method='L-BFGS-B', bounds=[(1e-5, 5.0)])
  File "<ParetoEnsembleAdaptiveTrustRegionBO_DKAP>", line 71, in obj_func
  69 |                     gp.kernel = kernel
  70 |                     gp.fit(X_train, y_train)
  71->                     return -gp.log_marginal_likelihood(gp.kernel_.hyperparameters, clone_kernel=False)
  72 | 
  73 |                 initial_length_scale = gp.kernel_.get_params()['k2__length_scale']
ValueError: theta has not the correct number of entries. Should be 0; given are 2
