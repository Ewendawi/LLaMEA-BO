# Description
**Adaptive Hybrid Volume Trust Region BO with Expected Improvement and Dynamic Trust Region Scaling (AHVTRBO-EI-DTRS):** This algorithm enhances the HybridVolumeTrustRegionBO by incorporating the Expected Improvement (EI) acquisition function, a more sophisticated dynamic trust region scaling mechanism, and adaptive scaling of the exploration factor. It retains the hybrid surrogate model (Gaussian Process and Gradient Boosting) and volume-aware exploration strategy. The trust region size is dynamically adjusted based on the agreement between the GP and GB models and the observed function values, using a more nuanced scaling factor. The exploration factor is adapted based on the remaining budget and the trust region size.

# Justification
The key improvements focus on refining the exploration-exploitation balance and trust region management:

1.  **Expected Improvement (EI) Acquisition:** EI is a more principled acquisition function than LCB, as it directly estimates the expected improvement over the current best value. This helps to balance exploration and exploitation more effectively.

2.  **Dynamic Trust Region Scaling (DTRS):** The trust region size is now adjusted based on both the model agreement (GP vs. GB) and the actual improvement observed. If the models agree and the improvement is significant, the trust region expands more aggressively. If the models disagree or the improvement is small, the trust region shrinks more cautiously. This adaptive scaling allows for more efficient exploration in promising regions and more focused exploitation when the models are confident.

3.  **Adaptive Exploration Factor Scaling:** The exploration factor is scaled based on the remaining budget and the current trust region size. When the budget is high and the trust region is small, the exploration factor is increased to encourage broader exploration. When the budget is low or the trust region is large, the exploration factor is decreased to focus on exploitation.

These changes aim to improve the algorithm's ability to find the global optimum by more effectively balancing exploration and exploitation and by dynamically adjusting the search space based on the model's confidence and the observed function values.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import minimize
from sklearn.metrics import pairwise_distances
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm


class AdaptiveHybridVolumeTrustRegionBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim
        self.trust_region_size = 2.0
        self.exploration_factor = 2.0
        self.diversity_weight = 0.1
        self.imputer = SimpleImputer(strategy='mean')
        self.epsilon = 1e-6
        self.gp_weight = 0.5  # Initial weight for GP model
        self.knn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree')
        self.best_y = np.inf  # Initialize best_y to infinity

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        samples = sampler.random(n=n_points)
        return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_gp_model(self, X, y):
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, nu=2.5)
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5, normalize_y=True)
        model.fit(X, y)
        return model

    def _fit_gb_model(self, X, y):
        # Impute missing values if any
        if np.isnan(X).any() or np.isnan(y).any():
            X = self.imputer.fit_transform(X)
            y = self.imputer.fit_transform(y)

        model = HistGradientBoostingRegressor(random_state=0)
        model.fit(X, y.ravel())
        return model

    def _expected_improvement(self, X, xi=0.01):
        mu_gp, sigma = self.gp_model.predict(X, return_std=True)
        mu_gp = mu_gp.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        mu_gb = self.gb_model.predict(X).reshape(-1, 1)

        mu = self.gp_weight * mu_gp + (1 - self.gp_weight) * mu_gb

        imp = mu - self.best_y - xi
        Z = imp / (sigma + 1e-9)
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Diversity term
        if self.X is not None and len(self.X) > 5:
            distances = cdist(X, self.X)
            min_distances = np.min(distances, axis=1).reshape(-1, 1)
            ei += self.diversity_weight * self.exploration_factor * min_distances

        # Volume-aware exploration
        if self.X is not None:
            distances, _ = self.knn.kneighbors(X)
            avg_distances = np.mean(distances, axis=1).reshape(-1, 1)
            ei += 0.01 * self.exploration_factor * avg_distances

        return ei

    def _select_next_points(self, batch_size):
        best_idx = np.argmin(self.y)
        best_x = self.X[best_idx]

        x_starts = best_x + np.random.normal(0, 0.1, size=(batch_size, self.dim))
        x_starts = np.clip(x_starts, self.bounds[0], self.bounds[1])

        candidates = []
        values = []
        for x_start in x_starts:
            lower_bound = np.maximum(x_start - self.trust_region_size / 2, self.bounds[0])
            upper_bound = np.minimum(x_start + self.trust_region_size / 2, self.bounds[1])

            res = minimize(lambda x: -self._expected_improvement(x.reshape(1, -1)),  # Negate EI for minimization
                           x_start,
                           bounds=np.array([lower_bound, upper_bound]).T,
                           method="L-BFGS-B")
            candidates.append(res.x)
            values.append(res.fun)

        return np.array(candidates)

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
        self.knn.fit(self.X)
        self.best_y = np.min(self.y)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        self.gp_model = self._fit_gp_model(self.X, self.y)
        self.gb_model = self._fit_gb_model(self.X, self.y)

        while self.n_evals < self.budget:
            batch_size = 1
            X_next = self._select_next_points(batch_size)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

            # Adaptive trust region adjustment
            mu_gp, sigma = self.gp_model.predict(X_next, return_std=True)
            mu_gp = mu_gp.reshape(-1, 1)
            mu_gb = self.gb_model.predict(X_next).reshape(-1, 1)
            y_pred = self.gp_weight * mu_gp + (1 - self.gp_weight) * mu_gb

            agreement = np.abs(mu_gp - mu_gb)  # Agreement between GP and GB

            improvement = self.y[-1] - y_next #check the sign

            scale_factor = 1.0
            if np.mean(agreement) < 0.5 and np.mean(improvement) > 0:  # Models agree and improvement is good
                scale_factor = 1.2
            elif np.mean(agreement) > 1.0 or np.mean(improvement) <= 0:  # Models disagree or no improvement
                scale_factor = 0.8

            self.trust_region_size *= scale_factor
            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0)

            # Dynamic exploration factor adjustment
            self.exploration_factor = 1.0 + (self.budget - self.n_evals) / self.budget * (0.5 + self.trust_region_size / 10)


            # Adaptive GP weight adjustment
            gp_error = np.mean(np.abs(mu_gp - y_next))
            gb_error = np.mean(np.abs(mu_gb - y_next))

            if gp_error < gb_error:
                self.gp_weight = min(1.0, self.gp_weight + 0.05)
            else:
                self.gp_weight = max(0.0, self.gp_weight - 0.05)

            self.gp_model = self._fit_gp_model(self.X, self.y)
            self.gb_model = self._fit_gb_model(self.X, self.y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        return best_y, best_x
```
## Feedback
 The algorithm AdaptiveHybridVolumeTrustRegionBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1547 with standard deviation 0.0932.

took 482.09 seconds to run.