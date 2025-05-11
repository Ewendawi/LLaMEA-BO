# Description
**Adaptive Trust Region with Hybrid Surrogate and Volume-Aware Acquisition (ATHR-Hybrid-VolumeBO):** This algorithm builds upon the HybridVolumeEI_DBS_BO framework by introducing a more sophisticated trust region adaptation mechanism. Instead of relying solely on the agreement between GP and GB models, it incorporates a measure of the uncertainty in the region, as predicted by the GP model. The trust region size is adjusted based on the model agreement, EI value, and the GP's predicted variance. This allows for a more nuanced balance between exploration and exploitation. Additionally, the algorithm dynamically adjusts the diversity weight based not only on the trust region size but also on the remaining budget, promoting exploration earlier in the optimization process. A final improvement is the use of a more robust method for selecting the starting points for the local optimization within the trust region.

# Justification
The algorithm incorporates several key improvements:

1.  **Trust Region Adaptation:** The previous algorithms adjusted the trust region size based on the agreement between the GP and GB models, and in one case, the EI value. This new algorithm adds the GP's predicted variance to the trust region adaptation. High variance suggests greater uncertainty, prompting an increase in the trust region size to encourage exploration. This is crucial for navigating complex search spaces.
2.  **Diversity Weight Adjustment:** The diversity weight is now adjusted based on both the trust region size and the remaining budget. A higher diversity weight is applied earlier in the optimization process (when the budget is large) to promote broader exploration. As the budget dwindles, the diversity weight decreases, favoring exploitation of promising regions.
3.  **Robust Starting Points:** Instead of simply sampling from a normal distribution around the current best point, the algorithm now samples multiple points within the trust region and selects the point with the highest EI value as the starting point for local optimization. This helps to avoid getting stuck in local optima.
4.  **Hybrid Surrogate Model:** Maintains the effective hybrid surrogate model (GP and Gradient Boosting) for robust and accurate function approximation.
5.  **Volume-Aware Exploration:** The algorithm continues to incorporate volume awareness into the EI calculation, encouraging exploration of less-sampled regions.

These changes are designed to address the limitations of the previous algorithms by providing a more adaptive and robust framework for balancing exploration and exploitation.

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
from scipy.stats import norm
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from scipy.stats import uniform


class ATHRHVolumeBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim
        self.trust_region_size = 2.0
        self.exploration_factor = 1.0
        self.diversity_weight = 0.01
        self.imputer = SimpleImputer(strategy='mean')
        self.epsilon = 1e-6
        self.gp_weight = 0.5  # Initial weight for GP model
        self.batch_size = 1
        self.knn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree')

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        samples = sampler.random(n=n_points)
        return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_gp_model(self, X, y):
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, nu=2.5)
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
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

    def _expected_improvement(self, X, best_y):
        mu_gp, sigma = self.gp_model.predict(X, return_std=True)
        mu_gp = mu_gp.reshape(-1, 1)
        sigma = sigma = np.maximum(sigma.reshape(-1, 1), 1e-6)
        mu_gb = self.gb_model.predict(X).reshape(-1, 1)

        # Weighted average of GP and GB predictions
        mu = self.gp_weight * mu_gp + (1 - self.gp_weight) * mu_gb

        imp = best_y - mu
        z = imp / sigma
        ei = imp * norm.cdf(z) + sigma * norm.pdf(z)

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
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        candidates = []
        values = []
        for _ in range(batch_size):
            # Sample multiple starting points within the trust region
            n_samples = 10 * self.dim
            lower_bound = np.maximum(best_x - self.trust_region_size / 2, self.bounds[0])
            upper_bound = np.minimum(best_x + self.trust_region_size / 2, self.bounds[1])
            x_samples = uniform.rvs(loc=lower_bound, scale=upper_bound - lower_bound, size=(n_samples, self.dim))

            # Select the starting point with the highest EI
            ei_values = self._expected_improvement(x_samples, best_y)
            best_sample_idx = np.argmax(ei_values)
            x_start = x_samples[best_sample_idx]

            res = minimize(lambda x: -self._expected_improvement(x.reshape(1, -1), best_y),
                           x_start,
                           bounds=np.array([lower_bound, upper_bound]).T,
                           method="L-BFGS-B")
            candidates.append(res.x)
            values.append(-res.fun)

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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        self.gp_model = self._fit_gp_model(self.X, self.y)
        self.gb_model = self._fit_gb_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Dynamic batch size adjustment
            self.batch_size = int(np.ceil((self.budget - self.n_evals) / 50.0))
            self.batch_size = max(1, min(self.batch_size, 10))  # Limit batch size

            X_next = self._select_next_points(self.batch_size)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

            # Adaptive trust region adjustment
            mu_gp, sigma = self.gp_model.predict(X_next, return_std=True)
            mu_gp = mu_gp.reshape(-1, 1)
            mu_gb = self.gb_model.predict(X_next).reshape(-1, 1)
            y_pred = self.gp_weight * mu_gp + (1 - self.gp_weight) * mu_gb

            agreement = np.abs(y_pred - y_next)
            mean_sigma = np.mean(sigma)

            # Trust region update based on model agreement, EI, and GP variance
            ei_values = self._expected_improvement(X_next, np.min(self.y))
            mean_ei = np.mean(ei_values)

            if np.mean(agreement) < 1.0 and mean_ei > 0.01 and mean_sigma > 0.1:
                self.trust_region_size *= 1.1
            else:
                self.trust_region_size *= 0.9

            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0)

            # Dynamic exploration factor adjustment
            self.exploration_factor = 0.5 + (self.budget - self.n_evals) / self.budget

            # Adaptive GP weight adjustment
            gp_error = np.mean(np.abs(mu_gp - y_next))
            gb_error = np.mean(np.abs(mu_gb - y_next))

            if gp_error < gb_error:
                self.gp_weight = min(1.0, self.gp_weight + 0.05)
            else:
                self.gp_weight = max(0.0, self.gp_weight - 0.05)

            # Dynamic diversity weight adjustment
            remaining_budget_ratio = (self.budget - self.n_evals) / self.budget
            self.diversity_weight = 0.001 + 0.099 * np.exp(-self.trust_region_size) * remaining_budget_ratio

            self.gp_model = self._fit_gp_model(self.X, self.y)
            self.gb_model = self._fit_gb_model(self.X, self.y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        return best_y, best_x
```
## Feedback
 The algorithm ATHRHVolumeBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1833 with standard deviation 0.1000.

took 1086.59 seconds to run.