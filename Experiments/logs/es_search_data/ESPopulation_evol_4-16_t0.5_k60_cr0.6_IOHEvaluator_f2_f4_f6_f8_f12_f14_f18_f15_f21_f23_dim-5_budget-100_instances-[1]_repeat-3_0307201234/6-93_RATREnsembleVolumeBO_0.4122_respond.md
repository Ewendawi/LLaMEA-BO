# Description
**Regret-Informed Adaptive Trust Region Ensemble Volume Bayesian Optimization (RATREnsembleVolumeBO):** This algorithm combines the strengths of AHTRBO_DER and ATREnsembleVolumeBO, focusing on enhancing the trust region adaptation and exploration-exploitation balance. It incorporates a hybrid surrogate model (Gaussian Process and Gradient Boosting), an adaptive trust region, dynamic batch size adjustment, and volume-aware exploration. The GP/GB model weight is dynamically adjusted based on observed regret, and the trust region size and exploration factor are adapted based on model agreement, GP uncertainty, and EI values. It also includes a refined volume-aware exploration strategy and a dynamic diversity weight adjustment based on GP uncertainty. A key addition is the use of a dynamic exploration factor that adapts based on both the trust region size and the remaining budget, promoting more exploration when the trust region is small or the budget is large, and more exploitation when the trust region is large or the budget is small.

# Justification
This algorithm builds upon the existing AHTRBO_DER and ATREnsembleVolumeBO to improve performance by addressing the limitations of each.
*   **Hybrid Surrogate Model (GP & GB):** Combines the strengths of both models. GP provides uncertainty estimates, while GB handles complex relationships.
*   **Adaptive Trust Region:** Dynamically adjusts the trust region size based on model agreement, EI value, and GP uncertainty to balance exploration and exploitation.
*   **Dynamic Batch Size:** Adjusts the batch size based on the remaining budget to efficiently utilize function evaluations.
*   **Volume-Aware Exploration:** Encourages exploration in under-sampled regions using a combination of k-NN distances and distances to existing points.
*   **Regret-Based GP Weight Adjustment:** Dynamically adjusts the GP/GB model weight based on the observed regret (difference between predicted and actual function values), giving more weight to the model with lower regret. This is taken from AHTRBO_DER.
*   **Dynamic Exploration Factor:** Adapts based on both the trust region size and the remaining budget. When the trust region is small or the budget is large, the algorithm explores more. When the trust region is large or the budget is small, the algorithm exploits more.
*   **Dynamic Diversity Weight:** Adjusts the diversity weight based on GP uncertainty, encouraging more diversity when the GP is uncertain and more exploitation when the GP is confident.

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


class RATREnsembleVolumeBO:
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
        self.trust_region_decay = 0.95

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
        sigma = sigma.reshape(-1, 1)
        mu_gb = self.gb_model.predict(X).reshape(-1, 1)

        # Weighted average of GP and GB predictions
        mu = self.gp_weight * mu_gp + (1 - self.gp_weight) * mu_gb
        sigma = np.maximum(sigma, 1e-6) # Prevent division by zero

        imp = best_y - mu
        z = imp / sigma
        ei = imp * norm.cdf(z) + sigma * norm.pdf(z)

        # Volume-aware exploration
        if self.X is not None:
            distances, _ = self.knn.kneighbors(X)
            avg_distances = np.mean(distances, axis=1).reshape(-1, 1)
            min_distances = cdist(X, self.X).min(axis=1).reshape(-1, 1)
            volume_term = 0.7 * avg_distances + 0.3 * min_distances  # Weighted combination
            ei += self.diversity_weight * self.exploration_factor * volume_term

        return ei

    def _select_next_points(self, batch_size):
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        x_starts = best_x + np.random.normal(0, 0.1, size=(batch_size, self.dim))
        x_starts = np.clip(x_starts, self.bounds[0], self.bounds[1])

        candidates = []
        values = []
        for x_start in x_starts:
            lower_bound = np.maximum(x_start - self.trust_region_size / 2, self.bounds[0])
            upper_bound = np.minimum(x_start + self.trust_region_size / 2, self.bounds[1])

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

            # Trust region update based on model agreement, EI, and GP uncertainty
            ei_values = self._expected_improvement(X_next, np.min(self.y))
            mean_ei = np.mean(ei_values)

            if np.mean(agreement) < 1.0 and mean_ei > 0.01 and mean_sigma < 1.0: #Added GP uncertainty condition
                self.trust_region_size *= 1.1
            else:
                self.trust_region_size *= 0.9
            
            self.trust_region_size *= self.trust_region_decay
            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0)

            # Dynamic exploration factor adjustment based on trust region and budget
            self.exploration_factor = (0.1 + 0.9 * np.exp(-self.trust_region_size / 2)) * (0.5 + (self.budget - self.n_evals) / self.budget)

            # Regret-based GP weight adjustment
            gp_regret = np.abs(mu_gp - y_next)
            gb_regret = np.abs(mu_gb - y_next)

            if np.mean(gp_regret) < np.mean(gb_regret):
                self.gp_weight = min(1.0, self.gp_weight + 0.05)
            else:
                self.gp_weight = max(0.0, self.gp_weight - 0.05)

            # Dynamic diversity weight adjustment based on GP uncertainty
            self.diversity_weight = 0.001 + 0.099 * np.exp(-mean_sigma)

            self.gp_model = self._fit_gp_model(self.X, self.y)
            self.gb_model = self._fit_gb_model(self.X, self.y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        return best_y, best_x
```
## Feedback
 The algorithm RATREnsembleVolumeBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1837 with standard deviation 0.1036.

took 910.35 seconds to run.