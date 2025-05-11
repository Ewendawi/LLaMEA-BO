# Description
**HybridTrustRegionDEBO (HTRDEBO):** This algorithm fuses the hybrid modeling approach of HybridTrustRegionBO (HTRBO) with the dynamic diversity and exploration strategies of ATRDDEBO. It employs a hybrid surrogate model (Gaussian Process and Gradient Boosting) for improved prediction accuracy and leverages an adaptive trust region mechanism to balance exploration and exploitation. The acquisition function incorporates a diversity term to encourage exploration in less-visited regions, and both the exploration factor and diversity weight are dynamically adjusted based on the remaining budget, model uncertainty, and sample distribution. It introduces a more robust trust region adaptation based on the prediction variance of both GP and GB models.

# Justification
The HTRDEBO algorithm builds upon the strengths of both HTRBO and ATRDDEBO. HTRBO's hybrid surrogate model can better capture the function landscape compared to a single GP model, while ATRDDEBO's dynamic diversity and exploration strategies can help escape local optima and promote a more even spread of exploration.

Key changes and justifications:

1.  **Hybrid Surrogate Model:** Uses both Gaussian Process (GP) and HistGradientBoostingRegressor (GB) as surrogate models to leverage their complementary strengths. GP provides uncertainty quantification, while GB offers computational efficiency and can capture complex relationships.
2.  **Dynamic GP/GB Weight Adjustment:** Adaptively adjusts the weight of GP and GB predictions in the acquisition function based on their respective errors. This allows the algorithm to prioritize the more accurate model in different regions of the search space.
3.  **Adaptive Trust Region:** Dynamically adjusts the trust region size based on the agreement between the hybrid model's predictions and the actual function values. It increases the trust region when the model is accurate and decreases it when the model is inaccurate.
4.  **Diversity Term:** Incorporates a diversity term in the acquisition function to encourage exploration in less-visited regions. This helps prevent premature convergence to local optima. The diversity weight is dynamically adjusted based on the distribution of samples within the trust region.
5.  **Dynamic Exploration Factor:** Dynamically adjusts the exploration factor based on the remaining budget and the model's uncertainty. This allows the algorithm to balance exploration and exploitation more effectively.
6.  **Robust Trust Region Adjustment:** The trust region adjustment is made more robust by considering the prediction variance of both GP and GB models.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import cdist

class HybridTrustRegionDEBO:
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

    def _acquisition_function(self, X):
        mu_gp, sigma_gp = self.gp_model.predict(X, return_std=True)
        mu_gp = mu_gp.reshape(-1, 1)
        sigma_gp = sigma_gp.reshape(-1, 1)
        mu_gb = self.gb_model.predict(X).reshape(-1, 1)

        # Weighted average of GP and GB predictions
        mu = self.gp_weight * mu_gp + (1 - self.gp_weight) * mu_gb
        sigma = self.gp_weight * sigma_gp + (1 - self.gp_weight) * np.std(mu_gb) # Estimate sigma from GP and GB

        # Lower Confidence Bound
        lcb = mu - self.exploration_factor * sigma

        # Diversity term: encourage exploration in less-visited regions
        if self.X is not None and len(self.X) > 5:
            kmeans = KMeans(n_clusters=min(5, len(self.X), 10), random_state=0, n_init = 'auto').fit(self.X)
            clusters = kmeans.predict(X)
            distances = np.array([np.min(pairwise_distances(x.reshape(1, -1), self.X[kmeans.labels_ == cluster].reshape(-1, self.dim))) if np.sum(kmeans.labels_ == cluster) > 0 else 0 for x, cluster in zip(X, clusters)])
            diversity = distances.reshape(-1, 1)
        else:
            diversity = np.zeros_like(lcb)

        return lcb + self.diversity_weight * diversity

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

            res = minimize(lambda x: self._acquisition_function(x.reshape(1, -1)),
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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        self.gp_model = self._fit_gp_model(self.X, self.y)
        self.gb_model = self._fit_gb_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Adaptive batch size
            batch_size = min(int(np.ceil(self.trust_region_size)), 4)  # Adjust batch size based on trust region
            batch_size = max(1, batch_size) # Ensure batch size is at least 1

            X_next = self._select_next_points(batch_size)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

            # Adaptive trust region adjustment
            mu_gp, sigma_gp = self.gp_model.predict(X_next, return_std=True)
            mu_gp = mu_gp.reshape(-1, 1)
            sigma_gp = sigma_gp.reshape(-1, 1)
            mu_gb = self.gb_model.predict(X_next).reshape(-1, 1)

            y_pred = self.gp_weight * mu_gp + (1 - self.gp_weight) * mu_gb

            agreement = np.abs(y_pred - y_next)

            # Combine GP and GB uncertainty for trust region adjustment
            combined_sigma = self.gp_weight * sigma_gp + (1 - self.gp_weight) * np.std(mu_gb)
            agreement_scaled = agreement / (combined_sigma + self.epsilon)

            if np.mean(agreement_scaled) < 1.0:
                self.trust_region_size *= 1.1
            else:
                self.trust_region_size *= 0.9

            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0)

            # Dynamic exploration factor adjustment
            self.exploration_factor = 1.0 + (self.budget - self.n_evals) / self.budget
            self.exploration_factor += np.mean(combined_sigma) # Increase exploration with uncertainty
            self.exploration_factor = max(0.1, self.exploration_factor)

            # Dynamic diversity weight adjustment
            if self.X is not None and len(self.X) > 5:
                # Calculate average distance between samples within the trust region
                distances = cdist(self.X, self.X)
                avg_distance = np.mean(distances)

                # Adjust diversity weight based on average distance
                if avg_distance < self.trust_region_size / 2:
                    self.diversity_weight *= 1.1  # Increase diversity weight if samples are clustered
                else:
                    self.diversity_weight *= 0.9  # Decrease diversity weight if samples are well-distributed

                self.diversity_weight = np.clip(self.diversity_weight, 0.01, 0.5)

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
 The algorithm HybridTrustRegionDEBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1749 with standard deviation 0.1004.

took 1207.02 seconds to run.