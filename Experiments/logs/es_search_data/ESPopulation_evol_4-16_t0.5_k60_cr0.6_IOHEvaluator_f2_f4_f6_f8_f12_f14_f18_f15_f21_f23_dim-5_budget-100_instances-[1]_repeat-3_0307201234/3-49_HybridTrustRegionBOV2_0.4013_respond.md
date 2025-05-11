# Description
**HybridTrustRegionBO-V2 (HTRBO-V2):** This algorithm builds upon HybridTrustRegionBO by incorporating several key improvements. First, it introduces a more sophisticated trust region adaptation mechanism that considers both the prediction error and the uncertainty of the Gaussian Process model. Second, it refines the diversity term in the acquisition function by using a volume-aware approach, which encourages exploration in less-visited regions more effectively. Third, it employs a dynamic batch size strategy, adjusting the number of points evaluated in each iteration based on the trust region size and the remaining budget. Finally, it uses the GP's uncertainty to dynamically adjust the exploration factor, balancing exploration and exploitation more effectively.

# Justification
The key components and changes are justified as follows:

1.  **Trust Region Adaptation:** The original HybridTrustRegionBO adjusts the trust region size based on the agreement between the predicted and actual function values. HTRBO-V2 improves this by also considering the uncertainty (sigma) of the GP model. If the model is highly uncertain, a larger trust region might be beneficial, even if the prediction error is small. This helps to escape local optima more effectively.

2.  **Volume-Aware Diversity:** The diversity term in the original HybridTrustRegionBO simply considers the minimum distance to existing points. HTRBO-V2 uses KMeans clustering to identify clusters of existing points and then calculates the minimum distance to each cluster. This volume-aware approach encourages exploration in regions that are far from existing clusters, rather than just individual points, leading to a more comprehensive exploration of the search space.

3.  **Dynamic Batch Size:** The original HybridTrustRegionBO uses a fixed batch size. HTRBO-V2 dynamically adjusts the batch size based on the trust region size and the remaining budget. A larger trust region allows for a larger batch size, as the algorithm is more confident in the region. The batch size is also limited by the remaining budget to avoid over-evaluation.

4.  **Dynamic Exploration Factor:** The original HybridTrustRegionBO adjusts the exploration factor linearly based on the remaining budget. HTRBO-V2 refines this by incorporating the GP model's uncertainty. Higher uncertainty implies greater potential for improvement, justifying a higher exploration factor. This helps to balance exploration and exploitation more effectively.

5. **GP and GB model handling:** The algorithm uses both GP and GB models, weighting them based on their accuracy. This allows us to leverage the strengths of both models and adapt to the function's landscape.

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
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import cdist


class HybridTrustRegionBOV2:
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
        mu_gp, sigma = self.gp_model.predict(X, return_std=True)
        mu_gp = mu_gp.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        mu_gb = self.gb_model.predict(X).reshape(-1, 1)

        # Weighted average of GP and GB predictions
        mu = self.gp_weight * mu_gp + (1 - self.gp_weight) * mu_gb

        # Lower Confidence Bound
        lcb = mu - self.exploration_factor * sigma

        # Volume-aware diversity term
        if self.X is not None and len(self.X) > 5:
            n_clusters = min(5, len(self.X))
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init = 'auto').fit(self.X)
            distances = cdist(X, kmeans.cluster_centers_)
            min_distances = np.min(distances, axis=1).reshape(-1, 1)
            lcb -= self.diversity_weight * self.exploration_factor * min_distances
        return lcb

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
            # Dynamic batch size adjustment
            batch_size = min(int(self.trust_region_size), self.budget - self.n_evals)
            batch_size = max(1, batch_size)

            X_next = self._select_next_points(batch_size)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

            # Adaptive trust region adjustment
            mu_gp, sigma = self.gp_model.predict(X_next, return_std=True)
            mu_gp = mu_gp.reshape(-1, 1)
            mu_gb = self.gb_model.predict(X_next).reshape(-1, 1)
            y_pred = self.gp_weight * mu_gp + (1 - self.gp_weight) * mu_gb

            agreement = np.abs(y_pred - y_next)

            # Trust region adjustment based on agreement and uncertainty
            if np.mean(agreement) < 1.0:
                self.trust_region_size *= 1.1 + np.mean(sigma) * 0.1
            else:
                self.trust_region_size *= 0.9 - np.mean(sigma) * 0.1

            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0)

            # Dynamic exploration factor adjustment
            self.exploration_factor = 1.0 + (self.budget - self.n_evals) / self.budget + np.mean(sigma)

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
 The algorithm HybridTrustRegionBOV2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1790 with standard deviation 0.0986.

took 899.04 seconds to run.