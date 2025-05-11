# Description
**Hybrid Adaptive Trust Region with Gradient Boosting and Efficient Local Search Bayesian Optimization (HATR-DGB-ELSBO):** This algorithm synergistically integrates the strengths of ATRDGBO and ATRELSBO. It employs a gradient boosting regressor for surrogate modeling, adaptive trust region management for balancing exploration and exploitation, diversity injection to prevent premature convergence, and efficient local search to refine promising solutions. Crucially, it addresses the uncertainty estimation limitations of gradient boosting by incorporating a quantile regressor to estimate prediction intervals, which are then used to inform the acquisition function. This hybrid approach aims for robust global exploration and efficient local refinement, leveraging the speed of gradient boosting and the refined search capabilities of local search within an adaptive trust region framework.

# Justification
The HATR-DGB-ELSBO algorithm builds upon the strengths of both ATRDGBO and ATRELSBO while addressing their limitations. ATRDGBO uses a HistGradientBoostingRegressor, which is computationally efficient but does not natively provide uncertainty estimates. ATRELSBO uses a Gaussian Process Regressor, which provides uncertainty estimates but is computationally expensive. This hybrid approach uses the HistGradientBoostingRegressor for its speed, but adds a QuantileRegressor to estimate the uncertainty. This allows for a more informed exploration-exploitation trade-off within the trust region. The local search component, adapted from ATRELSBO, is retained to efficiently refine promising solutions found by the gradient boosting surrogate model. The adaptive trust region balances exploration and exploitation. The diversity term encourages exploration in less-visited regions.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingQuantileRegressor
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.impute import SimpleImputer


class HATRDGBELSBO:
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
        self.imputer = SimpleImputer(strategy='mean')  # Impute NaN values with the mean
        self.quantile_alpha = 0.1  # Quantile for uncertainty estimation

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        samples = sampler.random(n=n_points)
        return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Impute missing values if any
        if np.isnan(X).any() or np.isnan(y).any():
            X = self.imputer.fit_transform(X)
            y = self.imputer.fit_transform(y)

        model = HistGradientBoostingRegressor(random_state=0)
        model.fit(X, y.ravel())  # HistGradientBoostingRegressor expects a 1D array for y

        # Train quantile regressors for uncertainty estimation
        lower_quantile = self.quantile_alpha / 2
        upper_quantile = 1 - self.quantile_alpha / 2

        lower_model = HistGradientBoostingQuantileRegressor(random_state=0, quantile=lower_quantile)
        lower_model.fit(X, y.ravel())

        upper_model = HistGradientBoostingQuantileRegressor(random_state=0, quantile=upper_quantile)
        upper_model.fit(X, y.ravel())

        return model, lower_model, upper_model

    def _acquisition_function(self, X):
        mu = self.model.predict(X).reshape(-1, 1)
        lower_bound = self.lower_model.predict(X).reshape(-1, 1)
        upper_bound = self.upper_model.predict(X).reshape(-1, 1)
        sigma = (upper_bound - lower_bound) / 2  # Estimate sigma from quantile range.

        # Lower Confidence Bound
        lcb = mu - self.exploration_factor * sigma

        # Diversity term
        if self.X is not None and len(self.X) > 5:
            kmeans = KMeans(n_clusters=min(5, len(self.X)), random_state=0, n_init='auto').fit(self.X)
            clusters = kmeans.predict(X)
            distances = np.array([np.min(pairwise_distances(x.reshape(1, -1), self.X[kmeans.labels_ == cluster])) for x, cluster in zip(X, clusters)])
            diversity = distances.reshape(-1, 1)
        else:
            diversity = np.zeros_like(lcb)

        return lcb - self.diversity_weight * diversity  # Minimize LCB - diversity

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

        self.model, self.lower_model, self.upper_model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            batch_size = 1
            X_next = self._select_next_points(batch_size)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

            # Adaptive trust region adjustment
            y_pred = self.model.predict(X_next).reshape(-1, 1)
            agreement = np.abs(y_pred - y_next)

            if np.mean(agreement) < 1.0:
                self.trust_region_size *= 1.1
            else:
                self.trust_region_size *= 0.9

            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0)

            # Dynamic exploration factor adjustment
            self.exploration_factor = 1.0 + (self.budget - self.n_evals) / self.budget

            # Local search around the best point
            best_idx = np.argmin(self.y)
            best_x = self.X[best_idx]

            # Dynamically adjust the number of local search steps based on the remaining budget
            remaining_budget = self.budget - self.n_evals
            n_local_steps = min(5, remaining_budget)

            X_local = best_x + np.random.normal(0, 0.1, size=(n_local_steps, self.dim))
            X_local = np.clip(X_local, self.bounds[0], self.bounds[1])
            y_local = self._evaluate_points(func, X_local)
            self._update_eval_points(X_local, y_local)

            self.model, self.lower_model, self.upper_model = self._fit_model(self.X, self.y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<HATRDGBELSBO>", line 5, in <module>
   3 | import numpy as np
   4 | from sklearn.experimental import enable_hist_gradient_boosting
   5-> from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingQuantileRegressor
   6 | from scipy.optimize import minimize
   7 | from sklearn.cluster import KMeans
ImportError: cannot import name 'HistGradientBoostingQuantileRegressor' from 'sklearn.ensemble' (/home/hpcl2325/envs/llmbo/lib/python3.10/site-packages/sklearn/ensemble/__init__.py)
