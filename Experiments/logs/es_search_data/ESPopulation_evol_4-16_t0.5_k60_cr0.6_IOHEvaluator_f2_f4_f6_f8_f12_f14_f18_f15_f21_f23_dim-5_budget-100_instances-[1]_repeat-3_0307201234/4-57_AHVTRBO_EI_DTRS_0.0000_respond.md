# Description
**Adaptive Hybrid Volume Trust Region Bayesian Optimization with Expected Improvement and Dynamic Trust Region Shape (AHVTRBO-EI-DTRS):** This algorithm builds upon the HybridVolumeTrustRegionBO by incorporating the Expected Improvement (EI) acquisition function, dynamically adjusting the trust region shape, and using a more robust method for selecting initial points. It retains the hybrid surrogate model (Gaussian Process and Gradient Boosting) and volume-aware exploration strategy. The trust region is adapted not only in size but also in shape based on the local curvature of the surrogate model. The initial sampling is improved using a Latin Hypercube design.

# Justification
The following refinements are made to the HybridVolumeTrustRegionBO algorithm:

1.  **Expected Improvement (EI) Acquisition Function:** Replacing the Lower Confidence Bound (LCB) with EI offers a more principled way to balance exploration and exploitation. EI directly estimates the expected improvement over the current best objective value, which can lead to faster convergence.
2.  **Dynamic Trust Region Shape:** Adapting the trust region shape, rather than just the size, allows the algorithm to better follow the contours of the objective function. This is achieved by estimating the local curvature using the Hessian of the Gradient Boosting model and scaling the trust region dimensions accordingly.
3.  **Improved Initial Sampling:** Latin Hypercube Sampling (LHS) provides a more uniform coverage of the search space compared to Sobol sequences, especially in higher dimensions. This can lead to better initial models and faster convergence.
4. **Batch Size Adjustment:** Dynamically adjusting the batch size based on the trust region size and the remaining budget. This allows for more efficient exploration in the early stages and focused exploitation in later stages.

These changes aim to improve the exploration-exploitation balance and adapt the search strategy to the local characteristics of the objective function, leading to better performance on the BBOB test suite.

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
from scipy.linalg import LinAlgError


class AHVTRBO_EI_DTRS:
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
        self.batch_size = 1

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim, seed=42)
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

        mu = self.gp_weight * mu_gp + (1 - self.gp_weight) * mu_gb

        imp = best_y - mu
        Z = imp / (sigma + 1e-9)
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 0] = 0.0
        return ei

    def _acquisition_function(self, X, best_y):
        ei = self._expected_improvement(X, best_y)

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

        return -ei  # Minimize negative EI

    def _estimate_local_curvature(self, x):
        try:
            hessian = self.gb_model.predict(x.reshape(1, -1), pred_deriv=1)[1].reshape(self.dim, self.dim)
            eigenvalues, _ = np.linalg.eig(hessian)
            return np.abs(eigenvalues)
        except (LinAlgError, ValueError) as e:
            return np.ones(self.dim)  # Return ones if Hessian cannot be computed

    def _select_next_points(self, batch_size):
        best_idx = np.argmin(self.y)
        best_x = self.X[best_idx]
        best_y = self.y[best_idx]

        x_starts = best_x + np.random.normal(0, 0.1, size=(batch_size, self.dim))
        x_starts = np.clip(x_starts, self.bounds[0], self.bounds[1])

        candidates = []
        values = []
        for x_start in x_starts:
            # Adaptive trust region shape
            curvature = self._estimate_local_curvature(x_start)
            lower_bound = np.maximum(x_start - self.trust_region_size / 2 * curvature, self.bounds[0])
            upper_bound = np.minimum(x_start + self.trust_region_size / 2 * curvature, self.bounds[1])

            res = minimize(lambda x: self._acquisition_function(x.reshape(1, -1), best_y),
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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        self.gp_model = self._fit_gp_model(self.X, self.y)
        self.gb_model = self._fit_gb_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Dynamic batch size adjustment
            self.batch_size = min(int((self.budget - self.n_evals) / 5), 5) if self.trust_region_size > 0.5 else 1
            self.batch_size = max(1, self.batch_size)

            X_next = self._select_next_points(self.batch_size)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

            # Adaptive trust region adjustment
            mu_gp, sigma = self.gp_model.predict(X_next, return_std=True)
            mu_gp = mu_gp.reshape(-1, 1)
            mu_gb = self.gb_model.predict(X_next).reshape(-1, 1)
            y_pred = self.gp_weight * mu_gp + (1 - self.gp_weight) * mu_gb

            agreement = np.abs(y_pred - y_next)

            if np.mean(agreement) < 1.0:
                self.trust_region_size *= 1.1
            else:
                self.trust_region_size *= 0.9

            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0)

            # Dynamic exploration factor adjustment
            self.exploration_factor = 1.0 + (self.budget - self.n_evals) / self.budget

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
## Error
 Traceback (most recent call last):
  File "<AHVTRBO_EI_DTRS>", line 149, in __call__
 149->             X_next = self._select_next_points(self.batch_size)
  File "<AHVTRBO_EI_DTRS>", line 109, in _select_next_points
 109->             curvature = self._estimate_local_curvature(x_start)
  File "<AHVTRBO_EI_DTRS>", line 91, in _estimate_local_curvature
  89 |     def _estimate_local_curvature(self, x):
  90 |         try:
  91->             hessian = self.gb_model.predict(x.reshape(1, -1), pred_deriv=1)[1].reshape(self.dim, self.dim)
  92 |             eigenvalues, _ = np.linalg.eig(hessian)
  93 |             return np.abs(eigenvalues)
TypeError: HistGradientBoostingRegressor.predict() got an unexpected keyword argument 'pred_deriv'
