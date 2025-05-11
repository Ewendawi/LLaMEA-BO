# Description
**AHTRBO-DEREIDAD: Adaptive Hybrid Trust Region Bayesian Optimization with Dynamic Exploration, Regret-Based GP Weighting, EI, Adaptive Diversity, and Error-Aware Trust Region Adjustment.** This algorithm combines the strengths of AHTRBO_DER and AHTRBO_EIDAD, further refining the exploration-exploitation balance and surrogate model accuracy. It utilizes a hybrid surrogate model (Gaussian Process and Gradient Boosting), adaptive trust region, dynamic batch size, and volume-aware exploration. Key innovations include regret-based GP weighting, dynamic exploration factor based on trust region size, adaptive diversity weight, and a trust region adjustment mechanism informed by both model agreement and prediction error. The algorithm also incorporates a trust region restart mechanism to escape potential stagnation. This version introduces an error-aware trust region adjustment, which considers the magnitude of the prediction error of the hybrid model when adjusting the trust region size, aiming to prevent premature shrinkage when the model is uncertain.

# Justification
This algorithm builds upon the best aspects of AHTRBO_DER and AHTRBO_EIDAD to create a more robust and efficient optimization strategy.

*   **Hybrid Surrogate Model (GP and GB):** Combines the strengths of both GP (uncertainty quantification) and GB (efficient learning of complex functions).
*   **Adaptive Trust Region:** Dynamically adjusts the trust region size based on model agreement and, crucially, prediction error. This error-aware adjustment is designed to prevent premature trust region shrinkage when the models are uncertain, leading to better exploration.
*   **Dynamic Batch Size:** Adjusts the batch size based on the remaining budget, allowing for more evaluations early in the optimization and fewer evaluations later.
*   **Regret-Based GP Weighting:** Dynamically adjusts the GP/GB model weight based on the observed regret (difference between predicted and actual function values), giving more weight to the model that is performing better.
*   **Adaptive Diversity Weight:** Adjusts the diversity weight based on the trust region size and model agreement, promoting more exploration when the trust region is large or model disagreement is high.
*   **Trust Region Restart:** Restarts the optimization from a new random location if the trust region shrinks below a certain threshold, indicating potential stagnation.
*   **Error-Aware Trust Region Adjustment:** The trust region size is adjusted based on both model agreement and the magnitude of prediction errors. If the prediction error is high, the trust region shrinks less, allowing for more exploration in uncertain regions. This helps to prevent premature convergence and improve the algorithm's ability to escape local optima.

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
from sklearn.impute import SimpleImputer
from scipy.stats import norm
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


class AHTRBO_DEREIDAD:
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
        self.diversity_weight = 0.01  # Initial value
        self.imputer = SimpleImputer(strategy='mean')
        self.epsilon = 1e-6
        self.gp_weight = 0.5  # Initial weight for GP model
        self.batch_size = 1
        self.knn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree')
        self.min_trust_region_size = 0.1
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

        # Volume-aware exploration with adaptive diversity weight
        distances, _ = self.knn.kneighbors(X)
        avg_distances = np.mean(distances, axis=1).reshape(-1, 1)
        ei += self.diversity_weight * self.exploration_factor * avg_distances

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
            mean_agreement = np.mean(agreement)

            # Error-aware trust region adjustment
            prediction_error = np.abs(y_pred - y_next)
            mean_prediction_error = np.mean(prediction_error)

            if mean_agreement < 1.0:
                tr_increase = 1.1
            else:
                tr_increase = 0.9

            # Dampen trust region shrinkage if prediction error is high
            if mean_prediction_error > 1.0:
                tr_increase = max(tr_increase, 0.95)  # Reduce shrinkage

            self.trust_region_size *= tr_increase
            self.trust_region_size *= self.trust_region_decay
            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0)

            # Trust region restart mechanism
            if self.trust_region_size < self.min_trust_region_size:
                X_init = self._sample_points(self.n_init)
                y_init = self._evaluate_points(func, X_init)
                self._update_eval_points(X_init, y_init)
                self.trust_region_size = 2.0  # Reset trust region size
                self.gp_model = self._fit_gp_model(self.X, self.y)
                self.gb_model = self._fit_gb_model(self.X, self.y)
                continue  # Skip the rest of the loop and restart

            # Dynamic exploration factor adjustment based on trust region
            self.exploration_factor = 0.1 + 0.9 * np.exp(-self.trust_region_size / 2)

            # Regret-based GP weight adjustment
            gp_regret = np.abs(mu_gp - y_next)
            gb_regret = np.abs(mu_gb - y_next)

            if np.mean(gp_regret) < np.mean(gb_regret):
                self.gp_weight = min(1.0, self.gp_weight + 0.05)
            else:
                self.gp_weight = max(0.0, self.gp_weight - 0.05)

            # Adaptive diversity weight adjustment
            self.diversity_weight = 0.01 + 0.1 * (self.trust_region_size / 5.0) + 0.1 * (1.0 if mean_agreement > 1.0 else 0.0)
            self.diversity_weight = np.clip(self.diversity_weight, 0.01, 0.1)  # Keep it within reasonable bounds

            self.gp_model = self._fit_gp_model(self.X, self.y)
            self.gb_model = self._fit_gb_model(self.X, self.y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        return best_y, best_x
```
## Feedback
 The algorithm AHTRBO_DEREIDAD got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1782 with standard deviation 0.0933.

took 896.85 seconds to run.