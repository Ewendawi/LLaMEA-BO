# Description
**Adaptive Trust Region with Gaussian Process and Gradient Boosting Ensemble (ATRGPGEnsembleBO):** This algorithm combines the strengths of Gaussian Processes (GPs) and Gradient Boosting (GB) within an adaptive trust region framework. It uses an ensemble of GP and GB models to capture different aspects of the objective function's behavior. The GP provides uncertainty estimates, which are crucial for exploration, while the GB model offers potentially faster and more accurate predictions. The trust region mechanism balances exploration and exploitation. The acquisition function uses a combination of LCB and a diversity term, similar to ADTRBO, to encourage exploration in less-visited regions. The algorithm adaptively adjusts the trust region size and exploration factor based on the agreement between model predictions and actual function evaluations. To improve computational efficiency, the GP model is only refitted periodically, while the GB model is updated more frequently.

# Justification
This algorithm aims to improve upon ADTRBO and ATRDGBO by leveraging the complementary strengths of GPs and GB models.

*   **Ensemble Modeling:** Combining GP and GB models allows the algorithm to capture both smooth and non-smooth features of the objective function. GPs are well-suited for modeling smooth functions and providing uncertainty estimates, while GB models can handle complex, non-linear relationships more efficiently.
*   **Adaptive Trust Region:** The trust region mechanism, inherited from ADTRBO, helps to balance exploration and exploitation by restricting the search space around the current best solution.
*   **Diversity Injection:** The diversity term in the acquisition function, also from ADTRBO, encourages exploration in less-visited regions, preventing premature convergence.
*   **Periodic GP Refitting:** To reduce computational cost, the GP model is refitted less frequently than the GB model. This is based on the assumption that the GP captures the overall shape of the objective function, while the GB model adapts more quickly to local changes.
*   **Exploration-Exploitation Balance:** The exploration factor is dynamically adjusted based on the remaining budget, ensuring that the algorithm explores more in the early stages and exploits more in the later stages.
*   **NaN Handling:** The imputer is used to handle potential NaN values in the input data, ensuring that the algorithm is robust to missing data.

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

class ATRGPGEnsembleBO:
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
        self.gp_refit_interval = 5  # Refit GP every 5 iterations
        self.gp_model = None
        self.gb_model = None

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        samples = sampler.random(n=n_points)
        return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_gp_model(self, X, y):
        # Impute missing values if any
        if np.isnan(X).any():
            X = self.imputer.fit_transform(X)
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
        model.fit(X, y.ravel())  # HistGradientBoostingRegressor expects a 1D array for y
        return model

    def _acquisition_function(self, X):
        if np.isnan(X).any():
            X = self.imputer.transform(X)

        # Predict with GP
        mu_gp, sigma = self.gp_model.predict(X, return_std=True)
        mu_gp = mu_gp.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Predict with GB
        mu_gb = self.gb_model.predict(X).reshape(-1, 1)

        # Ensemble prediction (weighted average)
        mu = 0.5 * mu_gp + 0.5 * mu_gb

        # Lower Confidence Bound
        lcb = mu - self.exploration_factor * sigma

        # Diversity term
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
            batch_size = 1
            X_next = self._select_next_points(batch_size)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

            # Adaptive trust region adjustment
            mu_gp, sigma = self.gp_model.predict(X_next, return_std=True)
            mu_gp = mu_gp.reshape(-1, 1)
            y_pred = 0.5 * mu_gp + 0.5 * self.gb_model.predict(X_next).reshape(-1,1)
            agreement = np.abs(y_pred - y_next) / sigma.reshape(-1, 1)

            if np.mean(agreement) < 1.0:
                self.trust_region_size *= 1.1
            else:
                self.trust_region_size *= 0.9

            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0)

            # Dynamic exploration factor adjustment
            self.exploration_factor = 1.0 + (self.budget - self.n_evals) / self.budget

            # Refit models
            if self.n_evals % self.gp_refit_interval == 0:
                self.gp_model = self._fit_gp_model(self.X, self.y)
            self.gb_model = self._fit_gb_model(self.X, self.y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        return best_y, best_x
```
## Feedback
 The algorithm ATRGPGEnsembleBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1789 with standard deviation 0.0936.

took 1457.83 seconds to run.