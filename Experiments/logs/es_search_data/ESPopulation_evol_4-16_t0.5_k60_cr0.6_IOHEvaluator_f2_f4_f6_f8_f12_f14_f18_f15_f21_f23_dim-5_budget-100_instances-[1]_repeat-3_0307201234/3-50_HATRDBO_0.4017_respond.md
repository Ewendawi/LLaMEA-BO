# Description
**HybridAdaptiveTrustRegionDiversityBayesOpt (HATRDBO):** This algorithm synergistically integrates the strengths of HybridTrustRegionBO (HTRBO) and Adaptive Trust Region with Dynamic Diversity and Exploration Bayesian Optimization (ATRDDEBO). It employs a hybrid surrogate model (Gaussian Process and Gradient Boosting) for enhanced prediction accuracy and leverages an adaptive trust region to balance exploration and exploitation. The acquisition function incorporates a dynamic diversity term, inspired by ATRDDEBO, to promote exploration in less-visited regions. Furthermore, it introduces a dynamic mechanism to adjust the trust region size, exploration factor, and diversity weight, adapting to the function landscape and optimization progress. A key innovation is the use of a dynamic batch size, informed by both the trust region size and the uncertainty of the Gaussian Process model, to enhance sampling efficiency.

# Justification
The algorithm builds upon the strengths of HTRBO and ATRDDEBO to address their limitations and improve overall performance.
*   **Hybrid Surrogate Model:** The combination of Gaussian Process (GP) and Gradient Boosting (GB) models allows capturing both global trends and local details of the objective function. The GP provides uncertainty estimates, while GB offers computational efficiency.
*   **Adaptive Trust Region:** The trust region mechanism balances exploration and exploitation by limiting the search space around promising regions. The adaptive adjustment of the trust region size, based on the agreement between model predictions and actual function values, ensures efficient exploration.
*   **Dynamic Diversity Term:** The diversity term in the acquisition function encourages exploration in less-visited regions, preventing premature convergence to local optima. The dynamic adjustment of the diversity weight, based on the distribution of samples, ensures effective exploration.
*   **Dynamic Batch Size:** Adjusting the batch size based on the trust region size and model uncertainty allows for efficient sampling. Larger batch sizes are used when the trust region is large or the model is uncertain, while smaller batch sizes are used when the trust region is small or the model is confident.
*   **Adaptive Exploration Factor:** The exploration factor dynamically adjusts based on the remaining budget and model uncertainty, balancing exploration and exploitation throughout the optimization process.
*   **Computational Efficiency:** The use of Gradient Boosting and efficient optimization techniques ensures computational efficiency, allowing the algorithm to handle high-dimensional problems within a reasonable time.

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

class HATRDBO:
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

        # Diversity term
        if self.X is not None and len(self.X) > 5:
            distances = cdist(X, self.X)
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
            # Adaptive batch size
            _, sigma = self.gp_model.predict(self.X, return_std=True)
            avg_uncertainty = np.mean(sigma)
            batch_size = min(int(np.ceil(self.trust_region_size * avg_uncertainty)), 4)
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

            # Dynamic diversity weight adjustment
            if self.X is not None and len(self.X) > 5:
                distances = cdist(self.X, self.X)
                avg_distance = np.mean(distances)

                if avg_distance < self.trust_region_size / 2:
                    self.diversity_weight *= 1.1
                else:
                    self.diversity_weight *= 0.9

                self.diversity_weight = np.clip(self.diversity_weight, 0.01, 0.5)

            self.gp_model = self._fit_gp_model(self.X, self.y)
            self.gb_model = self._fit_gb_model(self.X, self.y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        return best_y, best_x
```
## Feedback
 The algorithm HATRDBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1781 with standard deviation 0.1049.

took 704.38 seconds to run.