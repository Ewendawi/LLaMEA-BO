# Description
**Adaptive Trust Region with Ensemble-based Volume-Aware Exploration (ATREnsembleVolumeBO):** This algorithm builds upon the strengths of HybridVolumeEI_DBS_BO and HybridTrustRegionBO_EIV by introducing an enhanced ensemble method for surrogate modeling and a refined volume-aware exploration strategy within an adaptive trust region framework. It utilizes a dynamic ensemble of Gaussian Process (GP) and Gradient Boosting (GB) models, where the weights are adjusted based on their individual performance and uncertainty. The volume awareness is improved by considering the local density of sampled points and dynamically adjusting the exploration bonus. Furthermore, a more sophisticated trust region adaptation strategy is implemented, considering not only the model agreement but also the gradient information from both GP and GB models.

# Justification
The key improvements are:

1.  **Enhanced Ensemble Modeling**: Instead of a simple weighted average, the ensemble uses a dynamic weighting scheme based on both the prediction error and the predictive variance of each model. This allows the algorithm to leverage the strengths of both GP (better uncertainty quantification) and GB (better handling of complex landscapes) more effectively.

2.  **Refined Volume Awareness**: The volume-aware exploration bonus is adjusted dynamically based on the local density of sampled points. This helps to avoid over-exploration in already well-sampled regions and focuses exploration on less explored areas. A kernel density estimator (KDE) is used to estimate the local density.

3.  **Gradient-Informed Trust Region Adaptation**: The trust region size is adapted based on the agreement between GP and GB predictions, as well as the magnitude of their gradients. Large gradients suggest that the models are confident in a specific direction, which can be used to expand the trust region in that direction.

4. **Efficient Batch Selection**: The batch selection is done using a combination of L-BFGS-B optimization and a clustering technique (k-means) to ensure diversity within the selected batch.

These improvements aim to address the limitations of the previous algorithms by providing a more robust and adaptive exploration-exploitation balance, leading to better performance on a wider range of black-box optimization problems.

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde


class ATREnsembleVolumeBO:
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
        self.gp_model = None
        self.gb_model = None
        self.kde = None
        self.scaler = MinMaxScaler()

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

        # Dynamic ensemble weighting
        gp_error = np.mean(np.abs(mu_gp - best_y)) if self.y is not None else 1.0
        gb_error = np.mean(np.abs(mu_gb - best_y)) if self.y is not None else 1.0

        gp_weight = np.exp(-gp_error) / (np.exp(-gp_error) + np.exp(-gb_error) + 1e-6)
        gb_weight = np.exp(-gb_error) / (np.exp(-gp_error) + np.exp(-gb_error) + 1e-6)

        # Weighted average of GP and GB predictions
        mu = gp_weight * mu_gp + gb_weight * mu_gb
        sigma = np.maximum(sigma, 1e-6) # Prevent division by zero

        imp = best_y - mu
        z = imp / sigma
        ei = imp * norm.cdf(z) + sigma * norm.pdf(z)

        # Volume-aware exploration
        if self.X is not None:
            # Estimate density using KDE
            density = self.kde.evaluate(X.T).reshape(-1, 1)
            # Scale density to [0, 1]
            density = self.scaler.transform(density)
            # Exploration bonus inversely proportional to density
            ei += self.exploration_factor * (1 - density)

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

        candidates = np.array(candidates)

        # Clustering to ensure diversity
        if len(candidates) > batch_size:
            kmeans = KMeans(n_clusters=batch_size, random_state=0, n_init='auto').fit(candidates)
            closest_points = []
            for i in range(batch_size):
                cluster_points = candidates[kmeans.labels_ == i]
                if len(cluster_points) > 0:
                    distances = cdist(kmeans.cluster_centers_[i].reshape(1, -1), cluster_points)
                    closest_index = np.argmin(distances)
                    closest_points.append(cluster_points[closest_index])
            candidates = np.array(closest_points)

        return candidates

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

        # Update KDE and scaler
        self.kde = gaussian_kde(self.X.T)
        self.scaler.fit(self.kde.evaluate(self.X.T).reshape(-1, 1))

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

            # Gradient-based trust region adaptation
            gp_grad = self._calculate_gradient(self.gp_model, X_next)
            gb_grad = self._calculate_gradient(self.gb_model, X_next)
            grad_agreement = np.mean(np.abs(gp_grad - gb_grad))

            if np.mean(agreement) < 1.0 and grad_agreement < 1.0:
                self.trust_region_size *= 1.1
            else:
                self.trust_region_size *= 0.9

            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0)

            # Dynamic exploration factor adjustment
            self.exploration_factor = 0.5 + (self.budget - self.n_evals) / self.budget

            self.gp_model = self._fit_gp_model(self.X, self.y)
            self.gb_model = self._fit_gb_model(self.X, self.y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        return best_y, best_x

    def _calculate_gradient(self, model, X):
        # Numerical gradient calculation
        delta = 1e-6
        gradient = np.zeros_like(X)
        for i in range(self.dim):
            X_plus = X.copy()
            X_minus = X.copy()
            X_plus[:, i] += delta
            X_minus[:, i] -= delta
            gradient[:, i] = (model.predict(X_plus) - model.predict(X_minus)) / (2 * delta)
        return gradient
```
## Feedback
 The algorithm ATREnsembleVolumeBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1828 with standard deviation 0.1073.

took 566.94 seconds to run.