from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from scipy.stats import norm
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


class AdaptiveTrustRegionBO_HSEIV:
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
        self.stagnation_counter = 0
        self.max_stagnation = 5
        self.pca = PCA()

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        samples = sampler.random(n=n_points)
        return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_gp_model(self, X, y):
        # Optimize kernel length scale based on trust region data
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-3))
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
        distances, _ = self.knn.kneighbors(X)
        avg_distances = np.mean(distances, axis=1).reshape(-1, 1)
        ei += self.diversity_weight * self.exploration_factor * avg_distances

        return ei

    def _select_next_points(self, batch_size):
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        # Adapt trust region shape
        X_trust = self.X[cdist(self.X, best_x[None, :])[:, 0] < self.trust_region_size]
        if len(X_trust) > self.dim:
            self.pca.fit(X_trust)
            eigenvectors = self.pca.components_
            eigenvalues = self.pca.explained_variance_
        else:
            eigenvectors = np.eye(self.dim)
            eigenvalues = np.ones(self.dim)

        candidates = []
        values = []
        for _ in range(batch_size):
            # Generate initial points within the trust region (elliptical shape)
            x_start = best_x + np.random.normal(0, 0.1, size=self.dim)
            x_start = np.clip(x_start, self.bounds[0], self.bounds[1])

            # Transform to PCA space
            x_start_transformed = np.dot(x_start - best_x, eigenvectors.T)

            # Clip to elliptical trust region
            x_start_transformed = np.clip(x_start_transformed, -np.sqrt(eigenvalues) * self.trust_region_size / 2, np.sqrt(eigenvalues) * self.trust_region_size / 2)

            # Transform back to original space
            x_start = best_x + np.dot(x_start_transformed, eigenvectors)
            x_start = np.clip(x_start, self.bounds[0], self.bounds[1])

            lower_bound = np.maximum(best_x - self.trust_region_size / 2, self.bounds[0])
            upper_bound = np.minimum(best_x + self.trust_region_size / 2, self.bounds[1])

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

        best_y = np.min(self.y)
        best_y_history = [best_y]

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
            ei_values = self._expected_improvement(X_next, np.min(self.y))

            agreement_factor = np.mean(agreement)
            ei_factor = np.mean(ei_values)

            if agreement_factor < 1.0 and ei_factor > 0.01:
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

            # Adaptive diversity weight adjustment
            self.diversity_weight = 0.001 + 0.099 * np.exp(-self.trust_region_size)

            self.gp_model = self._fit_gp_model(self.X, self.y)
            self.gb_model = self._fit_gb_model(self.X, self.y)

            current_best_y = np.min(self.y)
            best_y_history.append(current_best_y)

            if current_best_y >= best_y:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                best_y = current_best_y

            if self.stagnation_counter >= self.max_stagnation:
                # Restart models with fresh initial points
                X_init = self._sample_points(self.n_init)
                y_init = self._evaluate_points(func, X_init)
                self._update_eval_points(X_init, y_init)

                self.gp_model = self._fit_gp_model(self.X, self.y)
                self.gb_model = self._fit_gb_model(self.X, self.y)
                self.stagnation_counter = 0
                best_y = np.min(self.y)
                best_y_history = [best_y]

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        return best_y, best_x
