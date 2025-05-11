from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.impute import SimpleImputer
from scipy.linalg import solve


class GEATRBO:
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
        self.epsilon = 1e-6
        self.gradient_weight = 0.1
        self.use_gp = True  # Start with GP
        self.imputer = SimpleImputer(strategy='mean')

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
        if np.isnan(X).any() or np.isnan(y).any():
            X = self.imputer.fit_transform(X)
            y = self.imputer.fit_transform(y)
        model = HistGradientBoostingRegressor(random_state=0)
        model.fit(X, y.ravel())
        return model

    def _acquisition_function(self, X):
        if self.use_gp:
            mu, sigma = self.model.predict(X, return_std=True)
            mu = mu.reshape(-1, 1)
            sigma = sigma.reshape(-1, 1)

            lcb = mu - self.exploration_factor * sigma

            if self.X is not None:
                distances = cdist(X, self.X)
                min_distances = np.min(distances, axis=1).reshape(-1, 1)
                lcb -= 0.01 * self.exploration_factor * min_distances

            # Gradient enhancement
            if self.X is not None and len(self.X) > 5:
                dmu_dx = self._predict_gradient(X)
                if dmu_dx is not None:
                    best_idx = np.argmin(self.y)
                    best_x = self.X[best_idx]
                    direction = best_x - X
                    direction = direction / (np.linalg.norm(direction, axis=1, keepdims=True) + self.epsilon)
                    gradient_alignment = np.sum(dmu_dx * direction, axis=1, keepdims=True)
                    lcb -= self.gradient_weight * gradient_alignment

            return lcb
        else:
            mu = self.model.predict(X).reshape(-1, 1)
            sigma = np.ones_like(mu)  # Dummy sigma for GB
            lcb = mu - self.exploration_factor * sigma
            return lcb

    def _predict_gradient(self, X):
        if not hasattr(self.model, 'kernel_'):
            return None

        X = np.atleast_2d(X)
        n_points = X.shape[0]

        K = self.model.kernel_(X, self.X)  # (n_points, n_samples)
        alpha = solve(self.model.kernel_(self.X, self.X) + np.eye(self.X.shape[0]) * self.model.alpha, self.y - self.model.y_train_mean_, assume_a='pos')  # (n_samples, 1)

        dK_dx = np.zeros((n_points, self.X.shape[0], self.dim)) # (n_points, n_samples, dim)

        for i in range(n_points):
            for j in range(self.X.shape[0]):
                for k in range(self.dim):
                    X_diff = X[i, k] - self.X[j, k]
                    dK_dx[i, j, k] = - self.model.kernel_.k2.length_scale * K[i, j] * X_diff

        dmu_dx = np.zeros((n_points, self.dim))
        for i in range(n_points):
            for k in range(self.dim):
                dmu_dx[i, k] = np.sum(dK_dx[i, :, k] * alpha.ravel())

        return dmu_dx

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

        self.model = self._fit_gp_model(self.X, self.y)

        while self.n_evals < self.budget:
            batch_size = min(int(np.ceil(self.trust_region_size)), 4)
            batch_size = max(1, batch_size)

            X_next = self._select_next_points(batch_size)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

            # Adaptive trust region adjustment
            if self.use_gp:
                y_pred, sigma = self.model.predict(X_next, return_std=True)
                y_pred = y_pred.reshape(-1, 1)
                sigma = sigma.reshape(-1, 1)
                agreement = np.abs(y_pred - y_next) / (sigma.reshape(-1, 1) + self.epsilon)
            else:
                y_pred = self.model.predict(X_next).reshape(-1, 1)
                agreement = np.abs(y_pred - y_next)

            if np.mean(agreement) < 1.0:
                self.trust_region_size *= 1.1
            else:
                self.trust_region_size *= 0.9

            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0)

            # Dynamic exploration factor adjustment
            self.exploration_factor = 1.0 + (self.budget - self.n_evals) / self.budget
            self.exploration_factor = max(0.1, self.exploration_factor)

            # Adaptive model selection
            if self.trust_region_size < 1.0 and self.use_gp:
                self.use_gp = False
                self.model = self._fit_gb_model(self.X, self.y)
            elif self.trust_region_size >= 1.0 and not self.use_gp:
                self.use_gp = True
                self.model = self._fit_gp_model(self.X, self.y)
            elif self.use_gp:
                self.model = self._fit_gp_model(self.X, self.y)
            else:
                self.model = self._fit_gb_model(self.X, self.y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        return best_y, best_x
