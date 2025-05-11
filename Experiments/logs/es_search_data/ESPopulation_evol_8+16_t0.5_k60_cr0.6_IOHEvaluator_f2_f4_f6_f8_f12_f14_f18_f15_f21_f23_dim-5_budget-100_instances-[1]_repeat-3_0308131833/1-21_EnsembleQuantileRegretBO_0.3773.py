from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.spatial.distance import cdist

class EnsembleQuantileRegretBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.n_models = 3
        self.gps = []
        self.weights = np.ones(self.n_models) / self.n_models
        self.weight_decay = 0.95
        self.novelty_weight = 0.1
        self.best_x = None
        self.best_y = np.inf
        self.quantile_level = 0.9
        self.quantile_decay = 0.95
        self.min_quantile_level = 0.5

        kernels = [
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed"),
            ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds="fixed", nu=1.5),
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=0.5, length_scale_bounds="fixed")
        ]
        for kernel in kernels:
            self.gps.append(GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6))

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        for gp in self.gps:
            gp.fit(X, y)

    def _acquisition_function(self, X):
        if self.X is None:
            return np.zeros((len(X), 1))

        mu = np.zeros((len(X), 1))
        sigma = np.zeros((len(X), 1))

        for i, gp in enumerate(self.gps):
            m, s = gp.predict(X, return_std=True)
            mu += self.weights[i] * m.reshape(-1, 1)
            sigma += self.weights[i] * s.reshape(-1, 1)

        # CVaR of the regret
        regret = mu - self.best_y
        alpha = self.quantile_level

        # CVaR approximation (using Gaussian quantiles)
        VaR = regret + sigma * norm.ppf(alpha)
        CVaR = regret - (sigma * norm.pdf(norm.ppf(alpha)) / (1 - alpha))

        # If alpha is close to 1, the above calculation can be unstable.
        # In this case, we can approximate CVaR with VaR.
        if alpha > 0.99:
            CVaR = VaR

        acquisition = CVaR

        # Novelty search component
        if len(self.X) > 0:
            distances = cdist(X, self.X)
            min_distances = np.min(distances, axis=1)
            acquisition += self.novelty_weight * min_distances.reshape(-1, 1)

        return acquisition.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        x_tries = self._sample_points(batch_size * 10)
        acq_values = self._acquisition_function(x_tries)
        indices = np.argsort(acq_values.flatten())[:batch_size]  # Minimize CVaR
        return x_tries[indices]

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

        best_index = np.argmin(self.y)
        if self.y[best_index, 0] < self.best_y:
            self.best_x = self.X[best_index]
            self.best_y = self.y[best_index, 0]

        if len(self.X) > self.n_init:
            for i, gp in enumerate(self.gps):
                y_pred, sigma = gp.predict(self.X, return_std=True)
                error = np.mean((y_pred.reshape(-1, 1) - self.y) ** 2)
                self.weights[i] = np.exp(-error)

            self.weights /= np.sum(self.weights)
            self.weights *= self.weight_decay
            self.weights += (1 - self.weight_decay) / self.n_models
            self.weights /= np.sum(self.weights)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        while self.n_evals < self.budget:
            self._fit_model(self.X, self.y)
            batch_size = min(10, self.budget - self.n_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            self.quantile_level *= self.quantile_decay
            self.quantile_level = max(self.quantile_level, self.min_quantile_level)

        return self.best_y, self.best_x
