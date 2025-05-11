from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import pairwise_distances


class DICEBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.diversity_weight = 0.1
        self.exploration_weight = 0.1
        self.best_x = None
        self.best_y = float('inf')
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

    def _sample_points(self, n_points, method='sobol'):
        if method == 'sobol':
            sampler = qmc.Sobol(d=self.dim, seed=42)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        elif method == 'random':
            return np.random.uniform(self.bounds[0], self.bounds[1], size=(n_points, self.dim))
        else:
            raise ValueError("Invalid sampling method.")

    def _fit_model(self, X, y):
        model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        self.kernel = model.kernel_  # Update kernel with optimized parameters
        return model

    def _acquisition_function(self, X, model):
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if self.best_y is None:
            ei = np.zeros_like(mu)
        else:
            imp = self.best_y - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0  # handle zero sigma

        # Add diversity term
        if self.X is not None:
            distances = pairwise_distances(X, self.X)
            min_distances = np.min(distances, axis=1).reshape(-1, 1)
            ei = ei + self.diversity_weight * min_distances

        # Add covariance-guided exploration term
        if self.X is not None:
            _, cov = model.predict(X, return_cov=True)
            # Use the diagonal elements of the covariance matrix as an exploration term
            exploration_term = np.diag(cov).reshape(-1, 1)
            ei = ei + self.exploration_weight * exploration_term

        return ei

    def _select_next_points(self, batch_size, model):
        # Sample candidate points using a combination of random and covariance-guided sampling
        n_candidates_random = batch_size // 2
        n_candidates_covariance = batch_size - n_candidates_random

        candidate_points_random = self._sample_points(n_candidates_random, method='random')

        # Generate more random samples and weight them by the acquisition function
        candidate_points = self._sample_points(10 * n_candidates_covariance, method='random')  # Generate more samples
        acquisition_values = self._acquisition_function(candidate_points, model)
        probabilities = acquisition_values.flatten() / np.sum(acquisition_values)  # Normalize to probabilities

        # Select points based on probabilities
        indices = np.random.choice(len(candidate_points), size=n_candidates_covariance, replace=False, p=probabilities)
        candidate_points_covariance = candidate_points[indices]

        candidate_points = np.vstack((candidate_points_random, candidate_points_covariance))

        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_points, model)

        # Select top batch_size points based on acquisition function values
        indices = np.argsort(-acquisition_values.flatten())[:batch_size]
        selected_points = candidate_points[indices]

        return selected_points

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

        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial sampling
        initial_X = self._sample_points(self.n_init, method='sobol')
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        batch_size = 5
        while self.n_evals < self.budget:
            # Optimization
            model = self._fit_model(self.X, self.y)

            # select points by acquisition function
            next_X = self._select_next_points(batch_size, model)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
