from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize

class ETRGradBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.gradients: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim
        self.batch_size = min(5, self.dim)
        self.delta = 1e-3
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 2.0
        self.trust_region_min_radius = 0.1
        self.trust_region_max_radius = 5.0
        self.success_threshold = 0.2

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _estimate_gradient(self, func, x):
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.delta
            x_minus[i] -= self.delta

            x_plus[i] = np.clip(x_plus[i], self.bounds[0][i], self.bounds[1][i])
            x_minus[i] = np.clip(x_minus[i], self.bounds[0][i], self.bounds[1][i])

            gradient[i] = (func(x_plus) - func(x_minus)) / (2 * self.delta)
        return gradient

    def _fit_model(self, X, y):
        nn = NearestNeighbors(n_neighbors=min(len(X), 10)).fit(X)
        distances, _ = nn.kneighbors(X)
        median_distance = np.median(distances[:, 1])

        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(
            length_scale=median_distance, length_scale_bounds=(1e-3, 1e3)
        ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-7, 1e-3))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X, model, x_center, gradient_center):
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        y_best = np.min(self.y)
        imp = y_best - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Predicted gradient at X
        predicted_gradients, _ = model.predict(X, return_std=True) # Use model to predict gradients

        # Gradient-based term
        if gradient_center is not None:
            gradient_norm = np.linalg.norm(predicted_gradients, axis=1, keepdims=True)
            acquisition = ei + 0.01 * gradient_norm
        else:
            acquisition = ei

        # Trust region constraint
        distances = np.linalg.norm(X - x_center, axis=1, keepdims=True)
        acquisition[distances > self.trust_region_radius] = -1e9  # Penalize points outside the trust region

        return acquisition

    def _select_next_points(self, func, batch_size, x_center, gradient_center):
        candidate_points = self._sample_points(10 * batch_size)
        model = self._fit_model(self.X, self.y)
        acquisition_values = self._acquisition_function(candidate_points, model, x_center, gradient_center)
        indices = np.argsort(acquisition_values.flatten())[::-1][:batch_size]
        return candidate_points[indices]

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y, new_gradients=None):
        if self.X is None:
            self.X = new_X
            self.y = new_y
            if new_gradients is not None:
                self.gradients = new_gradients
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))
            if new_gradients is not None:
                if self.gradients is None:
                    self.gradients = new_gradients
                else:
                    self.gradients = np.vstack((self.gradients, new_gradients))

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        initial_gradients = np.array([self._estimate_gradient(func, x) for x in initial_X])
        self._update_eval_points(initial_X, initial_y, initial_gradients)

        best_index = np.argmin(self.y)
        x_center = self.X[best_index].copy()
        gradient_center = self.gradients[best_index].copy()

        while self.n_evals < self.budget:
            # Optimization within trust region
            next_X = self._select_next_points(func, self.batch_size, x_center, gradient_center)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Update best point
            best_index = np.argmin(self.y)
            new_best_x = self.X[best_index]
            new_best_y = self.y[best_index][0]

            # Success ratio
            actual_improvement = np.min(self.y) - new_best_y
            model = self._fit_model(self.X, self.y)
            predicted_improvement = model.predict(x_center.reshape(1, -1))[0] - model.predict(new_best_x.reshape(1, -1))[0]

            if predicted_improvement != 0:
                success_ratio = actual_improvement / predicted_improvement
            else:
                success_ratio = 0  # Avoid division by zero

            # Adjust trust region radius
            if success_ratio > self.success_threshold:
                self.trust_region_radius = min(
                    self.trust_region_radius * self.trust_region_expand_factor, self.trust_region_max_radius
                )
                x_center = new_best_x.copy()
                gradient_center = self.gradients[best_index].copy()
            else:
                self.trust_region_radius = max(
                    self.trust_region_radius * self.trust_region_shrink_factor, self.trust_region_min_radius
                )

        best_index = np.argmin(self.y)
        best_y = self.y[best_index][0]
        best_x = self.X[best_index]
        return best_y, best_x
