from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

class ATRBODEG:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(10 * dim, self.budget // 5)
        self.trust_region_width = 2.0
        self.success_threshold = 0.1
        self.diversity_weight = 0.1
        self.best_y = np.inf
        self.best_x = None
        self.gradient_estimation_threshold = 0.5 # Trust region width threshold for gradient estimation
        self.finite_difference_step = 0.1
        self.ensemble_weights = [0.5, 0.5] # Initial weights for RBF and Matern kernels

        # Initialize GPR models with different kernels
        self.kernel_rbf = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        self.kernel_matern = ConstantKernel(1.0) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        self.model_rbf = GaussianProcessRegressor(kernel=self.kernel_rbf, n_restarts_optimizer=5)
        self.model_matern = GaussianProcessRegressor(kernel=self.kernel_matern, n_restarts_optimizer=5)

    def _sample_points(self, n_points, center=None, width=None):
        if center is None:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            lower_bound = np.maximum(self.bounds[0], center - width / 2)
            upper_bound = np.minimum(self.bounds[1], center + width / 2)
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            scaled_sample = qmc.scale(sample, lower_bound, upper_bound)
            return scaled_sample

    def _estimate_gradient(self, func, x):
        # Estimate gradient using finite differences
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.finite_difference_step
            x_minus[i] -= self.finite_difference_step
            x_plus = np.clip(x_plus, self.bounds[0][i], self.bounds[1][i])
            x_minus = np.clip(x_minus, self.bounds[0][i], self.bounds[1][i])

            y_plus = self._evaluate_points(func, x_plus.reshape(1, -1))[0, 0]
            y_minus = self._evaluate_points(func, x_minus.reshape(1, -1))[0, 0]

            gradient[i] = (y_plus - y_minus) / (2 * self.finite_difference_step)
        return gradient

    def _fit_model(self, X, y):
        # Fit both GPR models
        self.model_rbf.fit(X, y)
        self.model_matern.fit(X, y)

        # Update ensemble weights based on validation performance
        if len(X) > 5 * self.dim:  # Ensure enough data for validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            rbf_score = -np.mean((self.model_rbf.predict(X_val) - y_val)**2)
            matern_score = -np.mean((self.model_matern.predict(X_val) - y_val)**2)

            # Softmax to get weights
            scores = np.array([rbf_score, matern_score])
            softmax_scores = np.exp(scores - np.max(scores))
            probabilities = softmax_scores / np.sum(softmax_scores)
            self.ensemble_weights = probabilities

    def _acquisition_function(self, X, func):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        # Predict with both models
        mu_rbf, sigma_rbf = self.model_rbf.predict(X, return_std=True)
        mu_matern, sigma_matern = self.model_matern.predict(X, return_std=True)
        mu_rbf = mu_rbf.reshape(-1, 1)
        sigma_rbf = sigma_rbf.reshape(-1, 1)
        mu_matern = mu_matern.reshape(-1, 1)
        sigma_matern = sigma_matern.reshape(-1, 1)

        # Weighted average of predictions
        mu = self.ensemble_weights[0] * mu_rbf + self.ensemble_weights[1] * mu_matern
        sigma = self.ensemble_weights[0] * sigma_rbf + self.ensemble_weights[1] * sigma_matern

        # Expected Improvement
        best = np.min(self.y)
        imp = best - mu
        Z = imp / (sigma + 1e-9)
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei = np.clip(ei, 0, 1e10)

        # Diversity term
        if self.X is not None:
            distances = pairwise_distances(X, self.X)
            min_distances = np.min(distances, axis=1).reshape(-1, 1)
            diversity = min_distances
        else:
            diversity = np.zeros_like(ei)

        # Combined acquisition function
        acq = ei + self.diversity_weight * diversity
        return acq

    def _select_next_points(self, batch_size, func):
        n_candidates = max(2000, batch_size * 200)
        X_cand = self._sample_points(n_candidates, center=self.best_x, width=self.trust_region_width)

        acq_values = self._acquisition_function(X_cand, func)

        top_indices = np.argsort(acq_values.flatten())[::-1][:batch_size]
        return X_cand[top_indices]

    def _evaluate_points(self, func, x):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        if len(x.shape) == 1:
            y = func(x)
            self.n_evals += 1
            return y
        else:
            y = np.array([func(xi) for xi in x])
            self.n_evals += len(x)
            return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.concatenate((self.X, new_X), axis=0)
            self.y = np.concatenate((self.y, new_y), axis=0)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_x = X_init[np.argmin(y_init)][0]
        self.best_y = np.min(y_init)

        # Optimization loop
        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            # Fit the model
            self._fit_model(self.X, self.y)

            # Gradient estimation if trust region is small
            if self.trust_region_width < self.gradient_estimation_threshold:
                gradient = self._estimate_gradient(func, self.best_x)
                # Use gradient information (e.g., in acquisition function or model fitting) - not implemented here for brevity
                pass

            # Select next points
            X_next = self._select_next_points(batch_size, func)

            # Evaluate points
            y_next = self._evaluate_points(func, X_next)

            # Update evaluated points
            self._update_eval_points(X_next, y_next)

            # Update best solution
            new_best_y = np.min(self.y)
            new_best_x = self.X[np.argmin(self.y)]

            # Adjust trust region width
            if (self.best_y - new_best_y) / self.best_y > self.success_threshold:
                self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)
            else:
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)

            self.best_y = new_best_y
            self.best_x = new_best_x

        # Return best solution
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
