from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel
from scipy.optimize import minimize

class VarianceAwareEnsembleTrustRegionBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * (dim + 1)
        self.ensemble_size = 3
        self.kernels = [
            C(1.0, constant_value_bounds=(1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
            C(1.0, constant_value_bounds=(1e-2, 1e2)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5),
            C(1.0, constant_value_bounds=(1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
        ]
        self.gps = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-5) for kernel in self.kernels]
        self.update_interval = 5
        self.last_gp_update = 0
        self.trust_region_center = np.zeros(dim)
        self.trust_region_radius = 2.5
        self.min_radius = 0.1
        self.radius_decay_base = 0.95
        self.radius_grow_base = 1.1
        self.ei_scaling = 0.1
        self.ei_variance_scaling = 0.05
        self.recentering_threshold = 0.5
        self.gp_weights = np.ones(self.ensemble_size) / self.ensemble_size  # Initialize GP weights equally
        self.gp_errors = np.zeros(self.ensemble_size) # Track GP errors

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        sample = sampler.random(n=n_points)
        scaled_sample = qmc.scale(sample, -1.0, 1.0)
        points = self.trust_region_center + scaled_sample * self.trust_region_radius
        points = np.clip(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        for i, gp in enumerate(self.gps):
            gp.fit(X, y)
            # Calculate error on observed data
            y_pred, _ = gp.predict(X, return_std=True)
            self.gp_errors[i] = np.mean((y_pred.flatten() - y.flatten())**2)

        # Update GP weights based on inverse error
        if np.sum(1/(self.gp_errors + 1e-6)) > 0: #avoid division by zero
            self.gp_weights = (1 / (self.gp_errors + 1e-6)) / np.sum(1 / (self.gp_errors + 1e-6))
        else:
            self.gp_weights = np.ones(self.ensemble_size) / self.ensemble_size
        return self.gps

    def _acquisition_function(self, X):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        ei_values = []
        for i, gp in enumerate(self.gps):
            mu, sigma = gp.predict(X, return_std=True)
            sigma = np.clip(sigma, 1e-9, np.inf)
            y_best = np.min(self.y)
            gamma = (y_best - mu) / sigma
            ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
            ei_values.append(ei.reshape(-1, 1))

        # Weighted average of EI values
        ei_ensemble = np.zeros_like(ei_values[0])
        for i in range(self.ensemble_size):
            ei_ensemble += self.gp_weights[i] * ei_values[i]

        return ei_ensemble

    def _select_next_points(self, batch_size):
        # Sample candidate points
        candidates = self._sample_points(100 * self.dim)
        
        # Calculate acquisition function values
        ei = self._acquisition_function(candidates)
        
        # Select top batch_size candidates based on EI
        selected_indices = np.argsort(ei.flatten())[-batch_size:]
        selected_points = candidates[selected_indices]
        
        return selected_points

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        while self.n_evals < self.budget:
            if self.n_evals - self.last_gp_update >= self.update_interval:
                self._fit_model(self.X, self.y)
                self.last_gp_update = self.n_evals

            batch_size = min(self.budget - self.n_evals, 5)
            next_X = self._select_next_points(batch_size)

            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            best_idx = np.argmin(self.y)
            current_best_y = self.y[best_idx][0]
            current_best_x = self.X[best_idx]
            
            # Calculate EI values and statistics
            ei_values = []
            for gp in self.gps:
                mu, sigma = gp.predict(next_X, return_std=True)
                sigma = np.clip(sigma, 1e-9, np.inf)
                y_best = np.min(self.y)
                gamma = (y_best - mu) / sigma
                ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
                ei_values.append(ei)
            
            ei_values = np.array(ei_values)
            avg_ei = np.mean(self._acquisition_function(next_X)) #Use the weighted average EI
            ei_variance = np.var(ei_values) #Variance across ensemble members
            
            if current_best_y < best_y:
                self.trust_region_radius *= (self.radius_decay_base + self.ei_scaling * avg_ei - self.ei_variance_scaling * ei_variance)
                self.trust_region_center = current_best_x
                best_y = current_best_y
                best_x = current_best_x
            else:
                self.trust_region_radius *= (self.radius_grow_base - self.ei_scaling * avg_ei + self.ei_variance_scaling * ei_variance)
                self.trust_region_radius = min(self.trust_region_radius, 2.5)

            self.trust_region_radius = max(self.trust_region_radius, self.min_radius)

            # Re-center trust region if current center is far from best point
            distance = np.linalg.norm(self.trust_region_center - best_x)
            if distance > self.recentering_threshold * self.trust_region_radius:
                self.trust_region_center = best_x

        return best_y, best_x
