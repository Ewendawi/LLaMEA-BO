from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.spatial.distance import cdist

class AdaptiveHybridBO_DE:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * (dim + 1)
        self.trust_region_center = np.zeros(dim)
        self.trust_region_radius = 2.5
        self.min_radius = 0.1
        self.radius_decay = 0.95
        self.radius_grow = 1.1
        self.exploration_weight = 0.1  # Initial exploration weight
        self.exploration_decay = 0.98 # Decay exploration as optimization progresses

    def _sample_points(self, n_points, use_trust_region=True):
        if use_trust_region:
            sampler = qmc.Sobol(d=self.dim, scramble=False)
            sample = sampler.random(n=n_points)
            scaled_sample = qmc.scale(sample, -1.0, 1.0)
            points = self.trust_region_center + scaled_sample * self.trust_region_radius
            points = np.clip(points, self.bounds[0], self.bounds[1])
        else:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            points = qmc.scale(sample, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        kernel = C(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-5)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp, y_best):
        mu, sigma = gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)
        gamma = (y_best - mu) / sigma
        ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return ei

    def _select_next_points(self, gp, y_best, batch_size):
        # Dynamic exploration-exploitation balance
        num_trust_region = int(batch_size * (1 - self.exploration_weight))
        num_random = batch_size - num_trust_region

        # Sample points from trust region
        if num_trust_region > 0:
            candidates_tr = self._sample_points(100 * self.dim, use_trust_region=True)
            ei_tr = self._acquisition_function(candidates_tr, gp, y_best)
            selected_indices_tr = np.argsort(ei_tr)[-num_trust_region:]
            selected_points_tr = candidates_tr[selected_indices_tr]
        else:
            selected_points_tr = np.empty((0, self.dim))

        # Sample points randomly
        if num_random > 0:
            candidates_rand = self._sample_points(100 * self.dim, use_trust_region=False)
            ei_rand = self._acquisition_function(candidates_rand, gp, y_best)
            selected_indices_rand = np.argsort(ei_rand)[-num_random:]
            selected_points_rand = candidates_rand[selected_indices_rand]
        else:
            selected_points_rand = np.empty((0, self.dim))

        selected_points = np.concatenate([selected_points_tr, selected_points_rand], axis=0)

        if self.X is not None:
            distances = cdist(selected_points, self.X)
            min_distances = np.min(distances, axis=1)
            selected_points = selected_points[min_distances > 0.1] # Enforce diversity

            if len(selected_points) < batch_size:
                remaining_needed = batch_size - len(selected_points)
                candidates = self._sample_points(100 * self.dim)
                ei = self._acquisition_function(candidates, gp, y_best)
                additional_indices = np.argsort(ei)[:-batch_size-1:-1][:remaining_needed]
                additional_points = candidates[additional_indices]
                selected_points = np.concatenate([selected_points, additional_points], axis=0)

        return selected_points[:batch_size]

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
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init, use_trust_region=False)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        self.trust_region_center = best_x # Initialize trust region center with best initial point

        while self.n_evals < self.budget:
            gp = self._fit_model(self.X, self.y)
            batch_size = min(self.budget - self.n_evals, 5)
            next_X = self._select_next_points(gp, best_y, batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            best_idx = np.argmin(self.y)
            current_best_y = self.y[best_idx][0]
            current_best_x = self.X[best_idx]

            if current_best_y < best_y:
                self.trust_region_center = current_best_x
                self.trust_region_radius *= self.radius_decay
                best_y = current_best_y
                best_x = current_best_x
            else:
                if self.trust_region_radius < 1.0:
                    self.trust_region_radius *= self.radius_grow
                    self.trust_region_radius = min(self.trust_region_radius, 2.5)
            
            self.trust_region_radius = max(self.trust_region_radius, self.min_radius)
            self.exploration_weight *= self.exploration_decay # Reduce exploration over time

        return best_y, best_x
