from typing import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


class TrustRegionAdaptiveBOv1:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.gp = None
        self.n_init = min(20 * self.dim, self.budget // 2)
        self.batch_size = 1
        self.min_batch_size = 1
        self.max_batch_size = 10
        self.exploration_factor = 2.0
        self.best_y = np.inf
        self.best_x = None
        self.n_evals = 0
        self.uncertainty_threshold = 0.01
        self.improvement_threshold = 1e-4
        self.decay_factor_batch = 0.95
        self.decay_factor_exploration = 0.98
        self.ei_exploration_factor = 1.0
        self.uncertainty_weight = 0.5
        self.local_search_radius = 0.5
        self.local_search_points = 50
        self.local_search_prob = 0.8
        self.length_scale = 1.0
        self.length_scale_decay = 0.99
        self.length_scale_min = 1e-2
        self.length_scale_max = 10.0
        self.trust_region_radius = 1.0
        self.trust_region_decay = 0.95
        self.trust_region_min = 0.1
        self.trust_region_expansion = 1.1
        self.trust_region_success = 0.2
        self.trust_region_step_size = 0.2
        self.trust_region_local_prob = 0.5

    def _sample_points(self, n_points) -> np.ndarray:
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        points = sampler.random(n_points)
        return qmc.scale(points, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds="fixed") * Matern(
            length_scale=self.length_scale, length_scale_bounds=(self.length_scale_min, self.length_scale_max), nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                          alpha=1e-6, normalize_y=False)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X) -> np.ndarray:
        if self.gp is None or self.X is None or len(self.X) < 2:
            return np.ones((X.shape[0], 1)) * self.exploration_factor
        
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        
        best_y = np.min(self.y) if self.y is not None and len(self.y) > 0 else 0
        imp = best_y - mu
        Z = imp / (sigma + 1e-9)
        ei = imp * (0.5 * (1 + np.sign(Z)) * Z  + 0.5 * (1 - np.sign(Z)) * 0) + sigma * np.exp(-0.5*Z**2) / np.sqrt(2*np.pi)
        
        uncertainty_term = sigma
        
        return (self.uncertainty_weight * uncertainty_term + (1 - self.uncertainty_weight) * ei) * self.exploration_factor * self.ei_exploration_factor
    
    def _adjust_exploration_factor(self, improvement):
        if improvement > self.improvement_threshold:
            self.ei_exploration_factor = min(2.0, self.ei_exploration_factor * 1.1)
        else:
            self.ei_exploration_factor = max(0.5, self.ei_exploration_factor * 0.9)
    
    def _local_search_trust_region(self):
        if self.best_x is None:
             return self._sample_points(self.batch_size)
        
        local_points = np.random.normal(loc=self.best_x, scale=self.trust_region_radius, size=(self.local_search_points, self.dim))
        local_points = np.clip(local_points, self.bounds[0], self.bounds[1])
        
        acquisition_values = self._acquisition_function(local_points)
        indices = np.argsort(-acquisition_values.flatten())[:self.batch_size]
        return local_points[indices]

    def _local_search(self):
        if self.best_x is None:
            return self._sample_points(self.batch_size)
        
        local_points = np.random.normal(loc=self.best_x, scale=self.local_search_radius, size=(self.local_search_points, self.dim))
        local_points = np.clip(local_points, self.bounds[0], self.bounds[1])
        
        acquisition_values = self._acquisition_function(local_points)
        indices = np.argsort(-acquisition_values.flatten())[:self.batch_size]
        return local_points[indices]
    
    def _select_next_points(self, batch_size) -> np.ndarray:
        if self.gp is None or self.X is None or len(self.X) < 2:
             return self._sample_points(batch_size)
        
        if np.random.rand() < self.trust_region_local_prob:
             return self._local_search_trust_region()
        elif np.random.rand() < self.local_search_prob:
            return self._local_search()
        else:
            candidates = self._sample_points(1000)
            acquisition_values = self._acquisition_function(candidates)
            indices = np.argsort(-acquisition_values.flatten())[:batch_size]
            return candidates[indices]

    def _update_eval_points(self, X, y):
        if self.X is None:
            self.X = X
            self.y = y
        else:
            self.X = np.vstack((self.X, X))
            self.y = np.vstack((self.y, y))

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        
        initial_X = self._sample_points(self.n_init)
        initial_y = np.array([func(x) for x in initial_X]).reshape(-1, 1)
        self.n_evals += self.n_init
        
        self._update_eval_points(initial_X, initial_y)
        
        self.best_y = np.min(self.y)
        self.best_x = self.X[np.argmin(self.y)]
        
        rest_of_budget = self.budget - self.n_evals
        
        previous_best_y = self.best_y
        
        while rest_of_budget > 0:
            self._fit_model(self.X, self.y)
            
            if self.gp is not None:
                _, uncertainty = self.gp.predict(self.X, return_std=True)
                avg_uncertainty = np.mean(uncertainty)
                
                improvement = previous_best_y - self.best_y
                
                if avg_uncertainty > self.uncertainty_threshold or improvement > self.improvement_threshold:
                    self.batch_size = min(self.max_batch_size, int(self.batch_size * (1 + self.decay_factor_batch)))
                else:
                    self.batch_size = max(self.min_batch_size, int(self.batch_size * self.decay_factor_batch))
                
                self.batch_size = min(rest_of_budget, self.batch_size)
                self._adjust_exploration_factor(improvement)
                self.local_search_radius = max(0.05, self.local_search_radius * 0.95)
                self.length_scale = max(self.length_scale_min, self.length_scale * self.length_scale_decay)
                
                if improvement > self.trust_region_success:
                      self.trust_region_radius = min(1.0, self.trust_region_radius * self.trust_region_expansion)
                else:
                    self.trust_region_radius = max(self.trust_region_min, self.trust_region_radius * self.trust_region_decay)

            else:
                 self.batch_size = min(rest_of_budget, 1)
            
            next_points = self._select_next_points(self.batch_size)
            next_y = np.array([func(x) for x in next_points]).reshape(-1, 1)
            
            self.n_evals += self.batch_size
            
            self._update_eval_points(next_points, next_y)
            
            previous_best_y = self.best_y
            current_best_y = np.min(self.y)
            current_best_x = self.X[np.argmin(self.y)]
            
            if current_best_y < self.best_y:
                self.best_y = current_best_y
                self.best_x = current_best_x
            
            self.exploration_factor *= self.decay_factor_exploration
            rest_of_budget = self.budget - self.n_evals
            
        return self.best_y, self.best_x

