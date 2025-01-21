from typing import Callable
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize
from scipy.stats import norm

class AdaptiveBatchBOv1:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X_samples = []
        self.y_samples = []
        self.best_x = None
        self.best_y = float('inf')
        self.gp = None
        self.initial_points_multiplier = 2
        self.local_search_radius = 0.5
        self.local_search_iterations = 5

    def _sample_points(self, n_points) -> np.ndarray:
        points = np.random.uniform(self.bounds[0], self.bounds[1], size=(n_points, self.dim))
        return points

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X) -> np.ndarray:
        if self.gp is None:
             return np.random.rand(X.shape[0], 1) 
        mu, sigma = self.gp.predict(X, return_std=True)
        
        if self.best_y == float('inf'):
            return np.random.rand(X.shape[0], 1)
        
        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size) -> np.ndarray:
        
        if self.gp is None:
             return self._sample_points(batch_size)
        
        x_tries = self._sample_points(1000)
        acq_values = self._acquisition_function(x_tries)
        indices = np.argsort(acq_values.flatten())[-batch_size:]
        selected_points = x_tries[indices]

        return selected_points
    
    def _local_search(self, func, x_start):
        best_local_x = x_start
        best_local_y = func(x_start.reshape(1, -1))
        for _ in range(self.local_search_iterations):
            x_rand = np.random.uniform(-self.local_search_radius, self.local_search_radius, size=self.dim)
            x_new = best_local_x + x_rand
            x_new = np.clip(x_new, self.bounds[0], self.bounds[1])
            y_new = func(x_new.reshape(1, -1))
            if y_new < best_local_y:
                best_local_y = y_new
                best_local_x = x_new
        return best_local_x, best_local_y
    
    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        n_initial_points = self.dim * self.initial_points_multiplier
        
        initial_X = self._sample_points(n_initial_points)
        initial_y = np.array([func(x.reshape(1, -1)) for x in initial_X]).reshape(-1, 1)

        self.X_samples.extend(initial_X)
        self.y_samples.extend(initial_y)
        
        self.best_x = initial_X[np.argmin(initial_y)]
        self.best_y = np.min(initial_y)
        
        rest_of_budget = self.budget - n_initial_points
        
        while rest_of_budget > 0:
            batch_size = max(1, int(rest_of_budget * 0.1))
            
            self._fit_model(np.array(self.X_samples), np.array(self.y_samples).flatten())
            
            next_points = self._select_next_points(batch_size)
            
            for x_next in next_points:
                x_local_search, y_local_search = self._local_search(func, x_next)
                
                self.X_samples.append(x_local_search)
                self.y_samples.append(y_local_search)
                
                if y_local_search < self.best_y:
                    self.best_y = y_local_search
                    self.best_x = x_local_search
                
            rest_of_budget -= batch_size
            
        return self.best_y, self.best_x