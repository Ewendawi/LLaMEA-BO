from typing import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.optimize import Bounds
from sklearn.ensemble import RandomForestRegressor

class EnsembleLocalSearchBOv1:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X = None
        self.y = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.models = []
        self.n_models = 3
        self.best_y = float('inf')
        self.best_x = None
        self.local_search_radius = 0.5
        self.n_evals = 0
        self.n_init = 0

    def _sample_points(self, n_points: int) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])
    
    def _get_model(self):
        return self.models

    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        self.scaler_X.fit(X)
        X_scaled = self.scaler_X.transform(X)
        self.scaler_y.fit(y)
        y_scaled = self.scaler_y.transform(y)
        self.models = []
        for _ in range(self.n_models):
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-2, 1e2))
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)
            model.fit(X_scaled, y_scaled.ravel())
            self.models.append(model)

    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            return np.zeros((X.shape[0], 1))
        X_scaled = self.scaler_X.transform(X)
        
        mu_ensemble = []
        sigma_ensemble = []
        for model in self.models:
            mu, sigma = model.predict(X_scaled, return_std=True)
            mu = self.scaler_y.inverse_transform(mu.reshape(-1,1)).flatten()
            sigma = sigma * np.sqrt(self.scaler_y.var_)
            mu_ensemble.append(mu)
            sigma_ensemble.append(sigma)
        mu_ensemble = np.array(mu_ensemble)
        sigma_ensemble = np.array(sigma_ensemble)
        
        mu = np.mean(mu_ensemble, axis=0)
        sigma = np.mean(sigma_ensemble, axis=0)
        
        best_y = np.min(self.y)
        imp = best_y - mu
        Z = imp / sigma
        pi = 0.5 + 0.5 * np.sign(Z)
        pi[sigma <= 1e-6] = 0.0
        
        return pi.reshape(-1,1)

    def _local_search(self, func: Callable[[np.ndarray], np.float64], x_start: np.ndarray) -> tuple[np.ndarray, float]:
        bounds = Bounds(self.bounds[0], self.bounds[1])
        
        def obj_func(x):
           if self.n_evals >= self.budget:
             raise Exception("Overbudget")
           self.n_evals += 1
           return func(x)

        res = minimize(obj_func, x_start, method='L-BFGS-B', bounds = bounds, options={'maxiter': 10})
        return res.x, res.fun

    def _select_next_points(self, batch_size: int) -> np.ndarray:
        if batch_size <= 0:
            return np.array([])
        candidates = self._sample_points(1000)
        acq_values = self._acquisition_function(candidates)
        
        if batch_size == 1:
            best_idx = np.argmax(acq_values)
            return candidates[best_idx].reshape(1, -1)
        else:
            selected_indices = []
            for _ in range(batch_size):
                best_idx = np.argmax(acq_values)
                selected_indices.append(best_idx)
                acq_values[best_idx] = -np.inf
            return candidates[selected_indices]

    def _update_eval_points(self, new_X: np.ndarray, new_y: np.ndarray):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.concatenate((self.X, new_X), axis=0)
            self.y = np.concatenate((self.y, new_y), axis=0)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        n_initial_points = min(2 * self.dim, self.budget // 2)
        X = self._sample_points(n_initial_points)
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += n_initial_points
        self.n_init = n_initial_points
        self._update_eval_points(X, y)
        
        for i, y_val in enumerate(self.y):
            if y_val < self.best_y:
                self.best_y = y_val
                self.best_x = self.X[i]
                
        while self.n_evals < self.budget:
            rest_of_budget = self.budget - self.n_evals 
            self._fit_model(self.X, self.y)
            
            if self.X.shape[0] < 10:
                batch_size = min(max(1, rest_of_budget // 3), 10)
            else:
                uncertainty_values = []
                for model in self.models:
                    X_scaled = self.scaler_X.transform(self.X)
                    _, sigma = model.predict(X_scaled, return_std=True)
                    uncertainty_values.append(sigma)
                uncertainty_values = np.array(uncertainty_values)
                avg_uncertainty = np.mean(uncertainty_values, axis=0)
                
                exploration_score = np.mean(avg_uncertainty)
                batch_size = min(max(1, int(rest_of_budget * (1 - exploration_score/np.max(avg_uncertainty)))), 10)
            
            next_points = self._select_next_points(batch_size)
            if next_points.size == 0:
                break
            
            next_y = []
            for x in next_points:
              try:
                x_local, y_local = self._local_search(func, x)
                next_y.append(y_local)
                if y_local < self.best_y:
                    self.best_y = y_local
                    self.best_x = x_local
              except Exception as e:
                if "Overbudget" in str(e):
                  break
                
            next_y = np.array(next_y).reshape(-1,1)
            self._update_eval_points(next_points, next_y)

            if self.n_evals >= self.budget:
              break
        return self.best_y, self.best_x
