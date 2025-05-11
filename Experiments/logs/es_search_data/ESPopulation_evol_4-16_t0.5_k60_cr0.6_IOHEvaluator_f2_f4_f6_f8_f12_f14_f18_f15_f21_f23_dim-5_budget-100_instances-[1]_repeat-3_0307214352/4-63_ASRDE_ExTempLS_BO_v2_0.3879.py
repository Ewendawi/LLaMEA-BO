from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ASRDE_ExTempLS_BO_v2:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 4 * dim
        self.pop_size = 5 * dim
        self.F = 0.8
        self.CR = 0.7
        self.learning_rate = 0.1
        self.F_step = 0.05
        self.CR_step = 0.05
        self.temperature = 1.0
        self.temperature_decay = 0.95
        self.success_history = []
        self.success_window = 10
        self.F_momentum = 0.0
        self.CR_momentum = 0.0
        self.momentum_factor = 0.5

        self.gp = None
        self.best_y = float('inf')
        self.best_x = None

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        points = sampler.random(n=n_points)
        return qmc.scale(points, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0
        return ei, sigma

    def _select_next_points(self, batch_size):
        population = self._sample_points(self.pop_size)
        success_count = 0
        
        for _ in range(20):
            for i in range(self.pop_size):
                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs]
                x_mutated = x_r1 + self.F * (x_r2 - x_r3) * self.temperature
                x_mutated = np.clip(x_mutated, self.bounds[0], self.bounds[1])

                # Crossover
                x_trial = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        x_trial[j] = x_mutated[j]

                # Selection
                ei_trial, _ = self._acquisition_function(x_trial.reshape(1, -1))
                ei_current, _ = self._acquisition_function(population[i].reshape(1, -1))
                
                if ei_trial[0, 0] > ei_current[0, 0]:
                    population[i] = x_trial
                    improvement = ei_trial[0, 0] - ei_current[0, 0]
                    
                    # Adaptive F and CR step size
                    if len(self.success_history) > 0:
                        success_rate = np.mean(self.success_history)
                    else:
                        success_rate = 0.5
                    
                    f_step = self.F_step * (1 - success_rate)  # Smaller step when success is high
                    cr_step = self.CR_step * (1 - success_rate)
                    
                    # Momentum for F and CR
                    self.F_momentum = self.momentum_factor * self.F_momentum + (1 - self.momentum_factor) * f_step * improvement
                    self.CR_momentum = self.momentum_factor * self.CR_momentum + (1 - self.momentum_factor) * cr_step * improvement
                    
                    self.F = np.clip(self.F + self.F_momentum, 0.1, 0.9)
                    self.CR = np.clip(self.CR + self.CR_momentum, 0.1, 0.9)
                    success_count += 1
                
            self.success_history.append(success_count / self.pop_size)
            if len(self.success_history) > self.success_window:
                self.success_history.pop(0)

        # Anneal temperature
        self.temperature *= self.temperature_decay

        ei_values, sigma = self._acquisition_function(population)
        best_idx = np.argmax(ei_values)
        best_x = population[best_idx]
        best_sigma = sigma[best_idx, 0]  # Use scalar sigma for range adjustment

        # Enhanced Local Search
        def obj_func(x):
            ei, _ = self._acquisition_function(x.reshape(1, -1))
            return -ei[0,0]

        # Adjust local search range based on GP uncertainty
        search_range = min(best_sigma * 2, (self.bounds[1][0] - self.bounds[0][0]) / 2) # Limit search range

        # Improved Local search initialization
        local_search_points = self._sample_points(3) * search_range + best_x
        if self.best_x is not None:
            local_search_points = np.vstack([local_search_points, self.best_x + np.random.randn(1, self.dim) * search_range/2])
        local_search_points = np.clip(local_search_points, self.bounds[0], self.bounds[1])

        best_local_x = best_x
        best_local_ei = -obj_func(best_x)
        
        for start_point in local_search_points:
            res = minimize(obj_func, start_point, bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)], method='L-BFGS-B')
            if -res.fun > best_local_ei:
                best_local_ei = -res.fun
                best_local_x = res.x

        return best_local_x.reshape(1, -1)

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
            
        if np.min(new_y) < self.best_y:
            self.best_y = np.min(new_y)
            self.best_x = new_X[np.argmin(new_y)]
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        
        while self.n_evals < self.budget:
            self._fit_model(self.X, self.y)
            next_X = self._select_next_points(1)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
