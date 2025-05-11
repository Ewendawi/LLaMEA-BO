from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ATSDE_APLS_EIvar_LCB_BO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 4 * dim
        self.pop_size = 5 * dim  # Initial population size
        self.max_pop_size = 10 * dim
        self.min_pop_size = 2 * dim
        self.F = 0.8
        self.CR = 0.7
        self.success_rate_F = 0.5  # Initialize success rates for F and CR
        self.success_rate_CR = 0.5
        self.learning_rate = 0.1
        self.F_step = 0.05
        self.CR_step = 0.05
        self.pop_size_step = 0.1
        self.temperature = 1.0
        self.temperature_decay = 0.95
        self.lcb_kappa = 2.0 # Kappa parameter for LCB

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

    def _acquisition_function(self, X, ei_flag=True):
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if ei_flag:
            imp = self.best_y - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0
            # Estimate the variance of EI (using a simple approximation)
            ei_var = np.abs(imp * sigma * norm.pdf(Z))
            return ei, sigma, ei_var
        else:
            lcb = mu - self.lcb_kappa * sigma
            return lcb, sigma, np.zeros_like(mu) # Placeholder for ei_var

    def _select_next_points(self, batch_size):
        population = self._sample_points(self.pop_size)
        n_success_F = 0
        n_success_CR = 0
        
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
                # Adaptive EI/LCB selection
                mu_current, sigma_current = self.gp.predict(population[i].reshape(1, -1), return_std=True)
                explore_exploit_prob = np.clip(sigma_current, 0.1, 0.9)  # Higher sigma -> more exploration (EI)
                if np.random.rand() < explore_exploit_prob:
                    ei_trial, _, _ = self._acquisition_function(x_trial.reshape(1, -1), ei_flag=True)
                    ei_current, _, _ = self._acquisition_function(population[i].reshape(1, -1), ei_flag=True)
                    if ei_trial[0, 0] > ei_current[0, 0]:
                        population[i] = x_trial
                        n_success_F += 1
                        n_success_CR += 1
                else:
                    lcb_trial, _, _ = self._acquisition_function(x_trial.reshape(1, -1), ei_flag=False)
                    lcb_current, _, _ = self._acquisition_function(population[i].reshape(1, -1), ei_flag=False)
                    if lcb_trial[0, 0] > lcb_current[0, 0]:  # LCB is maximized
                        population[i] = x_trial
                        n_success_F += 1
                        n_success_CR += 1

            self.temperature *= self.temperature_decay

        # Adaptive F and CR based on success rate
        self.success_rate_F = 0.9 * self.success_rate_F + 0.1 * (n_success_F / self.pop_size)
        self.success_rate_CR = 0.9 * self.success_rate_CR + 0.1 * (n_success_CR / self.pop_size)
        
        # Dynamically adjust step sizes
        self.F_step = 0.05 * (1 - abs(self.success_rate_F - 0.5) * 2)  # Smaller steps when success rate is far from 0.5
        self.CR_step = 0.05 * (1 - abs(self.success_rate_CR - 0.5) * 2)

        self.F = np.clip(self.F + self.learning_rate * (self.success_rate_F - 0.5), 0.1, 0.9)
        self.CR = np.clip(self.CR + self.learning_rate * (self.success_rate_CR - 0.5), 0.1, 0.9)

        # Adaptive population size
        pop_size_change = self.pop_size_step * (0.5 - (n_success_F + n_success_CR) / (2 * self.pop_size))
        self.pop_size += int(pop_size_change * self.pop_size)
        self.pop_size = np.clip(self.pop_size, self.min_pop_size, self.max_pop_size)
        
        # Choose best point based on EI or LCB
        mu_population, sigma_population = self.gp.predict(population, return_std=True)
        explore_exploit_prob = np.clip(np.mean(sigma_population), 0.1, 0.9)
        if np.random.rand() < explore_exploit_prob:
            ei_values, sigma, ei_var = self._acquisition_function(population, ei_flag=True)
            best_idx = np.argmax(ei_values)
            best_x = population[best_idx]
            best_sigma = sigma[best_idx]
            best_ei = ei_values[best_idx]
            best_ei_var = ei_var[best_idx]
        else:
            lcb_values, sigma, ei_var = self._acquisition_function(population, ei_flag=False)
            best_idx = np.argmax(lcb_values)  # Maximize LCB
            best_x = population[best_idx]
            best_sigma = sigma[best_idx]
            best_ei = lcb_values[best_idx] # Reuse variable name for LCB value
            best_ei_var = ei_var[best_idx]

        # Probabilistic Local Search
        ls_prob = np.clip(best_sigma * np.abs(best_ei) * best_ei_var, 0.0, 1.0) # Probability proportional to uncertainty * acquisition function value * EI variance
        if np.random.rand() < ls_prob:
            def obj_func(x):
                ei, _, _ = self._acquisition_function(x.reshape(1, -1), ei_flag=True) if np.random.rand() < explore_exploit_prob else self._acquisition_function(x.reshape(1, -1), ei_flag=False)
                return -ei[0,0] if np.random.rand() < explore_exploit_prob else -ei[0,0]
            
            # Local search with L-BFGS-B
            res = minimize(obj_func, best_x, bounds=[(-5, 5)]*self.dim, method='L-BFGS-B')
            
            # Random restart if L-BFGS-B gets stuck
            if not res.success:
                random_start = self._sample_points(1).flatten()
                res_random = minimize(obj_func, random_start, bounds=[(-5, 5)]*self.dim, method='L-BFGS-B')
                if res_random.fun < res.fun:
                    res = res_random
            
            next_point = res.x.reshape(1, -1)
        else:
            next_point = best_x.reshape(1, -1)

        return next_point

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
