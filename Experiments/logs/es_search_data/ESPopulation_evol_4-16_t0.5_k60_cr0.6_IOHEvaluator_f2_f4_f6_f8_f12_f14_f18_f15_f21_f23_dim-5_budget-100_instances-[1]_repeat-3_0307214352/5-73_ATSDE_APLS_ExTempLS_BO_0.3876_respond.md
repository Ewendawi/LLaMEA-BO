# Description
**ATSDE_APLS_ExTempLS_BO: Adaptive Temperature and Success-Rate based DE-BO with Adaptive Probabilistic Local Search, Exploration Temperature, and Enhanced Local Search.** This algorithm combines adaptive temperature control, success-rate based F and CR adaptation, adaptive population size, probabilistic local search (APLS) from ATSDE_APLS_BO, and exploration temperature with enhanced local search (ExTempLS) from ASRDE_ExTempLS_BO_v2. The key improvement is a more robust local search strategy that leverages both GP uncertainty and EI, combined with momentum-based F/CR adaptation and temperature annealing. The algorithm dynamically adjusts the local search range and initializes it with multiple points, including the best-so-far point.

# Justification
This algorithm aims to improve upon ATSDE_APLS_BO and ASRDE_ExTempLS_BO_v2 by synergistically combining their strengths.
1.  **Adaptive Temperature and Success-Rate based DE:** This allows for efficient exploration and exploitation by tuning the DE parameters based on the optimization landscape.
2.  **Adaptive Population Size:** Adjusting the population size dynamically can improve the search efficiency by allocating more resources when needed and reducing them when the search is converging.
3.  **Probabilistic Local Search (APLS):** This allows for local refinement of promising solutions, guided by GP uncertainty and EI.
4.  **Exploration Temperature and Enhanced Local Search (ExTempLS):** This component focuses on refining the local search by adjusting the search range based on GP uncertainty and using multiple initialization points, including the best-so-far point, to avoid getting stuck in local optima.
5.  **Momentum-based F/CR Adaptation:** The momentum term in F and CR adaptation smooths the updates and allows for finer adjustments when the algorithm is converging well, and larger adjustments when it is stuck.
6.  **Computational Efficiency:** By combining the best features of both algorithms and streamlining the local search, the computational overhead is minimized. The local search is only performed on the best point from the DE step, and the number of local search restarts is limited.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ATSDE_APLS_ExTempLS_BO:
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
        n_success_F = 0
        n_success_CR = 0
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
                    n_success_F += 1
                    n_success_CR += 1
                    improvement = ei_trial[0, 0] - ei_current[0, 0]
                    success_count += 1
                    
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

            self.temperature *= self.temperature_decay
            self.success_history.append(success_count / self.pop_size)
            if len(self.success_history) > self.success_window:
                self.success_history.pop(0)

        # Adaptive F and CR based on success rate
        self.success_rate_F = 0.9 * self.success_rate_F + 0.1 * (n_success_F / self.pop_size)
        self.success_rate_CR = 0.9 * self.success_rate_CR + 0.1 * (n_success_CR / self.pop_size)
        
        # Dynamically adjust step sizes
        self.F_step = 0.05 * (1 - abs(self.success_rate_F - 0.5) * 2)  # Smaller steps when success rate is far from 0.5
        self.CR_step = 0.05 * (1 - abs(self.success_rate_CR - 0.5) * 2)

        # Adaptive population size
        pop_size_change = self.pop_size_step * (0.5 - (n_success_F + n_success_CR) / (2 * self.pop_size))
        self.pop_size += int(pop_size_change * self.pop_size)
        self.pop_size = np.clip(self.pop_size, self.min_pop_size, self.max_pop_size)
        
        ei_values, sigma = self._acquisition_function(population)
        best_idx = np.argmax(ei_values)
        best_x = population[best_idx]
        best_sigma = sigma[best_idx, 0]
        best_ei = ei_values[best_idx]

        # Probabilistic Local Search
        ls_prob = np.clip(best_sigma * best_ei, 0.0, 1.0) # Probability proportional to uncertainty * EI
        
        if np.random.rand() < ls_prob:
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

            next_point = best_local_x.reshape(1, -1)
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
```
## Feedback
 The algorithm ATSDE_APLS_ExTempLS_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1635 with standard deviation 0.1007.

took 935.20 seconds to run.