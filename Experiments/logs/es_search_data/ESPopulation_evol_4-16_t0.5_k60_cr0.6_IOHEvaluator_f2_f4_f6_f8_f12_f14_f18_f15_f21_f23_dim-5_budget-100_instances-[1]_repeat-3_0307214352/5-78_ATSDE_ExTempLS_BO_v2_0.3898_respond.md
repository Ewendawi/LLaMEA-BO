# Description
**ATSDE_ExTempLS_BO_v2: Adaptive Temperature and Success-Rate based DE-BO with Exploration Temperature, Enhanced Local Search, Adaptive F/CR Step Size and Momentum.** This algorithm builds upon ATSDE_ExTempLS_BO by incorporating adaptive step sizes and momentum for F and CR adaptation in Differential Evolution (DE). The local search initialization is improved by using a combination of Latin Hypercube sampling and the best point found so far. Also, the local search range is dynamically adjusted based on the GP's uncertainty, and a more robust local search initialization is used. Additionally, the algorithm employs a success rate-based adaptation for F and CR, making it more responsive to changes in the optimization landscape. The core idea is to leverage both global exploration (DE with temperature annealing) and local refinement (enhanced local search) in a balanced manner, guided by the Gaussian Process (GP) surrogate model.

# Justification
The adaptive step size for F and CR allows for finer adjustments when the algorithm is converging well and larger adjustments when it is stuck. Momentum is added to smooth the updates. This should lead to better exploration-exploitation balance and faster convergence. The local search initialization is improved by using a combination of Latin Hypercube sampling and the best point found so far.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ATSDE_ExTempLS_BO_v2:
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
        self.success_rate = 0.0

        self.gp = None
        self.best_y = float('inf')
        self.best_x = None

        # Adaptive F/CR step size and momentum
        self.F_step_size = 0.1
        self.CR_step_size = 0.1
        self.F_momentum = 0.0
        self.CR_momentum = 0.0
        self.momentum_factor = 0.6  # Momentum factor

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
        n_success = 0
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
                    n_success += 1
                    # Adapt F and CR based on improvement in EI
                    improvement = ei_trial[0, 0] - ei_current[0, 0]
                    F_change = self.F_step_size * (1 - improvement)
                    CR_change = self.CR_step_size * (1 - improvement)
                    self.F_momentum = self.momentum_factor * self.F_momentum + (1 - self.momentum_factor) * F_change
                    self.CR_momentum = self.momentum_factor * self.CR_momentum + (1 - self.momentum_factor) * CR_change

                    self.F = min(1.0, self.F + self.F_momentum)
                    self.CR = min(1.0, self.CR + self.CR_momentum)
                else:
                    # Adapt F and CR: Decrease if no improvement (EI improvement)
                    F_change = -self.F_step_size
                    CR_change = -self.CR_step_size
                    self.F_momentum = self.momentum_factor * self.F_momentum + (1 - self.momentum_factor) * F_change
                    self.CR_momentum = self.momentum_factor * self.CR_momentum + (1 - self.momentum_factor) * CR_change

                    self.F = max(0.1, self.F + self.F_momentum)
                    self.CR = max(0.1, self.CR + self.CR_momentum)

        # Anneal temperature
        self.temperature *= self.temperature_decay

        # Update F and CR adaptively (success rate)
        self.success_rate = 0.9 * self.success_rate + 0.1 * (n_success / self.pop_size)
        self.F = np.clip(self.F + self.learning_rate * (self.success_rate - 0.5), 0.1, 0.9)
        self.CR = np.clip(self.CR + self.learning_rate * (self.success_rate - 0.5), 0.1, 0.9)

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

        # Initialize local search with multiple points
        local_search_points = self._sample_points(5) * search_range + best_x
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
```
## Feedback
 The algorithm ATSDE_ExTempLS_BO_v2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1641 with standard deviation 0.1011.

took 1736.32 seconds to run.