# Description
**ASRDE_PLS_BO_v3: Adaptive Success-Rate based DE-BO with Probabilistic Local Search, Adaptive Population Size, and Improved Exploration-Exploitation Balance.** This algorithm enhances ASRDE_PLS_BO_v2 by incorporating a more sophisticated mechanism for balancing exploration and exploitation. Specifically, it introduces a dynamic temperature parameter within the acquisition function to modulate the influence of the GP's uncertainty (sigma). This temperature is annealed over time, starting with a high value to encourage exploration and gradually decreasing to promote exploitation. Furthermore, the local search probability is refined to consider the temperature-adjusted uncertainty, and a more robust method for updating F and CR parameters is used.

# Justification
The key improvements are:

1.  **Dynamic Temperature in Acquisition Function:** The introduction of a temperature parameter `T` in the acquisition function allows for a more controlled transition from exploration to exploitation. At the beginning of the optimization, `T` is high, amplifying the effect of uncertainty (sigma) and promoting exploration. As the optimization progresses, `T` decreases, reducing the influence of uncertainty and favoring exploitation of promising regions. The temperature is annealed using a simple exponential decay schedule.

2.  **Temperature-Adjusted Local Search Probability:** The probability of performing local search is now proportional to `sigma * EI / T`, which prioritizes regions with high uncertainty and high potential improvement, especially during the initial exploration phase.

3.  **Robust F and CR Update:** Instead of directly updating F and CR based on the success rate, a moving average of the success rate difference from 0.5 is used. This provides a smoother adaptation of F and CR and prevents oscillations.

These modifications aim to improve the algorithm's ability to escape local optima and converge to the global optimum more efficiently.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ASRDE_PLS_BO_v3:
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
        self.T = 1.0  # Initial temperature
        self.T_decay = 0.995 # Temperature decay rate

        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.success_rate_diff_F = 0.0
        self.success_rate_diff_CR = 0.0


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
        
        for _ in range(20):
            for i in range(self.pop_size):
                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs]
                x_mutated = x_r1 + self.F * (x_r2 - x_r3)
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

        # Adaptive F and CR based on success rate
        success_rate_F = n_success_F / self.pop_size
        success_rate_CR = n_success_CR / self.pop_size

        self.success_rate_diff_F = 0.9 * self.success_rate_diff_F + 0.1 * (success_rate_F - 0.5)
        self.success_rate_diff_CR = 0.9 * self.success_rate_diff_CR + 0.1 * (success_rate_CR - 0.5)
        
        # Dynamically adjust step sizes
        self.F_step = 0.05 * (1 - abs(self.success_rate_diff_F) * 2)  # Smaller steps when success rate is far from 0.5
        self.CR_step = 0.05 * (1 - abs(self.success_rate_diff_CR) * 2)

        self.F = np.clip(self.F + self.learning_rate * self.success_rate_diff_F, 0.1, 0.9)
        self.CR = np.clip(self.CR + self.learning_rate * self.success_rate_diff_CR, 0.1, 0.9)

        # Adaptive population size
        pop_size_change = self.pop_size_step * (0.5 - (n_success_F + n_success_CR) / (2 * self.pop_size))
        self.pop_size += int(pop_size_change * self.pop_size)
        self.pop_size = np.clip(self.pop_size, self.min_pop_size, self.max_pop_size)
        
        ei_values, sigma = self._acquisition_function(population)
        best_idx = np.argmax(ei_values)
        best_x = population[best_idx]
        best_sigma = sigma[best_idx]
        best_ei = ei_values[best_idx]

        # Probabilistic Local Search
        if np.random.rand() < np.clip(best_sigma * best_ei / self.T, 0.0, 1.0): # Probability proportional to uncertainty * EI and inversely proportional to T
            def obj_func(x):
                ei, _ = self._acquisition_function(x.reshape(1, -1))
                return -ei[0,0]
            
            res = minimize(obj_func, best_x, bounds=[(-5, 5)]*self.dim, method='L-BFGS-B')
            next_point = res.x.reshape(1, -1)
        else:
            next_point = best_x.reshape(1, -1)

        # Anneal temperature
        self.T *= self.T_decay
        
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
 The algorithm ASRDE_PLS_BO_v3 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1636 with standard deviation 0.1009.

took 842.85 seconds to run.