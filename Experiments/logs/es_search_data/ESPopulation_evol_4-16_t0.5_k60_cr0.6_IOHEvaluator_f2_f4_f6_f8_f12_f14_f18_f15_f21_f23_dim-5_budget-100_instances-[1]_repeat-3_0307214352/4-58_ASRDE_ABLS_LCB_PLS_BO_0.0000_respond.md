# Description
**ASRDE-ABLS-LCB-PLS-BO: Adaptive Success-Rate DE with Adaptive Batch size, LCB, Probabilistic Local Search Bayesian Optimization.** This algorithm combines the strengths of ASRDE_PLS_BO_v2 and DE_ABLS_LCB_BO. It uses Adaptive Success-Rate based Differential Evolution (ASRDE) for efficient global exploration, Adaptive Batch Size (ABLS) to balance exploration and exploitation based on GP uncertainty, Lower Confidence Bound (LCB) as the acquisition function to promote exploration, and Probabilistic Local Search (PLS) to refine promising solutions. The F and CR parameters of DE are adapted based on their success rate. The population size is also dynamically adjusted. The local search probability is proportional to the product of GP's uncertainty (sigma) and the Expected Improvement (EI).

# Justification
This algorithm combines the strengths of its predecessors to achieve a better balance between exploration and exploitation.

*   **Adaptive Success-Rate based DE (ASRDE):** The adaptive F and CR parameters in DE allow for efficient exploration of the search space.
*   **Adaptive Batch Size (ABLS):** Adjusting the batch size based on the GP's uncertainty helps balance exploration and exploitation. When the uncertainty is high, a larger batch size is used to explore more broadly. When the uncertainty is low, a smaller batch size is used to exploit the promising regions.
*   **Lower Confidence Bound (LCB):** The LCB acquisition function encourages exploration by favoring regions with high uncertainty.
*   **Probabilistic Local Search (PLS):** The probabilistic local search refines promising solutions by considering both the GP's uncertainty and the Expected Improvement. The local search probability is proportional to the product of GP's uncertainty (sigma) and the Expected Improvement (EI).
*   **Adaptive Population Size:** Dynamically adjusting population size based on success rate of DE.

The combination of these techniques allows the algorithm to efficiently explore the search space and refine promising solutions, leading to improved performance.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ASRDE_ABLS_LCB_PLS_BO:
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
        self.exploration_weight = 2.0
        self.de_iters = 20

        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.batch_size = 1

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

        # Lower Confidence Bound acquisition function
        lcb = mu - self.exploration_weight * sigma
        return lcb, sigma

    def _select_next_points(self, batch_size):
        population = self._sample_points(self.pop_size)
        n_success_F = 0
        n_success_CR = 0
        
        for _ in range(self.de_iters):
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
                
                if ei_trial[0, 0] < ei_current[0, 0]: #LCB: smaller is better
                    population[i] = x_trial
                    n_success_F += 1
                    n_success_CR += 1

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
        
        ei_values, sigma = self._acquisition_function(population)
        best_idx = np.argmin(ei_values) #LCB: smaller is better
        best_x = population[best_idx]
        best_sigma = sigma[best_idx]
        best_ei = ei_values[best_idx]

        # Probabilistic Local Search
        ei_val_for_ls = -best_ei[0,0] # convert LCB to EI
        if np.random.rand() < np.clip(best_sigma * ei_val_for_ls, 0.0, 1.0): # Probability proportional to uncertainty * EI
            def obj_func(x):
                ei, _ = self._acquisition_function(x.reshape(1, -1))
                return ei[0,0] #LCB: smaller is better
            
            res = minimize(obj_func, best_x, bounds=[(-5, 5)]*self.dim, method='L-BFGS-B')
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

            # Adaptive batch size
            _, std = self.gp.predict(self.X, return_std=True)
            mean_std = np.mean(std)
            self.batch_size = max(1, int(self.dim / (1 + mean_std * 10))) # Adjust batch size based on uncertainty
            self.batch_size = min(self.batch_size, self.budget - self.n_evals) # Ensure not exceeding budget
            
            next_X = self._select_next_points(self.batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<ASRDE_ABLS_LCB_PLS_BO>", line 152, in __call__
 152->             next_X = self._select_next_points(self.batch_size)
  File "<ASRDE_ABLS_LCB_PLS_BO>", line 108, in _select_next_points
 106 | 
 107 |         # Probabilistic Local Search
 108->         ei_val_for_ls = -best_ei[0,0] # convert LCB to EI
 109 |         if np.random.rand() < np.clip(best_sigma * ei_val_for_ls, 0.0, 1.0): # Probability proportional to uncertainty * EI
 110 |             def obj_func(x):
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
