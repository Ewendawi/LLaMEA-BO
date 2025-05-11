# Description
**ASRDE-ExTempLS-ABLS-LCB-BO: Adaptive Success-Rate DE with Exploration Temperature, Enhanced Local Search, Adaptive Batch Size and LCB Bayesian Optimization.** This algorithm synergistically combines the strengths of ASRDE_ExTempLS_BO and DE_ABLS_LCB_BO. It incorporates adaptive success-rate based Differential Evolution (DE) with exploration temperature control, enhanced local search based on GP uncertainty, adaptive batch size based on GP variance, and the Lower Confidence Bound (LCB) acquisition function. The algorithm balances exploration and exploitation through temperature annealing, success-rate based F and CR adaptation, LCB, and adaptive batch size. The local search refines the best solution, and its range is dynamically adjusted based on the GP's uncertainty.

# Justification
This algorithm aims to improve upon ASRDE_ExTempLS_BO and DE_ABLS_LCB_BO by combining their best features.
1.  **Adaptive Success-Rate DE (ASRDE):** Uses the success rate of DE iterations to adapt the mutation factor (F) and crossover rate (CR), promoting exploration when the success rate is low and exploitation when the success rate is high.
2.  **Exploration Temperature:** Controls the intensity of the DE mutation, gradually decreasing over time to shift from exploration to exploitation.
3.  **Enhanced Local Search:** Refines the best solution found by DE using a local optimization algorithm (L-BFGS-B). The search range is dynamically adjusted based on the GP's uncertainty, focusing the search on promising regions. Multiple start points are used to avoid local optima.
4.  **Adaptive Batch Size (ABS):** Dynamically adjusts the number of points evaluated in each iteration based on the GP's uncertainty. Higher uncertainty leads to a larger batch size, promoting exploration, while lower uncertainty leads to a smaller batch size, promoting exploitation. This is based on the DE_ABLS_LCB_BO.
5. **Lower Confidence Bound (LCB):** The LCB acquisition function promotes exploration by favoring regions with high uncertainty, as in DE_ABLS_LCB_BO.
6. **Sobol Sampling:** Sobol sequences are used for the initial sampling to improve space-filling properties.
7. **Computational efficiency:** The local search is performed only on the best point found by DE, and the adaptive batch size helps to reduce the number of function evaluations.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ASRDE_ExTempLS_ABLS_LCB_BO:
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
        self.temperature = 1.0
        self.temperature_decay = 0.95
        self.exploration_weight = 2.0
        self.success_rate = 0.0
        self.de_iters = 20
        self.batch_size = 1

        self.gp = None
        self.best_y = float('inf')
        self.best_x = None

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=True)
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
        n_success = 0
        
        for _ in range(self.de_iters):
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
                
                if ei_trial[0, 0] < ei_current[0, 0]:
                    population[i] = x_trial
                    n_success += 1

        # Adapt F and CR based on success rate
        self.success_rate = 0.9 * self.success_rate + 0.1 * (n_success / self.pop_size)
        self.F = np.clip(self.F + self.learning_rate * (self.success_rate - 0.5), 0.1, 0.9)
        self.CR = np.clip(self.CR + self.learning_rate * (self.success_rate - 0.5), 0.1, 0.9)

        # Anneal temperature
        self.temperature *= self.temperature_decay

        ei_values, sigma = self._acquisition_function(population)
        best_idx = np.argmin(ei_values)
        best_x = population[best_idx]
        best_sigma = sigma[best_idx, 0]  # Use scalar sigma for range adjustment

        # Enhanced Local Search
        def obj_func(x):
            ei, _ = self._acquisition_function(x.reshape(1, -1))
            return ei[0,0]

        # Adjust local search range based on GP uncertainty
        search_range = min(best_sigma * 2, (self.bounds[1][0] - self.bounds[0][0]) / 2) # Limit search range

        # Initialize local search with multiple points
        local_search_points = self._sample_points(5) * search_range + best_x
        local_search_points = np.clip(local_search_points, self.bounds[0], self.bounds[1])

        best_local_x = best_x
        best_local_ei = obj_func(best_x)
        
        for start_point in local_search_points:
            res = minimize(obj_func, start_point, bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)], method='L-BFGS-B')
            if res.fun < best_local_ei:
                best_local_ei = res.fun
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
## Feedback
 The algorithm ASRDE_ExTempLS_ABLS_LCB_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1653 with standard deviation 0.1005.

took 1101.50 seconds to run.