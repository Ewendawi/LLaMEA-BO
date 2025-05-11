# Description
**ATSDE_APLS_EIvarAdaptPop_BO: Adaptive Temperature and Success-Rate based DE-BO with Adaptive Probabilistic Local Search, EI variance, and Adaptive Population Sizing based on EI improvement.** This algorithm integrates the strengths of ATSDE_APLS_EIvar_BO and ATSDE_ABLSEI_BO. It features adaptive temperature and success-rate based Differential Evolution (DE), adaptive probabilistic local search informed by EI and its variance, and dynamic population size adjustment. The key enhancement lies in adapting the population size based on the *improvement* in Expected Improvement (EI) during the DE iterations. A larger population is maintained when EI improvement stagnates, promoting exploration, while a smaller population is used when EI improvement is rapid, favoring exploitation. This balances exploration and exploitation more effectively. Additionally, the local search uses multiple restarts to escape local optima and the local search range is adapted based on the GP uncertainty.

# Justification
The algorithm builds on the successful components of ATSDE_APLS_EIvar_BO, such as adaptive temperature control, success-rate based F and CR adaptation, and probabilistic local search guided by EI and its variance. The addition of adaptive population sizing, inspired by ATSDE_ABLSEI_BO's adaptive batch size, aims to further improve the exploration-exploitation balance.

1.  **Adaptive Population Size based on EI Improvement:** The population size is dynamically adjusted based on the change in EI from one DE generation to the next. If the EI improvement is high, the population size is reduced to focus on exploitation. If the EI improvement is low, the population size is increased to encourage exploration. This adaptive adjustment helps the algorithm to efficiently allocate resources and avoid premature convergence.

2.  **EI Variance in Local Search:** The inclusion of EI variance in the local search probability calculation, as in ATSDE_APLS_EIvar_BO, allows the algorithm to prioritize local searches in regions where the GP is uncertain and the EI is high, while also considering the stability of the EI estimates.

3.  **Adaptive Local Search Range:** The range of the local search is adapted based on the GP's uncertainty (sigma), ensuring that the search is focused in promising regions while avoiding excessive exploration of the search space.

4.  **Multiple Local Search Restarts:** Performing multiple local search restarts, as in ATSDE_APLS_EIvar_BO, helps the algorithm escape local optima and find better solutions.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ATSDE_APLS_EIvarAdaptPop_BO:
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
        self.ls_restarts = 3
        self.ei_history = []

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
        ei_var = (sigma**2) * (norm.pdf(Z)**2)  # Variance of EI
        return ei, sigma, ei_var

    def _select_next_points(self, batch_size):
        population = self._sample_points(self.pop_size)
        n_success_F = 0
        n_success_CR = 0
        
        ei_values_before, _, _ = self._acquisition_function(population)
        mean_ei_before = np.mean(ei_values_before)

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
                ei_trial, _, _ = self._acquisition_function(x_trial.reshape(1, -1))
                ei_current, _, _ = self._acquisition_function(population[i].reshape(1, -1))
                
                if ei_trial[0, 0] > ei_current[0, 0]:
                    population[i] = x_trial
                    n_success_F += 1
                    n_success_CR += 1

            self.temperature *= self.temperature_decay
        
        ei_values_after, _, _ = self._acquisition_function(population)
        mean_ei_after = np.mean(ei_values_after)
        ei_improvement = mean_ei_after - mean_ei_before
        self.ei_history.append(ei_improvement)

        # Adaptive F and CR based on success rate
        self.success_rate_F = 0.9 * self.success_rate_F + 0.1 * (n_success_F / self.pop_size)
        self.success_rate_CR = 0.9 * self.success_rate_CR + 0.1 * (n_success_CR / self.pop_size)
        
        # Dynamically adjust step sizes
        self.F_step = 0.05 * (1 - abs(self.success_rate_F - 0.5) * 2)  # Smaller steps when success rate is far from 0.5
        self.CR_step = 0.05 * (1 - abs(self.success_rate_CR - 0.5) * 2)

        self.F = np.clip(self.F + self.learning_rate * (self.success_rate_F - 0.5), 0.1, 0.9)
        self.CR = np.clip(self.CR + self.learning_rate * (self.success_rate_CR - 0.5), 0.1, 0.9)

        # Adaptive population size based on EI improvement
        if len(self.ei_history) > 1:
            # Adjust population size based on the last EI improvement
            if self.ei_history[-1] > np.mean(self.ei_history[:-1]): # Significant Improvement
                pop_size_change = -self.pop_size_step  # Reduce population size
            else: # Stagnation
                pop_size_change = self.pop_size_step  # Increase population size
        else:
            pop_size_change = 0

        self.pop_size += int(pop_size_change * self.pop_size)
        self.pop_size = np.clip(self.pop_size, self.min_pop_size, self.max_pop_size)
        
        ei_values, sigma, ei_var = self._acquisition_function(population)
        best_idx = np.argmax(ei_values)
        best_x = population[best_idx]
        best_sigma = sigma[best_idx]
        best_ei = ei_values[best_idx]
        best_ei_var = ei_var[best_idx]

        # Probabilistic Local Search
        ls_prob = np.clip(best_sigma * best_ei / (best_ei_var + 1e-9), 0.0, 1.0) # Probability proportional to uncertainty * EI, inversely proportional to EI variance
        if np.random.rand() < ls_prob:
            def obj_func(x):
                ei, _, _ = self._acquisition_function(x.reshape(1, -1))
                return -ei[0,0]
            
            best_lseval = float('inf')
            best_lsx = None
            for _ in range(self.ls_restarts):
                x0 = best_x + np.random.normal(0, 0.1, self.dim)  # Add small noise for restarts
                x0 = np.clip(x0, self.bounds[0], self.bounds[1])
                
                # Adjust local search range based on GP uncertainty
                search_range = min(best_sigma[0] * 2, (self.bounds[1][0] - self.bounds[0][0]) / 2) # Limit search range
                
                res = minimize(obj_func, x0, bounds=[(best_x[i] - search_range, best_x[i] + search_range) for i in range(self.dim)], method='L-BFGS-B')
                if res.fun < best_lseval:
                    best_lseval = res.fun
                    best_lsx = res.x
            next_point = best_lsx.reshape(1, -1)
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
 The algorithm ATSDE_APLS_EIvarAdaptPop_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1596 with standard deviation 0.0982.

took 3649.60 seconds to run.