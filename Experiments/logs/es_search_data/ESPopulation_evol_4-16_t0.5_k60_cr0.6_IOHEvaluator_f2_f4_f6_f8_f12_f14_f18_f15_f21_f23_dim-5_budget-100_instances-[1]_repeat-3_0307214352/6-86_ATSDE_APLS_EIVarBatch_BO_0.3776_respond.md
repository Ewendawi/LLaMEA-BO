# Description
**ATSDE_APLS_EIVarBatch_BO: Adaptive Temperature and Success-Rate based DE-BO with Adaptive Probabilistic Local Search, EI variance, and Batch Evaluation.** This algorithm combines the strengths of ATSDE_APLS_EIvar_BO and ATSDE_ABLSEI_BO. It integrates the adaptive temperature and success-rate based Differential Evolution (DE) with adaptive probabilistic local search, incorporating the variance of Expected Improvement (EI) to guide the local search. Additionally, it adopts an adaptive batch size strategy, adjusting the number of points evaluated in each iteration based on the GP's uncertainty. The local search is enhanced by using a combination of L-BFGS-B and a random restart strategy to escape local optima. The adaptive batch size helps balance exploration and exploitation, while the EI variance-aware local search focuses computational effort on promising and uncertain regions.

# Justification
The algorithm leverages the strengths of both parent algorithms:
- **ATSDE_APLS_EIvar_BO:** The EI variance-aware local search allows for a more informed decision on whether to perform local search, favoring regions where the potential improvement is not only high but also more reliable (lower variance). This improves the efficiency of local search. The combination of L-BFGS-B and random restarts helps escape local optima.
- **ATSDE_ABLSEI_BO:** The adaptive batch size strategy allows for better exploration/exploitation balance by adjusting the number of points evaluated based on the GP's uncertainty. This can lead to faster convergence, especially in the early stages of optimization.

The combination of these strategies should lead to a more robust and efficient algorithm, capable of handling a wide range of black-box optimization problems.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ATSDE_APLS_EIVarBatch_BO:
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
        self.batch_size = 1

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

        # Estimate the variance of EI (using a simple approximation)
        ei_var = np.abs(imp * sigma * norm.pdf(Z))
        return ei, sigma, ei_var

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
                ei_trial, _, _ = self._acquisition_function(x_trial.reshape(1, -1))
                ei_current, _, _ = self._acquisition_function(population[i].reshape(1, -1))
                
                if ei_trial[0, 0] > ei_current[0, 0]:
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
        
        ei_values, sigma, ei_var = self._acquisition_function(population)
        best_idx = np.argmax(ei_values)
        best_x = population[best_idx]
        best_sigma = sigma[best_idx]
        best_ei = ei_values[best_idx]
        best_ei_var = ei_var[best_idx]

        next_points = []
        for _ in range(batch_size):
            # Probabilistic Local Search
            ls_prob = np.clip(best_sigma * best_ei * best_ei_var, 0.0, 1.0) # Probability proportional to uncertainty * EI * EI variance
            if np.random.rand() < ls_prob:
                def obj_func(x):
                    ei, _, _ = self._acquisition_function(x.reshape(1, -1))
                    return -ei[0,0]
                
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
            
            next_points.append(next_point)
            
            # Update best_x for the next iteration in the batch
            ei_next, sigma, ei_var = self._acquisition_function(next_point)
            if ei_next > best_ei:
                best_x = next_point.flatten()
                best_ei = ei_next
                best_sigma = sigma
                best_ei_var = ei_var

        return np.vstack(next_points)

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
 The algorithm ATSDE_APLS_EIVarBatch_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1557 with standard deviation 0.1012.

took 341.63 seconds to run.