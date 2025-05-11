# Description
**ATSDE_APLS_EIvar_BatchBO: Adaptive Temperature and Success-Rate based DE-BO with Adaptive Probabilistic Local Search, EI variance, and Batch Evaluation.** This algorithm enhances ATSDE_APLS_EIvar_BO by incorporating batch evaluation of multiple points selected via DE, improving the exploration-exploitation balance. It adapts the batch size based on GP uncertainty. The local search probability is dynamically adjusted based on GP uncertainty, EI, and the variance of EI. This encourages local search in regions where the GP is both uncertain and the EI is high, but also where the EI values are less stable, indicating potential for improvement. Additionally, the local search is enhanced by using a combination of L-BFGS-B and a random restart strategy to escape local optima.

# Justification
The key improvement is the introduction of batch evaluation. Instead of selecting and evaluating a single point in each iteration, the algorithm now selects a batch of points using the DE strategy and evaluates them together. This improves the exploration of the search space and can lead to faster convergence. The batch size is adapted based on the GP's uncertainty, allowing for more exploration when the GP is uncertain and more exploitation when the GP is confident. This adaptive batch size helps to balance exploration and exploitation. The local search component remains the same, leveraging EI variance to guide local search efforts.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ATSDE_APLS_EIvar_BatchBO:
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
        self.batch_size = 1 # Initial batch size
        self.max_batch_size = 5
        self.min_batch_size = 1

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
        
        # Select top batch_size points based on EI
        idxs = np.argsort(ei_values.flatten())[::-1][:batch_size]
        next_points = population[idxs]
        sigmas = sigma[idxs]
        eis = ei_values[idxs]
        ei_vars = ei_var[idxs]

        # Probabilistic Local Search for each point in the batch
        refined_next_points = []
        for i in range(batch_size):
            best_x = next_points[i]
            best_sigma = sigmas[i]
            best_ei = eis[i]
            best_ei_var = ei_vars[i]

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
                
                refined_next_points.append(res.x.reshape(1, -1))
            else:
                refined_next_points.append(best_x.reshape(1, -1))
        
        return np.array(refined_next_points).reshape(batch_size, self.dim)

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
            
            # Adaptive Batch Size based on GP's uncertainty
            mean_sigma = np.mean(self.gp.predict(self.X, return_std=True)[1])
            self.batch_size = int(np.clip(self.max_batch_size * (mean_sigma / 0.5), self.min_batch_size, self.max_batch_size)) # Scale by 0.5 to normalize sigma
            
            next_X = self._select_next_points(self.batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ATSDE_APLS_EIvar_BatchBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1641 with standard deviation 0.1021.

took 993.03 seconds to run.