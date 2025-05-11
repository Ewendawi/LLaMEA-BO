# Description
**ATSDE_APLS_EIGrad_BO: Adaptive Temperature and Success-Rate based DE-BO with Adaptive Probabilistic Local Search using EI Gradient.** This algorithm builds upon ATSDE_APLS_EIvar_BO by incorporating the gradient of the Expected Improvement (EI) into the probabilistic local search. The local search probability and direction are dynamically adjusted based on GP uncertainty, EI, and the gradient of EI. This allows for a more informed decision on whether to perform local search and guides the search towards regions with high potential for improvement. It also includes adaptive temperature control, success-rate based F and CR adaptation for Differential Evolution (DE), and adaptive batch size.

# Justification
The core idea is to leverage the gradient of the EI to guide the local search more effectively. The previous versions relied on EI variance and multiple restarts, but using the gradient provides a more direct indication of the direction of improvement. This should lead to faster convergence and better exploitation of promising regions.

Key components:
1.  **EI Gradient Calculation:** The algorithm calculates the gradient of the EI with respect to the input X. This gradient indicates the direction in which the EI is increasing.
2.  **Adaptive Probabilistic Local Search with Gradient Guidance:** The probability of performing local search is still based on GP uncertainty and EI, but the local search step is now guided by the EI gradient. This allows the local search to efficiently explore the neighborhood of promising points.
3.  **Adaptive Temperature and Success-Rate based DE:** This component, inherited from previous algorithms, provides a robust global search strategy.
4.  **Adaptive Batch Size:** The population size is adapted based on the success rate, which helps balance exploration and exploitation.
5.  **L-BFGS-B with Gradient Information:** The local search uses L-BFGS-B, a gradient-based optimization algorithm, to find the local optimum. The EI gradient is used to inform the search direction.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from scipy.optimize import approx_fprime

class ATSDE_APLS_EIGrad_BO:
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

    def _acquisition_function_grad(self, x):
        # Numerical gradient of EI
        delta = 1e-6
        grad = approx_fprime(x, lambda x: self._acquisition_function(x.reshape(1, -1))[0], delta)
        return grad

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
                ei_trial, _ = self._acquisition_function(x_trial.reshape(1, -1))
                ei_current, _ = self._acquisition_function(population[i].reshape(1, -1))
                
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
        
        ei_values, sigma = self._acquisition_function(population)
        best_idx = np.argmax(ei_values)
        best_x = population[best_idx]
        best_sigma = sigma[best_idx]
        best_ei = ei_values[best_idx]

        # Probabilistic Local Search
        ls_prob = np.clip(best_sigma * best_ei, 0.0, 1.0) # Probability proportional to uncertainty * EI
        if np.random.rand() < ls_prob:
            ei_grad = self._acquisition_function_grad(best_x)
            
            def obj_func(x):
                ei, _ = self._acquisition_function(x.reshape(1, -1))
                return -ei[0,0]
            
            # Local search with L-BFGS-B, using the gradient
            res = minimize(obj_func, best_x, jac=lambda x: -self._acquisition_function_grad(x), bounds=[(-5, 5)]*self.dim, method='L-BFGS-B')
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
```
## Error
 Traceback (most recent call last):
  File "<ATSDE_APLS_EIGrad_BO>", line 158, in __call__
 158->             next_X = self._select_next_points(1)
  File "<ATSDE_APLS_EIGrad_BO>", line 120, in _select_next_points
 120->             ei_grad = self._acquisition_function_grad(best_x)
  File "<ATSDE_APLS_EIGrad_BO>", line 62, in _acquisition_function_grad
  60 |         # Numerical gradient of EI
  61 |         delta = 1e-6
  62->         grad = approx_fprime(x, lambda x: self._acquisition_function(x.reshape(1, -1))[0], delta)
  63 |         return grad
  64 | 
ValueError: `f0` passed has more than 1 dimension.
