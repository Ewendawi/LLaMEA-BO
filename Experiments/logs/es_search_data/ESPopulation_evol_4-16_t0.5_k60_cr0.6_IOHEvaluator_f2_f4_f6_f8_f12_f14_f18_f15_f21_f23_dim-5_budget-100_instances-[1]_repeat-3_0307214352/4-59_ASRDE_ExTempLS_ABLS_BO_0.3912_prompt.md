You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- ATSDE_LS_BO: 0.1706, 1539.81 seconds, **Adaptive Temperature and Success-Rate based DE-BO with Local Search (ATSDE_LS_BO):** This algorithm combines the adaptive temperature control from ADExBO with the adaptive success-rate based F and CR adaptation from ASRDE_LS_BO. It also incorporates a local search step using L-BFGS-B to refine the best solution. The key innovation is the synergistic combination of exploration-exploitation balancing via temperature annealing and success-rate based parameter adaptation, leading to a more robust and efficient optimization process. Additionally, the algorithm dynamically adjusts the local search probability based on the GP's uncertainty, similar to ASRDE_PLS_BO, to balance refinement and exploration.


- ASRDE_PLS_BO_v2: 0.1695, 835.80 seconds, **ASRDE_PLS_BO_v2: Adaptive Success-Rate based DE-BO with Probabilistic Local Search and Adaptive Population Size.** This algorithm builds upon ASRDE_PLS_BO by incorporating an adaptive population size for the DE component and a more refined probabilistic local search. The population size is dynamically adjusted based on the success rate of the DE iterations, promoting exploration with larger populations when the success rate is low and exploitation with smaller populations when the success rate is high. The probabilistic local search is enhanced by considering both the GP's uncertainty (sigma) and the Expected Improvement (EI) value. The probability of performing local search is now proportional to sigma * EI, which prioritizes regions with high uncertainty and high potential improvement. The adaptation of F and CR is also refined to use a more stable moving average and dynamically adjust step sizes.


- ASRDE_ExTempLS_BO: 0.1694, 1753.54 seconds, **ASRDE-ExTempLS-BO: Adaptive Success-Rate DE with Exploration Temperature and Enhanced Local Search Bayesian Optimization.** This algorithm combines the adaptive success-rate based Differential Evolution (DE) from ASRDE_PLS_BO with the exploration temperature control from ADExBO. It further enhances the local search strategy by incorporating a dynamic adjustment of the local search range based on the GP's uncertainty, and uses a more robust local search initialization. The success rate adaptation for F and CR is also made more responsive.


- DE_ABLS_LCB_BO: 0.1687, 712.13 seconds, **DE-ABLS-LCB-BO: Differential Evolution with Adaptive Batch size, Local Search and LCB Bayesian Optimization.** This algorithm refines DE-ABLS-BO by replacing the Expected Improvement (EI) acquisition function with the Lower Confidence Bound (LCB) acquisition function. The LCB acquisition function promotes exploration by favoring regions with high uncertainty. Also, the adaption of batch size is improved with a more robust formula. The dynamic adjustment of F and CR enhances exploration/exploitation in the acquisition function optimization, the adaptive batch size balances exploration and exploitation, and the local search refines the best solution.




The selected solutions to update are:
## ASRDE_ExTempLS_BO
**ASRDE-ExTempLS-BO: Adaptive Success-Rate DE with Exploration Temperature and Enhanced Local Search Bayesian Optimization.** This algorithm combines the adaptive success-rate based Differential Evolution (DE) from ASRDE_PLS_BO with the exploration temperature control from ADExBO. It further enhances the local search strategy by incorporating a dynamic adjustment of the local search range based on the GP's uncertainty, and uses a more robust local search initialization. The success rate adaptation for F and CR is also made more responsive.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ASRDE_ExTempLS_BO:
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
                    # Adapt F and CR based on improvement in EI
                    improvement = ei_trial[0, 0] - ei_current[0, 0]
                    self.F = np.clip(self.F + self.learning_rate * improvement, 0.1, 0.9)
                    self.CR = np.clip(self.CR + self.learning_rate * improvement, 0.1, 0.9)

        # Anneal temperature
        self.temperature *= self.temperature_decay

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
The algorithm ASRDE_ExTempLS_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1694 with standard deviation 0.1065.

took 1753.54 seconds to run.

## DE_ABLS_LCB_BO
**DE-ABLS-LCB-BO: Differential Evolution with Adaptive Batch size, Local Search and LCB Bayesian Optimization.** This algorithm refines DE-ABLS-BO by replacing the Expected Improvement (EI) acquisition function with the Lower Confidence Bound (LCB) acquisition function. The LCB acquisition function promotes exploration by favoring regions with high uncertainty. Also, the adaption of batch size is improved with a more robust formula. The dynamic adjustment of F and CR enhances exploration/exploitation in the acquisition function optimization, the adaptive batch size balances exploration and exploitation, and the local search refines the best solution.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class DE_ABLS_LCB_BO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 4 * dim
        self.pop_size = 15 # Population size for DE
        self.F = 0.8 # Initial mutation factor for DE
        self.CR = 0.7 # Initial crossover rate for DE
        self.success_rate = 0.0
        self.learning_rate = 0.1
        self.de_iters = 20 # Number of DE iterations
        self.exploration_weight = 2.0

        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.batch_size = 1

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        points = sampler.random(n=n_points)
        return qmc.scale(points, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        
        # Lower Confidence Bound acquisition function
        lcb = mu - self.exploration_weight * sigma
        return lcb

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using Differential Evolution
        # return array of shape (batch_size, n_dims)

        # Initialize population
        population = self._sample_points(self.pop_size)
        n_success = 0

        # DE optimization loop
        for iter in range(self.de_iters):
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
                ei_trial = self._acquisition_function(x_trial.reshape(1, -1))[0, 0]
                ei_current = self._acquisition_function(population[i].reshape(1, -1))[0, 0]

                if ei_trial < ei_current:
                    population[i] = x_trial
                    n_success += 1
        
        #Update F and CR adaptively
        self.success_rate = 0.9 * self.success_rate + 0.1 * (n_success / self.pop_size)
        self.F = np.clip(self.F + self.learning_rate * (self.success_rate - 0.5), 0.1, 0.9)
        self.CR = np.clip(self.CR + self.learning_rate * (self.success_rate - 0.5), 0.1, 0.9)

        # Return the best point from the population
        ei_values = self._acquisition_function(population)
        best_idx = np.argmin(ei_values)
        best_x = population[best_idx]

        # Local search
        def obj_func(x):
            return self._acquisition_function(x.reshape(1, -1))[0,0]

        res = minimize(obj_func, best_x, bounds=[(-5, 5)]*self.dim, method='L-BFGS-B')
        next_point = res.x.reshape(1, -1)

        return next_point

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        # Do not change the function signature
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

        # Update best seen value
        if np.min(new_y) < self.best_y:
            self.best_y = np.min(new_y)
            self.best_x = new_X[np.argmin(new_y)]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Optimization loop
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Adaptive batch size
            _, std = self.gp.predict(self.X, return_std=True)
            mean_std = np.mean(std)
            self.batch_size = max(1, int(self.dim / (1 + mean_std * 10))) # Adjust batch size based on uncertainty
            self.batch_size = min(self.batch_size, self.budget - self.n_evals) # Ensure not exceeding budget

            # Select next points by acquisition function
            next_X = self._select_next_points(self.batch_size)

            # Evaluate the points
            next_y = self._evaluate_points(func, next_X)

            # Update evaluated points
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x

```
The algorithm DE_ABLS_LCB_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1687 with standard deviation 0.0967.

took 712.13 seconds to run.

Combine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it. If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.




Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

