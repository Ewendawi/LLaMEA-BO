You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- DE_ABLS_BO: 0.1662, 1092.81 seconds, **DE-ABLS-BO: Differential Evolution with Adaptive Batch size and Local Search Bayesian Optimization.** This algorithm combines the strengths of DE_BO_Adaptive and DE_ABS_BO, incorporating adaptive control of DE parameters (mutation factor F and crossover rate CR) based on success rate, adaptive batch size based on GP uncertainty, and a local search step using L-BFGS-B to refine the best solutions. It uses a Gaussian Process (GP) surrogate model and the Expected Improvement (EI) acquisition function. The dynamic adjustment of F and CR enhances exploration/exploitation in the acquisition function optimization, the adaptive batch size balances exploration and exploitation based on the GP's uncertainty, and the local search refines the best solution.


- ASRDE_PLS_BO: 0.1653, 1599.71 seconds, Adaptive Success-Rate based DE-BO with Probabilistic Local Search (ASRDE-PLS-BO): This algorithm combines the strengths of Adaptive_DE_BO and DE_BO_Adaptive by incorporating adaptive mutation and crossover rates based on success rate, and further enhances it with a probabilistic local search. Instead of always performing local search, it does so with a probability proportional to the uncertainty of the GP model. This aims to refine promising solutions while saving computational cost when the GP is already confident. The adaptation of F and CR is modified to use a more stable moving average, and the step sizes are also dynamically adjusted based on the success rate.


- ADExBO: 0.1649, 1613.88 seconds, Adaptive DE with Exploration-Exploitation Balancing Bayesian Optimization (ADExBO): This algorithm refines the Adaptive_DE_BO by incorporating a dynamic adjustment of the exploration-exploitation trade-off within the Differential Evolution (DE) optimization of the Expected Improvement (EI) acquisition function. It introduces a temperature parameter that controls the intensity of exploration in DE. The temperature is annealed over time, starting with a high value to promote exploration and gradually decreasing to favor exploitation. Additionally, the adaptation of F and CR is modified to consider the magnitude of improvement, providing finer control over the adaptation process. A local search step using L-BFGS-B is added to refine the best solution found by DE.


- ASRDE_LS_BO: 0.1643, 1604.52 seconds, Adaptive Success-Rate DE-BO with Local Search (ASRDE_LS_BO): This algorithm combines the adaptive mutation and crossover rate control from `DE_BO_Adaptive` with the dynamic adaptation of F and CR based directly on improvement in EI from `Adaptive_DE_BO`. It further incorporates a local search step using L-BFGS-B, similar to `DE_BO_Adaptive`, to refine the best solution found by DE. The key innovation is the combination of two different adaptive schemes for F and CR, aiming to leverage the strengths of both.




The selected solutions to update are:
## DE_ABLS_BO
**DE-ABLS-BO: Differential Evolution with Adaptive Batch size and Local Search Bayesian Optimization.** This algorithm combines the strengths of DE_BO_Adaptive and DE_ABS_BO, incorporating adaptive control of DE parameters (mutation factor F and crossover rate CR) based on success rate, adaptive batch size based on GP uncertainty, and a local search step using L-BFGS-B to refine the best solutions. It uses a Gaussian Process (GP) surrogate model and the Expected Improvement (EI) acquisition function. The dynamic adjustment of F and CR enhances exploration/exploitation in the acquisition function optimization, the adaptive batch size balances exploration and exploitation based on the GP's uncertainty, and the local search refines the best solution.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class DE_ABLS_BO:
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

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0 # avoid nan values
        return ei

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

                if ei_trial > ei_current:
                    population[i] = x_trial
                    n_success += 1
        
        #Update F and CR adaptively
        self.success_rate = 0.9 * self.success_rate + 0.1 * (n_success / self.pop_size)
        self.F = np.clip(self.F + self.learning_rate * (self.success_rate - 0.5), 0.1, 0.9)
        self.CR = np.clip(self.CR + self.learning_rate * (self.success_rate - 0.5), 0.1, 0.9)

        # Return the best point from the population
        ei_values = self._acquisition_function(population)
        best_idx = np.argmax(ei_values)
        best_x = population[best_idx]

        # Local search
        def obj_func(x):
            return -self._acquisition_function(x.reshape(1, -1))[0,0]

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
The algorithm DE_ABLS_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1662 with standard deviation 0.1013.

took 1092.81 seconds to run.

## ASRDE_LS_BO
Adaptive Success-Rate DE-BO with Local Search (ASRDE_LS_BO): This algorithm combines the adaptive mutation and crossover rate control from `DE_BO_Adaptive` with the dynamic adaptation of F and CR based directly on improvement in EI from `Adaptive_DE_BO`. It further incorporates a local search step using L-BFGS-B, similar to `DE_BO_Adaptive`, to refine the best solution found by DE. The key innovation is the combination of two different adaptive schemes for F and CR, aiming to leverage the strengths of both.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ASRDE_LS_BO:
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
        self.pop_size = 5 * dim # Population size for DE, scaled with dimension
        self.F = 0.8 # Initial mutation factor for DE
        self.CR = 0.7 # Initial crossover rate for DE
        self.success_rate = 0.0
        self.learning_rate = 0.1
        self.F_step = 0.05 # Step size for adapting F (EI improvement)
        self.CR_step = 0.05 # Step size for adapting CR (EI improvement)

        self.gp = None
        self.best_y = float('inf')
        self.best_x = None

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
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

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0 # avoid nan values
        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using Differential Evolution
        # return array of shape (batch_size, n_dims)

        # Initialize population
        population = self._sample_points(self.pop_size)

        n_success = 0
        # DE optimization loop
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
                ei_trial = self._acquisition_function(x_trial.reshape(1, -1))[0, 0]
                ei_current = self._acquisition_function(population[i].reshape(1, -1))[0, 0]

                if ei_trial > ei_current:
                    population[i] = x_trial
                    n_success += 1
                    # Adapt F and CR: Increase if improvement (EI improvement)
                    self.F = min(1.0, self.F + self.F_step)
                    self.CR = min(1.0, self.CR + self.CR_step)
                else:
                    # Adapt F and CR: Decrease if no improvement (EI improvement)
                    self.F = max(0.1, self.F - self.F_step)
                    self.CR = max(0.1, self.CR - self.CR_step)

        # Update F and CR adaptively (success rate)
        self.success_rate = 0.9 * self.success_rate + 0.1 * (n_success / self.pop_size)
        self.F = np.clip(self.F + self.learning_rate * (self.success_rate - 0.5), 0.1, 0.9)
        self.CR = np.clip(self.CR + self.learning_rate * (self.success_rate - 0.5), 0.1, 0.9)

        # Return the best points from the population
        ei_values = self._acquisition_function(population)

        # Local search
        best_idx = np.argmax(ei_values)
        best_x = population[best_idx]

        def obj_func(x):
            return -self._acquisition_function(x.reshape(1, -1))[0,0]

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

            # Select next points by acquisition function
            next_X = self._select_next_points(1)

            # Evaluate the points
            next_y = self._evaluate_points(func, next_X)

            # Update evaluated points
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x

```
The algorithm ASRDE_LS_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1643 with standard deviation 0.0990.

took 1604.52 seconds to run.

Combine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it. If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.




Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

