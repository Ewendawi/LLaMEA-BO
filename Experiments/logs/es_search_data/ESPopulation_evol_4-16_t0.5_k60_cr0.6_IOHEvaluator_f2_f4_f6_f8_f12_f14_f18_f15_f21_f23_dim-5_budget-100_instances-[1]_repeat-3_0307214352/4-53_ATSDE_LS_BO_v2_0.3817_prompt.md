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




The selected solution to update is:
**Adaptive Temperature and Success-Rate based DE-BO with Local Search (ATSDE_LS_BO):** This algorithm combines the adaptive temperature control from ADExBO with the adaptive success-rate based F and CR adaptation from ASRDE_LS_BO. It also incorporates a local search step using L-BFGS-B to refine the best solution. The key innovation is the synergistic combination of exploration-exploitation balancing via temperature annealing and success-rate based parameter adaptation, leading to a more robust and efficient optimization process. Additionally, the algorithm dynamically adjusts the local search probability based on the GP's uncertainty, similar to ASRDE_PLS_BO, to balance refinement and exploration.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ATSDE_LS_BO:
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
        self.temperature = 1.0 # Initial temperature for exploration
        self.temperature_decay = 0.95 # Decay rate for temperature
        self.ls_prob = 0.1 # Initial probability of local search

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
                x_mutated = x_r1 + self.F * (x_r2 - x_r3) * self.temperature
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
                    improvement = ei_trial - ei_current
                    self.F = min(1.0, self.F + self.F_step * (1 - improvement))
                    self.CR = min(1.0, self.CR + self.CR_step * (1 - improvement))
                else:
                    # Adapt F and CR: Decrease if no improvement (EI improvement)
                    self.F = max(0.1, self.F - self.F_step)
                    self.CR = max(0.1, self.CR - self.CR_step)

            # Anneal temperature
            self.temperature *= self.temperature_decay

        # Update F and CR adaptively (success rate)
        self.success_rate = 0.9 * self.success_rate + 0.1 * (n_success / self.pop_size)
        self.F = np.clip(self.F + self.learning_rate * (self.success_rate - 0.5), 0.1, 0.9)
        self.CR = np.clip(self.CR + self.learning_rate * (self.success_rate - 0.5), 0.1, 0.9)

        # Return the best points from the population
        ei_values = self._acquisition_function(population)

        # Local search
        best_idx = np.argmax(ei_values)
        best_x = population[best_idx]

        # Probabilistic Local Search
        mu, sigma = self.gp.predict(best_x.reshape(1, -1), return_std=True)
        self.ls_prob = 0.1 + 0.9 * np.exp(-sigma)  # Higher uncertainty -> higher LS probability

        if np.random.rand() < self.ls_prob:
            def obj_func(x):
                return -self._acquisition_function(x.reshape(1, -1))[0,0]

            res = minimize(obj_func, best_x, bounds=[(-5, 5)]*self.dim, method='L-BFGS-B')
            next_point = res.x.reshape(1, -1)
        else:
            next_point = best_x.reshape(1, -1)

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
The algorithm ATSDE_LS_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1706 with standard deviation 0.1047.

took 1539.81 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

