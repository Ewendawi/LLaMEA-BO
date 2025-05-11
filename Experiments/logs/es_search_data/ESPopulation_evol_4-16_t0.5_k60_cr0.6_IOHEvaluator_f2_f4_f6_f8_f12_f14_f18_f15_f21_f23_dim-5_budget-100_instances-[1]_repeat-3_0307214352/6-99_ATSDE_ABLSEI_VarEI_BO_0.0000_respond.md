# Description
**ATSDE_ABLSEI_VarEI_BO: Adaptive Temperature and Success-Rate based DE-BO with Adaptive Batch size, Local Search with EI, EI acquisition function, and Variance-Aware EI.** This algorithm builds upon ATSDE_ABLSEI_BO by incorporating the variance of the Expected Improvement (EI) into both the local search objective and the overall point selection strategy. Specifically, the local search objective is modified to consider not only the EI but also the variance of the EI, encouraging exploration in regions where the EI is high but also uncertain. The point selection strategy is also enhanced by using EI combined with the variance of EI. This allows for a more informed decision on whether to perform local search, favoring regions where the potential improvement is not only high but also more reliable (lower variance). Additionally, the local search is enhanced by using multiple restarts to escape local optima.

# Justification
The key improvements focus on refining the local search and point selection by incorporating the variance of the EI.

*   **Variance-Aware Local Search:** The original algorithm used only the EI as the objective function for local search. By incorporating the variance of EI, the local search is guided towards regions where the GP model is more uncertain, promoting exploration and potentially escaping local optima. The objective function is modified to balance EI with the inverse of the EI variance.

*   **Multiple Local Search Restarts:** To further enhance the local search's ability to escape local optima, multiple restarts are introduced. This involves running the local search multiple times from different starting points within the defined search range.

*   **Variance-Aware EI:** The EI acquisition function is modified to incorporate the variance of EI. This encourages exploration in regions where the GP model is uncertain, promoting exploration and potentially escaping local optima.

These changes aim to improve the algorithm's ability to balance exploration and exploitation, leading to better performance on the BBOB test suite.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ATSDE_ABLSEI_VarEI_BO:
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
        self.batch_size = 1

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

    def _acquisition_function(self, X, return_var=False):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        if return_var:
            # Approximate variance of EI (simplified)
            ei_var = (norm.pdf(Z) * (imp + sigma * Z))**2
            return ei, ei_var
        else:
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
                ei_trial, ei_var_trial = self._acquisition_function(x_trial.reshape(1, -1), return_var=True)
                ei_current, ei_var_current = self._acquisition_function(population[i].reshape(1, -1), return_var=True)

                # Combine EI and variance (exploration-exploitation balance)
                selection_criterion_trial = ei_trial - 0.1 * np.sqrt(ei_var_trial)  # Adjust weight as needed
                selection_criterion_current = ei_current - 0.1 * np.sqrt(ei_var_current)

                if selection_criterion_trial > selection_criterion_current:
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
        ei_values, ei_var_values = self._acquisition_function(population, return_var=True)

        # Local search
        best_idx = np.argmax(ei_values)
        best_x = population[best_idx]

        # Probabilistic Local Search
        mu, sigma = self.gp.predict(best_x.reshape(1, -1), return_std=True)
        self.ls_prob = 0.1 + 0.9 * np.exp(-sigma)  # Higher uncertainty -> higher LS probability

        if np.random.rand() < self.ls_prob:
            def obj_func(x):
                ei, ei_var = self._acquisition_function(x.reshape(1, -1), return_var=True)
                # Combine EI and variance for local search objective
                return -(ei - 0.1 * np.sqrt(ei_var)) # Adjust weight as needed

            # Adjust local search range based on GP uncertainty
            search_range = min(sigma[0] * 2, (self.bounds[1][0] - self.bounds[0][0]) / 2) # Limit search range
            
            # Multiple restarts for local search
            best_local_x = best_x
            best_local_obj = obj_func(best_x)
            for _ in range(3): # Number of restarts
                x0 = best_x + np.random.normal(0, search_range/2, size=self.dim) # Random starting point
                x0 = np.clip(x0, self.bounds[0], self.bounds[1])
                res = minimize(obj_func, x0, bounds=[(best_x[i] - search_range, best_x[i] + search_range) for i in range(self.dim)], method='L-BFGS-B')
                if res.success and res.fun < best_local_obj:
                    best_local_obj = res.fun
                    best_local_x = res.x

            next_point = best_local_x.reshape(1, -1)
            next_point = np.clip(next_point, self.bounds[0], self.bounds[1])
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

            # Adaptive batch size
            _, std = self.gp.predict(self.X, return_std=True)
            mean_std = np.mean(std)
            self.batch_size = max(1, int(self.dim / (1 + mean_std * 10))) # Adjust batch size based on uncertainty
            self.batch_size = min(self.batch_size, self.budget - self.n_evals) # Ensure not exceeding budget

            # Select next points by acquisition function
            next_X = np.vstack([self._select_next_points(1) for _ in range(self.batch_size)])

            # Evaluate the points
            next_y = self._evaluate_points(func, next_X)

            # Update evaluated points
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```
## Error
 TypeError: only size-1 arrays can be converted to Python scalars

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<ATSDE_ABLSEI_VarEI_BO>", line 209, in __call__
 209->             next_X = np.vstack([self._select_next_points(1) for _ in range(self.batch_size)])
  File "<ATSDE_ABLSEI_VarEI_BO>", line 209, in <listcomp>
 209->             next_X = np.vstack([self._select_next_points(1) for _ in range(self.batch_size)])
  File "<ATSDE_ABLSEI_VarEI_BO>", line 95, in _select_next_points
  93 |                 for j in range(self.dim):
  94 |                     if np.random.rand() < self.CR:
  95->                         x_trial[j] = x_mutated[j]
  96 | 
  97 |                 # Selection
ValueError: setting an array element with a sequence.
