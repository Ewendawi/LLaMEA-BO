# Description
Adaptive DE with Uncertainty-Aware Batch Size Bayesian Optimization (ADUABS_BO): This algorithm combines the strengths of Adaptive_DE_BO and DE_ABS_BO. It uses Differential Evolution (DE) to optimize the Expected Improvement (EI) acquisition function, with adaptive mutation factor (F) and crossover rate (CR) based on the success of previous DE iterations. It also incorporates an adaptive batch size strategy based on the uncertainty of the Gaussian Process (GP) surrogate model. The adaptation of F and CR is refined to use a success-history based approach, and the batch size adaptation is modified to be more robust.

# Justification
This algorithm builds upon the strengths of Adaptive_DE_BO and DE_ABS_BO.
- Adaptive Mutation and Crossover: The adaptive mutation factor (F) and crossover rate (CR) in DE are crucial for balancing exploration and exploitation. Instead of linearly decreasing/increasing F and CR, a success-history based adaptation is used, which is more adaptive to the landscape of the acquisition function.
- Adaptive Batch Size: The adaptive batch size strategy from DE_ABS_BO is incorporated to balance exploration and exploitation. The batch size is adjusted based on the GP's uncertainty, allowing for more exploration when the GP is uncertain and more exploitation when the GP is confident. The batch size adjustment is modified to be more robust and prevent it from becoming too large or too small.
- Computational Efficiency: DE is a relatively efficient optimization algorithm, and the adaptive batch size strategy helps to reduce the number of function evaluations, improving computational efficiency.
- Sobol Sampling: Using Sobol sampling for initial sampling and DE population initialization provides better space-filling properties than Latin Hypercube sampling.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ADUABS_BO:
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
        self.de_iters = 20 # Number of DE iterations
        self.success_history_F = []
        self.success_history_CR = []

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

        # DE optimization loop
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
                ei_trial = self._acquisition_function(x_trial.reshape(1, -1))[0, 0]
                ei_current = self._acquisition_function(population[i].reshape(1, -1))[0, 0]

                if ei_trial > ei_current:
                    population[i] = x_trial
                    self.success_history_F.append(self.F)
                    self.success_history_CR.append(self.CR)
                    if len(self.success_history_F) > 10:
                        self.success_history_F.pop(0)
                        self.success_history_CR.pop(0)
                
                if self.success_history_F:
                    self.F = np.mean(self.success_history_F)
                    self.CR = np.mean(self.success_history_CR)
                
                self.F = np.clip(self.F, 0.1, 0.9)
                self.CR = np.clip(self.CR, 0.1, 0.9)

        # Return the best point from the population
        ei_values = self._acquisition_function(population)
        indices = np.argsort(ei_values.flatten())[::-1]
        next_points = population[indices[:batch_size]]
        return next_points

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
            self.batch_size = min(min(self.batch_size, self.budget - self.n_evals), 5) # Ensure not exceeding budget, and limit to 5

            # Select next points by acquisition function
            next_X = self._select_next_points(self.batch_size)

            # Evaluate the points
            next_y = self._evaluate_points(func, next_X)

            # Update evaluated points
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ADUABS_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1622 with standard deviation 0.1002.

took 284.91 seconds to run.