# Description
Adaptive DE with Uncertainty-Aware Batch Size and Dynamic F/CR Bayesian Optimization (ADUBO): This algorithm synergizes the strengths of Adaptive_DE_BO and DE_ABS_BO. It incorporates an adaptive batch size strategy based on the Gaussian Process's uncertainty, similar to DE_ABS_BO, to balance exploration and exploitation. Furthermore, it retains the adaptive mutation factor (F) and crossover rate (CR) from Adaptive_DE_BO, but modifies the adaptation mechanism. Instead of adapting F and CR at each DE iteration, the adaptation is performed at the end of each DE optimization loop based on the overall success of the loop. This algorithm also uses a more aggressive strategy for updating the evaluated points by selecting the best points from the batch rather than all points.

# Justification
1.  **Adaptive Batch Size:** The adaptive batch size, as implemented in DE_ABS_BO, allows the algorithm to dynamically adjust the number of points evaluated in each iteration. This is crucial for balancing exploration and exploitation. When the GP model has high uncertainty (high average standard deviation), a larger batch size is used to explore the search space more broadly. Conversely, when the GP model has low uncertainty, a smaller batch size is used to exploit the promising regions more effectively.
2.  **Adaptive Mutation and Crossover:** Adaptive_DE_BO's approach to adapting the mutation factor (F) and crossover rate (CR) is valuable. However, adapting at each DE iteration might be too sensitive to individual trial vector outcomes. By adapting at the end of each DE loop, the algorithm considers the overall success of the DE optimization in finding better EI values. This provides a more stable and robust adaptation strategy. The adaptation is based on the ratio of improved EI values to the total population size.
3.  **Aggressive Update of Evaluated Points:** Instead of adding all points from the batch to the evaluated points, we select only the best `batch_size // 2 + 1` points. This reduces the computational cost of fitting the GP model, especially when the batch size is large, and focuses on the most promising regions.
4.  **Sobol Sampling:** Sobol sampling is used for initial sampling and DE population initialization. Sobol sequences are quasi-random numbers that offer better space-filling properties than pseudo-random numbers, leading to improved exploration of the search space.
5.  **Computational Efficiency:** By combining adaptive batch size and a more aggressive update strategy, the algorithm aims to achieve a better balance between exploration, exploitation, and computational cost.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ADUBO:
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
        self.pop_size = 5 * dim # Population size for DE
        self.F = 0.8 # Initial mutation factor for DE
        self.CR = 0.7 # Initial crossover rate for DE
        self.F_step = 0.05 # Step size for adapting F
        self.CR_step = 0.05 # Step size for adapting CR
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
        initial_ei = self._acquisition_function(population)
        
        n_improved = 0

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
                    n_improved += 1

        # Adapt F and CR based on the success of the DE loop
        success_ratio = n_improved / self.pop_size
        if success_ratio > 0.2:  # Tune this threshold
            self.F = min(1.0, self.F + self.F_step)
            self.CR = min(1.0, self.CR + self.CR_step)
        else:
            self.F = max(0.1, self.F - self.F_step)
            self.CR = max(0.1, self.CR - self.CR_step)

        # Return the best points from the population
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
        
        # Select the best points from the batch to update the evaluated points
        n_best = max(1, self.batch_size // 2 + 1)
        indices = np.argsort(new_y.flatten())[:n_best]
        best_X = new_X[indices]
        best_y = new_y[indices]

        if self.X is None:
            self.X = best_X
            self.y = best_y
        else:
            self.X = np.vstack((self.X, best_X))
            self.y = np.vstack((self.y, best_y))

        # Update best seen value
        if np.min(best_y) < self.best_y:
            self.best_y = np.min(best_y)
            self.best_x = best_X[np.argmin(best_y)]

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
## Feedback
 The algorithm ADUBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1563 with standard deviation 0.1040.

took 442.16 seconds to run.