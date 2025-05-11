# Description
DE-ABS-BO with Improved DE Parameter Adaptation and Acquisition Smoothing. This enhanced version of DE-ABS-BO focuses on refining the Differential Evolution (DE) parameter adaptation and introducing acquisition function smoothing to improve exploration and exploitation. The DE mutation factor (F) and crossover rate (CR) are adapted using a success-history based adaptation (SHADE) approach. Additionally, the acquisition function is smoothed by averaging EI values over a small neighborhood to encourage broader exploration and avoid premature convergence to local optima.

# Justification
1.  **SHADE-inspired DE Parameter Adaptation:** The original DE-ABS-BO uses a simple linear adaptation of F and CR. SHADE-inspired adaptation maintains a memory of successful F and CR values from previous generations and samples from this memory to set the parameters for the current generation. This approach has been shown to be more effective in adapting DE parameters to the specific characteristics of the optimization problem.
2.  **Acquisition Function Smoothing:** The Expected Improvement (EI) acquisition function can sometimes be noisy, leading DE to converge prematurely to local optima. Smoothing the acquisition function by averaging EI values over a small neighborhood can help to mitigate this issue and encourage broader exploration of the search space. This is implemented by evaluating EI at multiple points around a candidate and averaging the results.
3.  **Computational Efficiency:** The changes are designed to be computationally efficient, adding minimal overhead to the existing DE-ABS-BO algorithm. The SHADE-inspired adaptation has a small memory footprint and the acquisition function smoothing is implemented using a small number of additional EI evaluations.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class DE_ABS_BO_Improved:
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
        self.F = 0.5 # Initial mutation factor for DE
        self.CR = 0.5 # Initial crossover rate for DE
        self.de_iters = 20 # Number of DE iterations

        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.batch_size = 1

        # SHADE-inspired adaptation
        self.memory_F = np.full(self.pop_size, self.F)
        self.memory_CR = np.full(self.pop_size, self.CR)
        self.archive_F = []
        self.archive_CR = []

        self.memory_size = 5 # Size of the memory for F and CR

        # Acquisition smoothing parameter
        self.smoothing_neighborhood_size = 0.1

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

    def _smoothed_acquisition_function(self, x):
        # Smooth the acquisition function by averaging over a neighborhood
        num_samples = 5  # Number of samples in the neighborhood
        neighborhood = np.random.uniform(
            low=np.maximum(self.bounds[0], x - self.smoothing_neighborhood_size),
            high=np.minimum(self.bounds[1], x + self.smoothing_neighborhood_size),
            size=(num_samples, self.dim)
        )
        ei_values = self._acquisition_function(neighborhood)
        return np.mean(ei_values)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using Differential Evolution
        # return array of shape (batch_size, n_dims)

        # Initialize population
        population = self._sample_points(self.pop_size)

        # DE optimization loop
        for iter in range(self.de_iters):
            new_population = np.copy(population)
            successful_F = []
            successful_CR = []

            for i in range(self.pop_size):
                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs]

                # Sample F and CR from memory
                F = self.memory_F[i]
                CR = self.memory_CR[i]

                x_mutated = x_r1 + F * (x_r2 - x_r3)
                x_mutated = np.clip(x_mutated, self.bounds[0], self.bounds[1])

                # Crossover
                x_trial = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < CR:
                        x_trial[j] = x_mutated[j]

                # Selection
                ei_trial = self._smoothed_acquisition_function(x_trial)
                ei_current = self._smoothed_acquisition_function(population[i])

                if ei_trial > ei_current:
                    new_population[i] = x_trial
                    successful_F.append(F)
                    successful_CR.append(CR)

            # Update population
            population = new_population

            # Update memory of F and CR
            if successful_F:
                self.archive_F.extend(successful_F)
                self.archive_CR.extend(successful_CR)
                if len(self.archive_F) > self.memory_size * self.pop_size:
                    self.archive_F = self.archive_F[-self.memory_size * self.pop_size:]
                    self.archive_CR = self.archive_CR[-self.memory_size * self.pop_size:]

                # Update memory_F and memory_CR
                for i in range(self.pop_size):
                    if self.archive_F:
                        self.memory_F[i] = np.random.choice(self.archive_F)
                        self.memory_CR[i] = np.random.choice(self.archive_CR)
                    self.memory_F[i] = np.clip(self.memory_F[i], 0, 1)
                    self.memory_CR[i] = np.clip(self.memory_CR[i], 0, 1)

        # Return the best point from the population
        ei_values = np.array([self._smoothed_acquisition_function(x) for x in population])
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
 The algorithm DE_ABS_BO_Improved got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1594 with standard deviation 0.1004.

took 368.66 seconds to run.