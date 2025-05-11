# Description
**ATSDE_LS_BO_v2: Adaptive Temperature and Success-Rate based DE-BO with Enhanced Local Search and Acquisition Function Balancing.** This algorithm refines the ATSDE_LS_BO algorithm by introducing improvements to the local search strategy and the acquisition function. The local search is enhanced by adaptively adjusting the search range based on the GP's uncertainty and the distance to the nearest evaluated point. This allows for more focused exploration in promising regions. Additionally, the acquisition function is modified to balance exploration and exploitation by incorporating a dynamic weighting factor that adjusts the influence of the Expected Improvement (EI) and the Lower Confidence Bound (LCB). This weighting factor is based on the success rate of the DE iterations, promoting exploration with LCB when the success rate is low and exploitation with EI when the success rate is high. The temperature decay is also made adaptive based on the success rate.

# Justification
The key improvements are:
1.  **Enhanced Local Search:** The local search range is dynamically adjusted based on the GP's uncertainty (sigma) and the distance to the nearest evaluated point. This allows the local search to focus on promising regions with high uncertainty and potential for improvement.
2.  **Acquisition Function Balancing:** The acquisition function is modified to balance exploration and exploitation by incorporating a dynamic weighting factor that adjusts the influence of the Expected Improvement (EI) and the Lower Confidence Bound (LCB). This allows the algorithm to adapt to the landscape of the objective function and switch between exploration and exploitation as needed.
3.  **Adaptive Temperature Decay:** The temperature decay is made adaptive based on the success rate, allowing for faster cooling when DE is successful and slower cooling when DE is struggling.

These changes are designed to improve the algorithm's ability to find the global optimum by balancing exploration and exploitation more effectively. The adaptive local search ensures that the algorithm focuses on promising regions, while the acquisition function balancing allows it to switch between exploration and exploitation as needed. The adaptive temperature decay further enhances the exploration-exploitation balance.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors

class ATSDE_LS_BO_v2:
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
        self.ei_lcb_ratio = 0.5 # Ratio between EI and LCB

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

        # Lower Confidence Bound
        lcb = mu - 2 * sigma

        # Adaptive EI/LCB balancing
        self.ei_lcb_ratio = 0.9 * self.ei_lcb_ratio + 0.1 * (self.success_rate > 0.5)

        # Weighted acquisition function
        acq = self.ei_lcb_ratio * ei + (1 - self.ei_lcb_ratio) * lcb

        return acq

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
            self.temperature_decay = 0.9 + 0.09 * (self.success_rate > 0.5)

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

        # Adaptive Local Search Range
        if self.X is not None:
            neighbors = NearestNeighbors(n_neighbors=1)
            neighbors.fit(self.X)
            distance, _ = neighbors.kneighbors(best_x.reshape(1, -1))
            ls_range = min(1.0, 0.1 * sigma + distance[0][0])  # Smaller range if close to existing points or low uncertainty
        else:
            ls_range = 1.0

        self.ls_prob = 0.1 + 0.9 * np.exp(-sigma)  # Higher uncertainty -> higher LS probability

        if np.random.rand() < self.ls_prob:
            def obj_func(x):
                return -self._acquisition_function(x.reshape(1, -1))[0,0]

            res = minimize(obj_func, best_x, bounds=[(max(-5, best_x[i] - ls_range), min(5, best_x[i] + ls_range)) for i in range(self.dim)], method='L-BFGS-B')
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
## Feedback
 The algorithm ATSDE_LS_BO_v2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1589 with standard deviation 0.1044.

took 1553.81 seconds to run.