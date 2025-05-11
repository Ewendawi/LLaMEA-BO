# Description
**ADRBF_BO_v2**: Adaptive DE with RBF Bandwidth Bayesian Optimization, enhanced with dynamic population size adjustment for DE and a more aggressive adaptation of DE parameters. This version builds upon ADRBF_BO by dynamically adjusting the population size of the Differential Evolution (DE) optimizer based on the optimization progress. A larger population size is used in the initial stages for broader exploration, which gradually decreases as the optimization progresses to focus on exploitation. The adaptation of the DE parameters (mutation factor F and crossover rate CR) is also made more aggressive to accelerate convergence.

# Justification
The key components of the algorithm are:

1.  **Gaussian Process Regression (GPR):** GPR is used as a surrogate model to approximate the objective function. It provides a probabilistic estimate of the function value at any given point, which is crucial for balancing exploration and exploitation.
2.  **Expected Improvement (EI) Acquisition Function:** EI is used to guide the search for the next point to evaluate. It quantifies the expected improvement over the current best value, considering both the predicted function value and the uncertainty.
3.  **Differential Evolution (DE):** DE is used to optimize the EI acquisition function. DE is a population-based optimization algorithm that is well-suited for non-convex and multi-modal functions.
4.  **Adaptive RBF Kernel Bandwidth:** The bandwidth of the RBF kernel is dynamically tuned using the median heuristic. This allows the algorithm to adapt to the local characteristics of the objective function.
5.  **Dynamic Population Size for DE:** Adjusting the population size dynamically allows for a better exploration-exploitation trade-off. A larger population size at the beginning helps to explore the search space more thoroughly, while a smaller population size at the end helps to fine-tune the solution.
6.  **Aggressive DE Parameter Adaptation:** Increasing the adaptation rate of F and CR helps DE to converge faster.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ADRBF_BO_v2:
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
        self.pop_size = int(1.5 * dim) # Initial Population size for DE, scaled by dimension
        self.F = 0.8 # Mutation factor for DE
        self.CR = 0.7 # Crossover rate for DE
        self.bandwidth = 1.0 # Initial RBF bandwidth
        self.bandwidth_update_interval = 5 * dim # Update bandwidth every this many evaluations
        self.de_iter = 20 # DE iterations

        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.de_success_rate = 0.0
        self.de_success_history = []
        self.exploration_exploitation_tradeoff = 0.5 # 0.0 for pure exploitation, 1.0 for pure exploration

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
        kernel = C(1.0, (1e-3, 1e3)) * RBF(self.bandwidth, (1e-3, 1e3))
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

        #Adjust EI with exploration_exploitation_tradeoff
        ei = (1-self.exploration_exploitation_tradeoff) * ei + self.exploration_exploitation_tradeoff * sigma

        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using Differential Evolution
        # return array of shape (batch_size, n_dims)

        # Adjust population size based on remaining budget
        remaining_budget = self.budget - self.n_evals
        self.pop_size = max(5, int(1.5 * self.dim * (remaining_budget / self.budget))) # Ensure pop_size is at least 5

        # Initialize population
        population = self._sample_points(self.pop_size)
        ei_values = self._acquisition_function(population)
        best_idx = np.argmax(ei_values)
        best_ei = ei_values[best_idx]
        
        successful_mutations = 0

        # DE optimization loop
        for _ in range(self.de_iter):
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
                    successful_mutations += 1
            
            ei_values = self._acquisition_function(population)
            current_best_idx = np.argmax(ei_values)
            current_best_ei = ei_values[current_best_idx]
            if current_best_ei > best_ei:
                best_ei = current_best_ei
                best_idx = current_best_idx

        # Update DE parameters adaptively
        self.de_success_history.append(successful_mutations / (self.pop_size * self.de_iter))
        if len(self.de_success_history) > 5:
            self.de_success_history = self.de_success_history[-5:]
        self.de_success_rate = np.mean(self.de_success_history)

        if self.de_success_rate > 0.5:
            self.CR = min(1.0, self.CR + 0.2) # Increased adaptation rate
            self.F = max(0.1, self.F - 0.1)
        else:
            self.F = min(1.0, self.F + 0.2) # Increased adaptation rate
            self.CR = max(0.1, self.CR - 0.2)


        # Return the best point from the population
        next_point = population[best_idx]
        return next_point.reshape(1, -1)

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

    def _update_bandwidth(self):
        # Update the RBF kernel bandwidth using the median heuristic
        distances = np.linalg.norm(self.X[:, None, :] - self.X[None, :, :], axis=2)
        distances = distances[np.triu_indices_from(distances, k=1)]
        if len(distances) > 0:
            self.bandwidth = np.median(distances)

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

            # Update bandwidth periodically
            if self.n_evals % self.bandwidth_update_interval == 0:
                self._update_bandwidth()

            # Reduce DE iterations over time
            self.de_iter = max(5, int(20 * (self.budget - self.n_evals) / self.budget))

            # Update exploration-exploitation tradeoff
            self.exploration_exploitation_tradeoff = 0.1 + 0.8 * (self.budget - self.n_evals) / self.budget

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ADRBF_BO_v2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1570 with standard deviation 0.0982.

took 278.22 seconds to run.