# Description
Adaptive DE with Exploration-Exploitation Balancing Bayesian Optimization (ADExBO): This algorithm refines the Adaptive_DE_BO by incorporating a dynamic adjustment of the exploration-exploitation trade-off within the Differential Evolution (DE) optimization of the Expected Improvement (EI) acquisition function. It introduces a temperature parameter that controls the intensity of exploration in DE. The temperature is annealed over time, starting with a high value to promote exploration and gradually decreasing to favor exploitation. Additionally, the adaptation of F and CR is modified to consider the magnitude of improvement, providing finer control over the adaptation process. A local search step using L-BFGS-B is added to refine the best solution found by DE.

# Justification
1.  **Dynamic Exploration-Exploitation Balance:** Introducing a temperature parameter allows for a more controlled shift from exploration to exploitation during the optimization process. Starting with a high temperature encourages DE to explore the search space more broadly, while gradually decreasing it focuses the search on promising regions.
2.  **Refined F and CR Adaptation:** Modifying the adaptation of the mutation factor (F) and crossover rate (CR) to consider the magnitude of improvement allows for finer control over the adaptation process. Larger improvements lead to smaller adjustments, preventing over-adaptation, while smaller improvements lead to larger adjustments to encourage further exploration.
3.  **Local Search:** Incorporating a local search step using L-BFGS-B helps to refine the best solution found by DE, potentially leading to further improvements in the EI value.
4.  **Computational Efficiency:** The core DE algorithm remains computationally efficient, and the addition of a local search step is performed only on the best solution, minimizing the computational overhead.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ADExBO:
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
        self.F_step = 0.05 # Step size for adapting F
        self.CR_step = 0.05 # Step size for adapting CR
        self.temperature = 1.0 # Initial temperature for exploration
        self.temperature_decay = 0.95 # Decay rate for temperature

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
                    # Adapt F and CR: Increase if improvement
                    improvement = ei_trial - ei_current
                    self.F = min(1.0, self.F + self.F_step * (1 - improvement))
                    self.CR = min(1.0, self.CR + self.CR_step * (1 - improvement))
                else:
                    # Adapt F and CR: Decrease if no improvement
                    self.F = max(0.1, self.F - self.F_step)
                    self.CR = max(0.1, self.CR - self.CR_step)

            # Anneal temperature
            self.temperature *= self.temperature_decay

        # Return the best point from the population
        ei_values = self._acquisition_function(population)
        best_idx = np.argmax(ei_values)
        next_point = population[best_idx]
        
        # Local search
        def obj_func(x):
            return -self._acquisition_function(x.reshape(1, -1))[0, 0]
        
        res = minimize(obj_func, next_point, bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)], method='L-BFGS-B')
        next_point = res.x

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
 The algorithm ADExBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1649 with standard deviation 0.0993.

took 1613.88 seconds to run.