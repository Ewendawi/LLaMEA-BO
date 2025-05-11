# Description
DE-BO with Improved Adaptive Parameter Control and Acquisition Ensemble: This algorithm refines the DE_BO_Adaptive by enhancing the adaptation mechanism for the DE parameters (F and CR) using a more responsive update rule based on the relative EI improvement. It also introduces an acquisition ensemble by combining EI with the Lower Confidence Bound (LCB) to balance exploration and exploitation better. The local search is retained to refine promising solutions.

# Justification
The adaptive parameter control in the original DE_BO_Adaptive uses a simple success rate to update F and CR. This can be slow to respond to changes in the optimization landscape. By using the relative improvement in EI, the adaptation becomes more sensitive to the actual gains achieved by different parameter settings. The ensemble acquisition function, combining EI and LCB, provides a more robust exploration-exploitation trade-off. EI focuses on areas with high potential improvement, while LCB encourages exploration in regions with high uncertainty. The combination helps to prevent premature convergence and improves the algorithm's ability to find the global optimum. Using a weighted sum allows to control the trade-off between exploration and exploitation.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class Improved_DE_BO:
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
        self.F = 0.8 # Mutation factor for DE
        self.CR = 0.7 # Crossover rate for DE
        self.learning_rate = 0.1
        self.ei_weight = 0.5 # Weight for EI in the ensemble acquisition function
        self.lcb_weight = 0.5 # Weight for LCB in the ensemble acquisition function

        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.batch_size = 3

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
        # Implement acquisition function using EI and LCB
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement
        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0 # avoid nan values

        # Lower Confidence Bound
        lcb = mu - 2 * sigma

        # Ensemble acquisition function
        return self.ei_weight * ei + self.lcb_weight * lcb

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using Differential Evolution
        # return array of shape (batch_size, n_dims)

        # Initialize population
        population = self._sample_points(self.pop_size)
        
        # DE optimization loop
        ei_current_values = self._acquisition_function(population)
        n_success = 0
        sum_rel_improvement = 0.0

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
                ei_current = ei_current_values[i, 0]
                
                if ei_trial > ei_current:
                    population[i] = x_trial
                    ei_current_values[i, 0] = ei_trial
                    n_success += 1
                    rel_improvement = (ei_trial - ei_current) / abs(ei_current) if abs(ei_current) > 1e-9 else (ei_trial - ei_current)
                    sum_rel_improvement += rel_improvement
        
        #Update F and CR adaptively based on relative improvement
        if n_success > 0:
            avg_rel_improvement = sum_rel_improvement / n_success
            self.F = np.clip(self.F + self.learning_rate * (avg_rel_improvement - 0.1), 0.1, 0.9)
            self.CR = np.clip(self.CR + self.learning_rate * (avg_rel_improvement - 0.1), 0.1, 0.9)
        else:
            # If no success, reduce exploration
            self.F = np.clip(self.F - self.learning_rate * 0.1, 0.1, 0.9)
            self.CR = np.clip(self.CR - self.learning_rate * 0.1, 0.1, 0.9)



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
            next_X = self._select_next_points(self.batch_size)

            # Evaluate the points
            next_y = self._evaluate_points(func, next_X)
            
            # Update evaluated points
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm Improved_DE_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1418 with standard deviation 0.1010.

took 1227.25 seconds to run.