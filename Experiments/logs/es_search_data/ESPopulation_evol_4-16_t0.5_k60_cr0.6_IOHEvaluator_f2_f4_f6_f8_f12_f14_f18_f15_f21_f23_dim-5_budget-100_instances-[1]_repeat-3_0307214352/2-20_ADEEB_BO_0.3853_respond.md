# Description
Adaptive DE with Exploration-Exploitation Balancing Bayesian Optimization (ADEEB-BO): This algorithm enhances the Adaptive_DE_BO by introducing a mechanism to balance exploration and exploitation in the Differential Evolution (DE) optimization of the Expected Improvement (EI) acquisition function. The key idea is to dynamically adjust the mutation factor (F) and crossover rate (CR) of DE not only based on the immediate success of a trial vector but also on the overall uncertainty of the Gaussian Process (GP) surrogate model. High GP uncertainty encourages exploration (higher F and CR), while low uncertainty encourages exploitation (lower F and CR). Additionally, a probability of random restart is introduced in the DE optimization to avoid premature convergence.

# Justification
1.  **Exploration-Exploitation Balancing:** The original Adaptive_DE_BO adjusts F and CR solely based on whether a trial vector improves the EI. This can lead to getting stuck in local optima. By incorporating GP uncertainty (estimated by the predicted standard deviation), the algorithm can better balance exploration and exploitation. When the GP is uncertain, the algorithm increases F and CR to explore more broadly. When the GP is confident, it decreases F and CR to exploit the promising regions.
2.  **Uncertainty-Aware Adaptation:** The adaptation of F and CR is now dependent on the predicted standard deviation (uncertainty) from the GP model. This allows the DE to adapt its search behavior based on the quality of the surrogate model.
3.  **Random Restart:** Adding a small probability of random restart within the DE optimization helps to escape local optima and explore diverse regions of the acquisition function landscape, especially in complex or multimodal problems.
4. **Computational Efficiency:** The changes maintain the computational efficiency of the original Adaptive_DE_BO. The additional calculations for GP uncertainty are relatively inexpensive compared to function evaluations.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ADEEB_BO:
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
        self.random_restart_prob = 0.05 # Probability of random restart in DE

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
        return ei, sigma

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using Differential Evolution
        # return array of shape (batch_size, n_dims)

        # Initialize population
        population = self._sample_points(self.pop_size)
        
        # DE optimization loop
        for _ in range(20):
            for i in range(self.pop_size):
                # Random Restart
                if np.random.rand() < self.random_restart_prob:
                    population[i] = self._sample_points(1)[0]
                    continue

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
                ei_trial, sigma_trial = self._acquisition_function(x_trial.reshape(1, -1))
                ei_trial = ei_trial[0, 0]
                sigma_trial = sigma_trial[0, 0]

                ei_current, sigma_current = self._acquisition_function(population[i].reshape(1, -1))
                ei_current = ei_current[0, 0]
                sigma_current = sigma_current[0, 0]
                
                if ei_trial > ei_current:
                    population[i] = x_trial
                    # Adapt F and CR: Increase if improvement, and GP is uncertain
                    self.F = min(1.0, self.F + self.F_step * (sigma_trial / (sigma_trial + sigma_current)))
                    self.CR = min(1.0, self.CR + self.CR_step * (sigma_trial / (sigma_trial + sigma_current)))
                else:
                    # Adapt F and CR: Decrease if no improvement, and GP is confident
                    self.F = max(0.1, self.F - self.F_step * (sigma_current / (sigma_trial + sigma_current)))
                    self.CR = max(0.1, self.CR - self.CR_step * (sigma_current / (sigma_trial + sigma_current)))

        # Return the best point from the population
        ei_values, _ = self._acquisition_function(population)
        next_point = population[np.argmax(ei_values)]
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
 The algorithm ADEEB_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1615 with standard deviation 0.1005.

took 1460.06 seconds to run.