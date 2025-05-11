# Description
**Bayesian Evolutionary Optimization (BEO):** This algorithm combines Bayesian Optimization with evolutionary strategies. It uses a Gaussian Process (GP) to model the objective function and an acquisition function based on Expected Improvement (EI). Instead of directly optimizing the acquisition function, it employs an evolutionary algorithm (specifically, a differential evolution strategy) to generate candidate points. This approach leverages the global search capabilities of evolutionary algorithms while still benefiting from the sample efficiency of Bayesian Optimization. The algorithm also incorporates a mechanism to adapt the mutation rate of the differential evolution based on the success rate of recent generations, further enhancing its adaptability.

# Justification
This algorithm is designed to be diverse from the previous ones in several ways:

1.  **Optimization Strategy:** Instead of directly optimizing the acquisition function (as in TRBO and GEBO) or using Thompson sampling (as in EHBBO), it employs a differential evolution strategy. This provides a different exploration-exploitation balance and can be more effective in complex search spaces.
2.  **Gradient-Free:** Unlike GEBO, this algorithm does not rely on gradient information, making it applicable to a wider range of black-box functions.
3.  **Adaptive Mutation Rate:** The differential evolution strategy includes an adaptive mutation rate, which helps the algorithm adjust its exploration behavior based on the observed performance.
4.  **Error Avoidance:** The previous `AdaptiveVarianceBO` had a `TypeError` due to the `sample_weight` argument in `GaussianProcessRegressor.fit()`. This algorithm avoids this error by not using the `sample_weight` argument.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution

class BayesianEvolutionaryBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim
        self.gp = None
        self.de_popsize = 15 # Population size for differential evolution
        self.de_mutation = 0.5 # Initial mutation rate for differential evolution
        self.de_crossover = 0.7 # Crossover rate for differential evolution
        self.mutation_adaptation_rate = 0.1 # Rate to adapt mutation based on success
        self.success_threshold = 0.2 # Threshold for considering a generation successful
        self.best_x = None
        self.best_y = np.inf

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None:
            return np.zeros((len(X), 1)) # Return zeros if the model is not fitted yet

        mu, sigma = self.gp.predict(X, return_std=True)
        # Expected Improvement acquisition function
        improvement = self.best_y - mu
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # Avoid division by zero
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a differential evolution strategy to optimize the acquisition function
        # return array of shape (batch_size, n_dims)

        def de_objective(x):
            # Objective function for differential evolution (negative acquisition function)
            return -self._acquisition_function(x.reshape(1, -1))[0, 0]

        # Perform differential evolution
        de_result = differential_evolution(
            func=de_objective,
            bounds=list(zip(self.bounds[0], self.bounds[1])),
            popsize=self.de_popsize,
            mutation=self.de_mutation,
            recombination=self.de_crossover,
            maxiter=5, # Reduce maxiter for computational efficiency
            tol=0.01,
            seed=None,
            strategy='rand1bin',
            init='latinhypercube'
        )
        
        # Select the best point from the differential evolution result
        next_point = de_result.x.reshape(1, -1)
        
        return next_point

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y
    
    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        # Do not change the function signature
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))
        
        # Update best observed solution
        best_index = np.argmin(self.y)
        if self.y[best_index, 0] < self.best_y:
            self.best_x = self.X[best_index]
            self.best_y = self.y[best_index, 0]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        while self.n_evals < self.budget:
            # Optimization

            # Fit GP model
            self.gp = self._fit_model(self.X, self.y)

            # Select points by acquisition function using differential evolution
            batch_size = min(1, self.budget - self.n_evals) # Batch size of 1 for evolutionary strategy
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Adapt mutation rate based on success
            if len(self.y) > self.de_popsize:
                recent_ys = self.y[-self.de_popsize:]
                best_recent_y = np.min(recent_ys)
                if best_recent_y < self.best_y + self.success_threshold:
                    self.de_mutation *= (1 + self.mutation_adaptation_rate)
                else:
                    self.de_mutation *= (1 - self.mutation_adaptation_rate)
                self.de_mutation = np.clip(self.de_mutation, 0.1, 2.0)

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<BayesianEvolutionaryBO>", line 133, in __call__
 133->             next_X = self._select_next_points(batch_size)
  File "<BayesianEvolutionaryBO>", line 72, in _select_next_points
  70 | 
  71 |         # Perform differential evolution
  72->         de_result = differential_evolution(
  73 |             func=de_objective,
  74 |             bounds=list(zip(self.bounds[0], self.bounds[1])),
ValueError: The mutation constant must be a float in U[0, 2), or specified as a tuple(min, max) where min < max and min, max are in U[0, 2).
