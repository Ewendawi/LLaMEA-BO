# Description
**Adaptive Local Search Hybrid Bayesian Optimization (ALSHBO):** This algorithm refines the Efficient Hybrid Bayesian Optimization (EHBBO) by introducing an adaptive local search strategy. The local search radius is dynamically adjusted based on the optimization progress and the characteristics of the function landscape. Specifically, if the local search consistently yields improvements, the radius is increased to explore a wider region. Conversely, if the local search stagnates, the radius is decreased to focus on finer details. Additionally, the algorithm incorporates a momentum term in the local search, where the search direction is influenced by the previous successful search direction. This helps to accelerate convergence in regions with consistent gradients.

# Justification
The key improvements are:

1.  **Adaptive Local Search Radius:** The original EHBBO used a fixed local search radius. Adapting this radius allows the algorithm to adjust its search intensity based on the observed function landscape. A larger radius promotes exploration when improvements are consistent, while a smaller radius encourages exploitation when the search stagnates.
2.  **Momentum in Local Search:** Introducing momentum helps the local search to escape local optima and accelerate convergence in regions with consistent gradients. The search direction is influenced by the previous successful direction, which can lead to faster progress.
3.  **GP Kernel Tuning:** While the previous implementation uses fixed kernel parameters, tuning them can significantly improve the GP model's accuracy. The length_scale parameter of the RBF kernel is optimized during GP fitting, allowing the model to adapt to the function's characteristic length scales.

These changes aim to improve the algorithm's ability to balance exploration and exploitation, adapt to different function landscapes, and accelerate convergence, leading to better overall performance.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

class ALSHBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim # Number of initial points
        self.gp = None
        self.update_interval = 5 # Update GP every 5 iterations
        self.local_search_radius = 0.5 # Initial radius for local search
        self.best_x = None
        self.best_y = np.inf
        self.local_search_success = True
        self.momentum = 0.1 # Momentum for local search direction
        self.previous_direction = None

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
        # Tune the length_scale parameter
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None:
            return np.zeros((len(X), 1)) # Return zeros if the model is not fitted yet

        mu, sigma = self.gp.predict(X, return_std=True)
        # Thompson Sampling
        xi = np.random.normal(mu, sigma)
        return xi.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Optimize the acquisition function to find the next points
        x_tries = self._sample_points(batch_size * 10) # Generate more candidates
        acq_values = self._acquisition_function(x_tries)
        
        # Select the top batch_size points based on the acquisition function values
        indices = np.argsort(acq_values.flatten())[::-1][:batch_size]
        return x_tries[indices]

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
            self.local_search_success = True # Reset success flag
        else:
            self.local_search_success = False
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        iteration = 0
        while self.n_evals < self.budget:
            # Optimization
            iteration += 1
            
            # Update GP model periodically
            if iteration % self.update_interval == 0:
                self.gp = self._fit_model(self.X, self.y)

            # Select points by acquisition function
            batch_size = min(10, self.budget - self.n_evals) # Adjust batch size to budget
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Local search around the best solution
            if self.best_x is not None and self.n_evals < self.budget:
                # Adaptive radius adjustment
                if self.local_search_success:
                    self.local_search_radius *= 1.1 # Increase radius
                else:
                    self.local_search_radius *= 0.9 # Decrease radius
                self.local_search_radius = np.clip(self.local_search_radius, 0.01, 1.0) # Keep radius within bounds

                # Momentum in local search direction
                if self.previous_direction is None:
                    direction = self._sample_points(1).flatten() - self.best_x
                else:
                    direction = (1 - self.momentum) * (self._sample_points(1).flatten() - self.best_x) + self.momentum * self.previous_direction
                
                direction /= np.linalg.norm(direction) # Normalize direction
                self.previous_direction = direction

                local_X = self.best_x + direction * self.local_search_radius
                local_X = np.clip(local_X, self.bounds[0], self.bounds[1])
                local_y = self._evaluate_points(func, local_X.reshape(1, -1))
                self._update_eval_points(local_X.reshape(1, -1), local_y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ALSHBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1531 with standard deviation 0.0999.

took 1.96 seconds to run.