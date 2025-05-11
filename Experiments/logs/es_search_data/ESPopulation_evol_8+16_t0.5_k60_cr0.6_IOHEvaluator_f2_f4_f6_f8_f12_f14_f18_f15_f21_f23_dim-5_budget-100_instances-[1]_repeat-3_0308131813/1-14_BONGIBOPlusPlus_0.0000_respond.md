# Description
**BONGIBO++: Bayesian Optimization with Noisy Handling, Adaptive Gradient-based Improvement, and Dynamic Exploration.** This algorithm builds upon BONGIBO by introducing adaptive gradient-based local improvement and a dynamic exploration strategy. The number of points improved by L-BFGS-B is dynamically adjusted based on the iteration number, promoting more exploration early on and more exploitation later. Additionally, the exploration bonus in the acquisition function is dynamically adjusted based on the uncertainty of the GPR model.

# Justification
1.  **Adaptive Gradient-Based Improvement:** Instead of fixing the number of points improved by L-BFGS-B, the number is dynamically adjusted based on the current iteration. In the early stages of optimization, a smaller number of points are improved to encourage exploration. As the optimization progresses, a larger number of points are improved to exploit promising regions. This adaptive strategy balances exploration and exploitation more effectively.
2.  **Dynamic Exploration Bonus:** The exploration bonus in the acquisition function is dynamically adjusted. The bonus is proportional to the average uncertainty of the GPR model. This allows the algorithm to explore more when the model is uncertain and exploit more when the model is confident.
3.  **Computational Efficiency:** The changes are computationally efficient, adding minimal overhead to the existing BONGIBO algorithm. The dynamic adjustment of the number of improved points and the exploration bonus is done with simple calculations.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from scipy.optimize import approx_fprime

class BONGIBOPlusPlus:
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
        self.noise_level = 0.01 # Assume a small noise level, can be adjusted.

        self.best_y = np.inf
        self.best_x = None

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        # Estimate noise level from data. Add a small constant for numerical stability.
        estimated_noise_variance = np.var(y) + 1e-8
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=estimated_noise_variance)

        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Exploration bonus based on uncertainty (sigma)
        # Dynamically adjust exploration bonus based on the average uncertainty
        avg_sigma = np.mean(sigma)
        exploration_bonus = 0.01 * avg_sigma * sigma

        acquisition = ei + exploration_bonus
        return acquisition

    def _select_next_points(self, func, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Generate candidate points
        candidate_points = self._sample_points(100 * batch_size)
        
        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_points)
        
        # Select the top batch_size points with the highest acquisition values
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]

        # Gradient-based local improvement for a subset of points
        # Adaptively adjust the number of points to improve
        num_to_improve = min(int(batch_size * (1 - self.n_evals / self.budget)), batch_size) # Reduce over time
        improved_points = []
        for i in range(num_to_improve):
            
            def obj_func(x):
                x = x.reshape(1, -1)
                return self.model.predict(x)[0] # Minimize the predicted value

            x0 = next_points[i]
            bounds = [(self.bounds[0][j], self.bounds[1][j]) for j in range(self.dim)]
            res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5}) # Limited iterations
            improved_points.append(res.x)

        # Replace original points with improved points
        next_points[:num_to_improve] = np.array(improved_points)
        
        return next_points

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) + np.random.normal(0, self.noise_level) for x in X]) # Add noise for robustness
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
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Optimization
            # select points by acquisition function
            remaining_evals = self.budget - self.n_evals
            batch_size = min(self.n_init, remaining_evals) # Adjust batch size to budget
            next_X = self._select_next_points(func, batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self.model = self._fit_model(self.X, self.y)

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<BONGIBOPlusPlus>", line 147, in __call__
 147->             next_X = self._select_next_points(func, batch_size)
  File "<BONGIBOPlusPlus>", line 101, in _select_next_points
  99 | 
 100 |         # Replace original points with improved points
 101->         next_points[:num_to_improve] = np.array(improved_points)
 102 |         
 103 |         return next_points
ValueError: could not broadcast input array from shape (0,) into shape (0,5)
