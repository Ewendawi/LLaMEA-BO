# Description
**GradientEnhancedBO (GEBO):** This algorithm leverages gradient information, estimated using finite differences, to enhance the Gaussian Process surrogate model and guide the search. It uses Expected Improvement (EI) as the acquisition function, modified to incorporate gradient information. A simple local search is performed around the best point found so far, using gradient information to accelerate convergence.

# Justification
The EHBBO algorithm uses Thompson Sampling and k-means clustering for batch selection. To create a diverse algorithm, GEBO uses Expected Improvement (EI) as the acquisition function, which is a more standard approach. More importantly, GEBO incorporates gradient information, which EHBBO does not. Gradient information can significantly improve the efficiency of the optimization, especially in smooth or locally convex regions. The local search step further exploits the gradient information to refine the solution. Finite differences are used to estimate the gradient, which is computationally efficient. The initial sampling is reduced to favor exploitation.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

class GradientEnhancedBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(5*dim, self.budget//10)

        # Do not add any other arguments without a default value
        self.gp = None
        self.best_x = None
        self.best_y = float('inf')

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
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function: Expected Improvement
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            mu, sigma = self.gp.predict(X, return_std=True)
            sigma = np.clip(sigma, 1e-9, np.inf)
            imp = self.best_y - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Optimize the acquisition function using a multi-start approach
        best_points = []
        for _ in range(batch_size):
            x0 = self._sample_points(1)  # Start from a random point
            
            def obj(x):
                return -self._acquisition_function(x.reshape(1, -1))[0, 0]  # Negative EI for minimization

            res = minimize(obj, x0, bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)], method='L-BFGS-B')
            best_points.append(res.x)
        
        return np.array(best_points)

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
        
        # Update best seen value
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]

    def _estimate_gradient(self, func, x, h=1e-5):
        # Estimate the gradient of func at x using finite differences
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus_h = x.copy()
            x_plus_h[i] += h
            grad[i] = (func(x_plus_h) - func(x)) / h
        return grad

    def _local_search(self, func, x, lr=0.1, num_steps=5):
        # Perform a simple gradient-based local search
        x_current = x.copy()
        for _ in range(num_steps):
            grad = self._estimate_gradient(func, x_current)
            x_new = x_current - lr * grad
            x_new = np.clip(x_new, self.bounds[0], self.bounds[1])  # Clip to bounds
            
            f_new = func(x_new)
            self.n_evals += 1 # increment n_evals here since func is called directly
            
            if f_new < func(x_current):
                x_current = x_new
            else:
                break # Stop if no improvement
        return x_current, func(x_current)

    
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
        batch_size = min(2, self.dim)
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)
            
            # Select points by acquisition function
            next_X = self._select_next_points(batch_size)

            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)

            # Update the evaluated points
            self._update_eval_points(next_X, next_y)

            # Local search around the best point
            best_x_local, best_y_local = self._local_search(func, self.best_x)
            if best_y_local < self.best_y:
                self.best_y = best_y_local
                self.best_x = best_x_local
        
        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<GradientEnhancedBO>", line 148, in __call__
 148->             next_y = self._evaluate_points(func, next_X)
  File "<GradientEnhancedBO>", line 80, in _evaluate_points
  80->         y = np.array([func(x) for x in X]).reshape(-1, 1)
  File "<GradientEnhancedBO>", line 80, in <listcomp>
  78 |         # func: takes array of shape (n_dims,) and returns np.float64.
  79 |         # return array of shape (n_points, 1)
  80->         y = np.array([func(x) for x in X]).reshape(-1, 1)
  81 |         self.n_evals += len(X)
  82 |         return y
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
