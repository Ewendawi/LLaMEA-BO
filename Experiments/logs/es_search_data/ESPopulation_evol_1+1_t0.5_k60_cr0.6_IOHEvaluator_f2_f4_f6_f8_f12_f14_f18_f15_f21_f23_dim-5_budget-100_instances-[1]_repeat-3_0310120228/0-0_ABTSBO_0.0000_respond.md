# Description
Adaptive Batch Bayesian Optimization with Thompson Sampling and Local Search (ABTSBO): This algorithm combines adaptive batch Bayesian optimization with Thompson Sampling for exploration and exploitation, and integrates a local search strategy to refine promising solutions. The batch size is dynamically adjusted based on the optimization progress. Thompson Sampling is used as the acquisition function. A simple local search is performed around the best point found so far.

# Justification
1.  **Adaptive Batch Size:** Adjusting the batch size allows for a balance between exploration (larger batch size) and exploitation (smaller batch size). Initially, a larger batch size is used to explore the search space, and as the optimization progresses, the batch size is reduced to focus on promising regions. The adaptive nature is crucial for different BBOB functions with varying characteristics.
2.  **Thompson Sampling:** Thompson Sampling is a probabilistic acquisition function that naturally balances exploration and exploitation. Its simplicity and effectiveness make it suitable for high-dimensional problems.
3.  **Local Search:** Integrating a local search strategy refines the solutions obtained by Bayesian optimization. It helps to fine-tune the solutions and potentially escape local optima. The local search is computationally inexpensive and can significantly improve the performance.
4.  **Computational Efficiency:** The algorithm is designed to be computationally efficient by using a simple Gaussian Process Regression model (using scikit-learn) and a computationally inexpensive local search strategy.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ABTSBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim  # Initial number of points
        self.batch_size = min(10, dim) # Initial batch size, adaptively adjusted
        self.gp = None
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
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None:
            return np.zeros((len(X), 1))  # Return zeros if the model is not fitted yet

        mu, sigma = self.gp.predict(X, return_std=True)
        return mu.reshape(-1, 1) # Thompson Sampling: use the predicted mean

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Sample candidate points
        candidate_points = self._sample_points(10 * batch_size)
        
        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_points)
        
        # Select the points with the highest acquisition function values
        indices = np.argsort(acquisition_values.flatten())[::-1][:batch_size]
        
        return candidate_points[indices]

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
        
        # Update best solution
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]

    def _local_search(self, func, x, radius=0.1, num_points=5):
        # Perform local search around x
        # radius: the search radius
        # num_points: the number of points to sample

        # Sample points around x
        X = np.random.uniform(low=np.maximum(self.bounds[0], x - radius),
                                high=np.minimum(self.bounds[1], x + radius),
                                size=(num_points, self.dim))
        
        # Evaluate the points
        y = self._evaluate_points(func, X)
        
        # Find the best point
        best_index = np.argmin(y)
        
        return X[best_index], y[best_index][0]
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Fit the model
            self._fit_model(self.X, self.y)
            
            # Select next points
            X_next = self._select_next_points(self.batch_size)
            
            # Evaluate points
            y_next = self._evaluate_points(func, X_next)
            
            # Update evaluated points
            self._update_eval_points(X_next, y_next)

            # Local search around the best point
            best_x_local, best_y_local = self._local_search(func, self.best_x)
            if best_y_local < self.best_y:
                self.best_x = best_x_local
                self.best_y = best_y_local
        
            # Adapt batch size
            self.batch_size = max(1, int(self.batch_size * 0.95)) # Reduce batch size gradually

            if self.n_evals >= self.budget:
                break

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<ABTSBO>", line 139, in __call__
 139->             best_x_local, best_y_local = self._local_search(func, self.best_x)
  File "<ABTSBO>", line 106, in _local_search
 106->         y = self._evaluate_points(func, X)
  File "<ABTSBO>", line 75, in _evaluate_points
  75->         y = np.array([func(x) for x in X]).reshape(-1, 1)
  File "<ABTSBO>", line 75, in <listcomp>
  73 |         # return array of shape (n_points, 1)
  74 |         
  75->         y = np.array([func(x) for x in X]).reshape(-1, 1)
  76 |         self.n_evals += len(X)
  77 |         return y
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
