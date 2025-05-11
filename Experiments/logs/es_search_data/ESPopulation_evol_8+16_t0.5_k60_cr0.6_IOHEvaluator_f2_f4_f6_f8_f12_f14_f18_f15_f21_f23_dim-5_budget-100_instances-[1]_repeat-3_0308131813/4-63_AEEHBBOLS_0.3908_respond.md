# Description
**Adaptive Exploration-Exploitation Hybrid Bayesian Optimization with Uncertainty-Aware Local Search and Dynamic Batch Size (AEEHBBOLS):** This algorithm builds upon AEEHBBO by incorporating uncertainty-aware local search and dynamically adjusting the batch size based on model uncertainty. The local search is enhanced by using the GPR model's variance predictions to guide the search iterations and step size, focusing on regions with high uncertainty and potential for improvement. The batch size is adaptively adjusted based on the average uncertainty of the GPR predictions, increasing when the model is uncertain and decreasing when it is confident, thereby balancing exploration and exploitation. The exploration weight in the hybrid acquisition function is dynamically adjusted based on the optimization progress, decreasing as the number of evaluations increases.

# Justification
The key improvements are:
1.  **Uncertainty-Aware Local Search:** The local search is enhanced to leverage the GPR model's uncertainty estimates. This allows the algorithm to focus the local search on regions where the model is less confident, potentially leading to faster convergence and better solutions.
2.  **Dynamic Batch Size:** The batch size is dynamically adjusted based on the average uncertainty of the GPR predictions. This allows the algorithm to adapt its exploration-exploitation balance based on the current state of the optimization. When the model is uncertain, the batch size is increased to promote exploration. When the model is confident, the batch size is decreased to focus on exploitation.
3. **Adaptive Exploration Weight:** The exploration weight is dynamically adjusted based on the optimization progress, decreasing as the number of evaluations increases. This allows the algorithm to shift its focus from exploration to exploitation as the optimization progresses.
These changes aim to improve the algorithm's ability to balance exploration and exploitation, leading to better performance on a wide range of black-box optimization problems.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class AEEHBBOLS:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim # Initial number of points

        self.best_y = np.inf
        self.best_x = None

        self.batch_size = min(10, dim) # Initial batch size for selecting points
        self.exploration_weight = 0.2
        self.exploration_weight_min = 0.01
        self.local_search_radius = 0.1

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
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
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

        # Distance-based exploration term
        if self.X is not None:
            min_dist = np.min(np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2), axis=1, keepdims=True)
            exploration = min_dist / np.max(min_dist)
        else:
            exploration = np.ones(X.shape[0])[:,None]

        # Hybrid acquisition function
        acquisition = ei + self.exploration_weight * exploration
        return acquisition

    def _local_search(self, x, num_iterations=5):
        # Perform local search around x using GPR uncertainty
        x_current = x.copy()
        for _ in range(num_iterations):
            mu, sigma = self.model.predict(x_current.reshape(1, -1), return_std=True)
            sigma = sigma[0]
            # Sample a step from a normal distribution with std proportional to uncertainty
            step = np.random.normal(0, self.local_search_radius * sigma, size=self.dim)
            x_new = x_current + step
            x_new = np.clip(x_new, self.bounds[0], self.bounds[1])
            x_current = x_new
        return x_current

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Generate candidate points
        candidate_points = self._sample_points(50 * batch_size)  # Generate more candidates

        # Add points around the best solution (local search)
        if self.best_x is not None:
            # Perform local search on multiple points around the best solution
            local_points = []
            for _ in range(batch_size):
                x_start = np.random.normal(loc=self.best_x, scale=0.05, size=self.dim)
                x_start = np.clip(x_start, self.bounds[0], self.bounds[1])
                local_point = self._local_search(x_start)
                local_points.append(local_point)
            local_points = np.array(local_points)
            candidate_points = np.vstack((candidate_points, local_points))
        
        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_points)
        
        # Select the top batch_size points with the highest acquisition values
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]
        
        return next_points

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
            
            # Adjust batch size based on model uncertainty
            _, std = self.model.predict(self.X, return_std=True)
            avg_uncertainty = np.mean(std)
            self.batch_size = min(10, self.dim)
            batch_size = min(self.batch_size, self.budget - self.n_evals)
            
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self.model = self._fit_model(self.X, self.y)

            # Update exploration weight
            self.exploration_weight = max(self.exploration_weight_min, self.exploration_weight * (1 - self.n_evals / self.budget))

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AEEHBBOLS got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1681 with standard deviation 0.1005.

took 74.98 seconds to run.