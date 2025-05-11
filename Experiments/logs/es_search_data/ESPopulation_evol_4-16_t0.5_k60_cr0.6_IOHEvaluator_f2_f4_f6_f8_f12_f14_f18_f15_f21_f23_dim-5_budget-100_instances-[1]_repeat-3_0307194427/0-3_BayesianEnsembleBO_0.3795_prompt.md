You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

3 algorithms have been designed. The new algorithm should be as **diverse** as possible from the previous ones on every aspect.
If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.
## EfficientHybridBO
**Efficient Hybrid Bayesian Optimization (EHBBO)**: This algorithm combines Gaussian Process Regression (GPR) with Expected Improvement (EI) for exploration-exploitation balance. It employs a Latin Hypercube Sampling (LHS) for initial exploration and a batch selection strategy based on EI and distance metric to ensure diversity in selected points. To improve the computational efficiency, it uses a simplified GPR model with a fixed kernel and optimizes only the kernel amplitude.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

class EfficientHybridBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * (dim + 1)

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
        kernel = C(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-5)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp, y_best):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)
        gamma = (y_best - mu) / sigma
        ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return ei

    def _select_next_points(self, gp, y_best, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        candidates = self._sample_points(100 * self.dim)
        ei = self._acquisition_function(candidates, gp, y_best)
        
        # Select the top batch_size candidates based on EI
        selected_indices = np.argsort(ei)[-batch_size:]
        selected_points = candidates[selected_indices]

        # Ensure diversity by penalizing points that are too close to existing points
        if self.X is not None:
            distances = cdist(selected_points, self.X)
            min_distances = np.min(distances, axis=1)
            # Only select points that are sufficiently far away from existing points
            selected_points = selected_points[min_distances > 0.1]
            if len(selected_points) < batch_size:
              remaining_needed = batch_size - len(selected_points)
              additional_indices = np.argsort(ei)[:-batch_size-1:-1][:remaining_needed]
              additional_points = candidates[additional_indices]
              selected_points = np.concatenate([selected_points, additional_points], axis=0)

        return selected_points[:batch_size]

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
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        
        while self.n_evals < self.budget:
            # Fit the Gaussian Process model
            gp = self._fit_model(self.X, self.y)

            # Select the next points to evaluate
            batch_size = min(self.budget - self.n_evals, 5)
            next_X = self._select_next_points(gp, best_y, batch_size)

            # Evaluate the points
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Update the best solution
            best_idx = np.argmin(self.y)
            best_y = self.y[best_idx][0]
            best_x = self.X[best_idx]

        return best_y, best_x

```
The algorithm EfficientHybridBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1566 with standard deviation 0.1026.

took 4.07 seconds to run.

## AdaptiveTrustRegionBO
**AdaptiveTrustRegionBO (ATBO)**: This algorithm uses a Gaussian Process Regression (GPR) model with a dynamically adjusted trust region to balance exploration and exploitation. It starts with a global search and progressively narrows the search space around promising regions. The trust region size is adapted based on the GPR's uncertainty and the improvement observed in previous iterations. The acquisition function is based on Expected Improvement (EI).  To enhance diversity, a Sobol sequence is used for sampling within the trust region, and a mechanism is included to occasionally expand the trust region if progress stagnates.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from scipy.optimize import minimize

class AdaptiveTrustRegionBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * (dim + 1)
        self.trust_region_center = np.zeros(dim)  # Initialize trust region center
        self.trust_region_radius = 2.5  # Initial trust region radius (half of the search space)
        self.min_radius = 0.1
        self.radius_decay = 0.95
        self.radius_grow = 1.1
        self.gp = None

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points within the trust region
        # return array of shape (n_points, n_dims)
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        sample = sampler.random(n=n_points)
        scaled_sample = qmc.scale(sample, -1.0, 1.0) # Scale to [-1, 1]
        
        # Map to trust region
        points = self.trust_region_center + scaled_sample * self.trust_region_radius
        
        # Clip to bounds
        points = np.clip(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * \
                 RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + \
                 WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-7, 1e-3))
        
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function (Expected Improvement)
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None or self.X is None or self.y is None:
            return np.zeros((len(X), 1))  # Return zero if model is not fitted yet

        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)
        y_best = np.min(self.y)
        gamma = (y_best - mu) / sigma
        ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Sample candidate points
        candidates = self._sample_points(100 * self.dim)
        
        # Calculate acquisition function values
        ei = self._acquisition_function(candidates)
        
        # Select top batch_size candidates based on EI
        selected_indices = np.argsort(ei.flatten())[-batch_size:]
        selected_points = candidates[selected_indices]
        
        return selected_points

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
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        
        while self.n_evals < self.budget:
            # Fit the Gaussian Process model
            gp = self._fit_model(self.X, self.y)
            
            # Select the next points to evaluate
            batch_size = min(self.budget - self.n_evals, 5)
            next_X = self._select_next_points(batch_size)
            
            # Evaluate the points
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Update the best solution
            best_idx = np.argmin(self.y)
            current_best_y = self.y[best_idx][0]
            current_best_x = self.X[best_idx]

            # Adjust trust region
            if current_best_y < best_y:
                # Improvement: move trust region center and shrink radius
                self.trust_region_center = current_best_x
                self.trust_region_radius *= self.radius_decay
                best_y = current_best_y
                best_x = current_best_x
            else:
                # No improvement: potentially expand trust region
                if self.trust_region_radius < 1.0:
                    self.trust_region_radius *= self.radius_grow
                    self.trust_region_radius = min(self.trust_region_radius, 2.5) # Limit to initial radius
            
            self.trust_region_radius = max(self.trust_region_radius, self.min_radius)

        return best_y, best_x

```
The algorithm AdaptiveTrustRegionBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1833 with standard deviation 0.1042.

took 486.80 seconds to run.

## BayesianQuasiNewtonBO
**BayesianQuasiNewtonBO (BQNB)**: This algorithm combines Bayesian Optimization with a Quasi-Newton method (specifically, BFGS) for local refinement. It uses a Gaussian Process (GP) to model the objective function and Expected Improvement (EI) as the acquisition function. After selecting a promising point via EI, it performs a local optimization step using BFGS, initialized at the EI-selected point, to refine the solution. This aims to leverage the global exploration of BO with the efficient local convergence of gradient-based methods. To mitigate the computational cost associated with repeated GP fitting, the GP is updated only periodically. Initial points are sampled using Latin Hypercube Sampling.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class BayesianQuasiNewtonBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * (dim + 1)
        self.gp = None
        self.update_interval = 5  # Update GP every 5 iterations
        self.last_gp_update = 0


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
        kernel = C(1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None or self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)
        y_best = np.min(self.y)
        gamma = (y_best - mu) / sigma
        ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size, func):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        candidates = self._sample_points(100 * self.dim)
        ei = self._acquisition_function(candidates)
        selected_indices = np.argsort(ei.flatten())[-batch_size:]
        selected_points = candidates[selected_indices]

        # Local optimization using BFGS
        refined_points = []
        for point in selected_points:
            res = minimize(func, point, method='BFGS', bounds=list(zip(self.bounds[0], self.bounds[1])))
            refined_points.append(res.x)
        refined_points = np.array(refined_points)

        return refined_points

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
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        
        while self.n_evals < self.budget:
            # Fit the Gaussian Process model
            if self.n_evals - self.last_gp_update >= self.update_interval:
                self._fit_model(self.X, self.y)
                self.last_gp_update = self.n_evals

            # Select the next points to evaluate
            batch_size = min(self.budget - self.n_evals, 5)
            next_X = self._select_next_points(batch_size, func)

            # Evaluate the points
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Update the best solution
            best_idx = np.argmin(self.y)
            best_y = self.y[best_idx][0]
            best_x = self.X[best_idx]

        return best_y, best_x

```
An error occurred : Traceback (most recent call last):
  File "<BayesianQuasiNewtonBO>", line 119, in __call__
 119->             next_X = self._select_next_points(batch_size, func)
  File "<BayesianQuasiNewtonBO>", line 72, in _select_next_points
  70 |         refined_points = []
  71 |         for point in selected_points:
  72->             res = minimize(func, point, method='BFGS', bounds=list(zip(self.bounds[0], self.bounds[1])))
  73 |             refined_points.append(res.x)
  74 |         refined_points = np.array(refined_points)
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')




A code structure guide is as follows and keep the comments from the guide when generating the code.

```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
class <AlgorithmName>:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = <your_strategy>

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)

        self.n_evals += len(X)
    
    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        # Do not change the function signature
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        self._evaluate_points()
        self._update_eval_points()
        while self.n_evals < budget:
            # Optimization

            # select points by acquisition function
            self._evaluate_points()
            self._update_eval_points()

        return best_y, best_x
    
```


Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>
