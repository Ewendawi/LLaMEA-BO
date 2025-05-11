You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- ADTRBO: 0.1850, 665.50 seconds, **Adaptive Diversity-Enhanced Trust Region Bayesian Optimization (ADTRBO):** This algorithm combines the strengths of ATRBO and DIBO while addressing the NaN error encountered in DIBO. It incorporates an adaptive trust region mechanism, diversity injection, and a robust surrogate model. The algorithm uses a Gaussian Process with a Matern kernel for modeling the objective function. To avoid premature convergence, a diversity term is added to the Lower Confidence Bound (LCB) acquisition function, encouraging exploration in less-visited regions. The trust region size is dynamically adjusted based on the agreement between the GP model's predictions and the actual function evaluations. To handle potential NaN values in the input data, a simple imputation strategy (replacing NaNs with the mean) is employed before fitting the GP model.


- ATRDGBO: 0.1793, 1368.10 seconds, **Adaptive Trust Region with Diversity and Gradient Boosting (ATRDGBO):** This algorithm combines the strengths of ATRBO and DIBO while addressing the NaN issue encountered in DIBO and leveraging gradient boosting for potentially faster and more accurate surrogate modeling. It uses an adaptive trust region to balance exploration and exploitation, injects diversity into the acquisition function to avoid premature convergence, and employs a gradient-boosted tree model as the surrogate for improved accuracy and robustness. It also includes NaN handling.


- EATRBO: 0.1785, 319.28 seconds, **Enhanced Adaptive Trust Region Bayesian Optimization (EATRBO):** This algorithm builds upon ATRBO by incorporating several enhancements to improve its performance and robustness. Key improvements include: 1) Adaptive batch size selection based on the trust region size, allowing for more efficient exploration when the model is uncertain. 2) A modified acquisition function that combines LCB with a term that encourages exploration of the entire search space, especially in early iterations. 3) A more robust trust region adaptation strategy that considers both the prediction error and the uncertainty of the GP model. 4) A dynamic scaling of the exploration factor to balance exploration and exploitation more effectively throughout the optimization process. These modifications aim to improve the algorithm's ability to escape local optima and converge to the global optimum more quickly.


- ATRELSBO: 0.1766, 32.43 seconds, **Adaptive Trust Region with Efficient Local Search Bayesian Optimization (ATRELSBO):** This algorithm combines the adaptive trust region management from ATRBO with the efficient local search strategy from EHBBO. It uses a Gaussian Process with a Matérn kernel, adaptive trust region, LCB acquisition function, and Sobol initial sampling from ATRBO. It incorporates the local search around the best-observed point from EHBBO, with the number of local search steps adjusted dynamically based on the remaining budget. This hybrid approach aims to balance global exploration with local refinement, leveraging the strengths of both ATRBO and EHBBO.




The selected solutions to update are:
## EATRBO
**Enhanced Adaptive Trust Region Bayesian Optimization (EATRBO):** This algorithm builds upon ATRBO by incorporating several enhancements to improve its performance and robustness. Key improvements include: 1) Adaptive batch size selection based on the trust region size, allowing for more efficient exploration when the model is uncertain. 2) A modified acquisition function that combines LCB with a term that encourages exploration of the entire search space, especially in early iterations. 3) A more robust trust region adaptation strategy that considers both the prediction error and the uncertainty of the GP model. 4) A dynamic scaling of the exploration factor to balance exploration and exploitation more effectively throughout the optimization process. These modifications aim to improve the algorithm's ability to escape local optima and converge to the global optimum more quickly.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

class EATRBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * self.dim # Initial samples
        self.trust_region_size = 2.0  # Initial trust region size
        self.exploration_factor = 2.0 # Initial exploration factor
        self.epsilon = 1e-6

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        samples = sampler.random(n=n_points)
        return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, nu=2.5)
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

        # Lower Confidence Bound
        lcb = mu - self.exploration_factor * sigma

        # Encourage exploration of the entire search space
        if self.X is not None:
            distances = cdist(X, self.X)
            min_distances = np.min(distances, axis=1).reshape(-1, 1)
            lcb -= 0.01 * self.exploration_factor * min_distances

        return lcb

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Optimize the acquisition function within the trust region using L-BFGS-B
        best_idx = np.argmin(self.y)
        best_x = self.X[best_idx]
        
        x_starts = best_x + np.random.normal(0, 0.1, size=(batch_size, self.dim))
        x_starts = np.clip(x_starts, self.bounds[0], self.bounds[1])

        candidates = []
        values = []
        for x_start in x_starts:
            # Define trust region bounds
            lower_bound = np.maximum(x_start - self.trust_region_size / 2, self.bounds[0])
            upper_bound = np.minimum(x_start + self.trust_region_size / 2, self.bounds[1])
            
            res = minimize(lambda x: self._acquisition_function(x.reshape(1, -1)),
                           x_start,
                           bounds=np.array([lower_bound, upper_bound]).T,
                           method="L-BFGS-B")
            candidates.append(res.x)
            values.append(res.fun)
        
        return np.array(candidates)

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
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        self.model = self._fit_model(self.X, self.y)
        
        while self.n_evals < self.budget:
            # Adaptive batch size
            batch_size = min(int(np.ceil(self.trust_region_size)), 4)  # Adjust batch size based on trust region
            batch_size = max(1, batch_size) # Ensure batch size is at least 1

            # Optimization
            X_next = self._select_next_points(batch_size)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

            # Adaptive trust region adjustment
            y_pred, sigma = self.model.predict(X_next, return_std=True)
            y_pred = y_pred.reshape(-1, 1)
            sigma = sigma.reshape(-1, 1)

            # Agreement between prediction and actual value
            agreement = np.abs(y_pred - y_next) / (sigma.reshape(-1, 1) + self.epsilon)

            if np.mean(agreement) < 1.0:
                self.trust_region_size *= 1.1  # Increase trust region if model is accurate
            else:
                self.trust_region_size *= 0.9  # Decrease trust region if model is inaccurate
            
            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0) # Clip trust region size

            # Dynamic exploration factor adjustment
            self.exploration_factor = 1.0 + (self.budget - self.n_evals) / self.budget # Reduce exploration over time
            self.exploration_factor = max(0.1, self.exploration_factor) # Ensure exploration factor is at least 0.1
            
            self.model = self._fit_model(self.X, self.y) # Refit the model with new data
            
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        return best_y, best_x

```
The algorithm EATRBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1785 with standard deviation 0.0990.

took 319.28 seconds to run.

## ATRELSBO
**Adaptive Trust Region with Efficient Local Search Bayesian Optimization (ATRELSBO):** This algorithm combines the adaptive trust region management from ATRBO with the efficient local search strategy from EHBBO. It uses a Gaussian Process with a Matérn kernel, adaptive trust region, LCB acquisition function, and Sobol initial sampling from ATRBO. It incorporates the local search around the best-observed point from EHBBO, with the number of local search steps adjusted dynamically based on the remaining budget. This hybrid approach aims to balance global exploration with local refinement, leveraging the strengths of both ATRBO and EHBBO.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.optimize import minimize

class ATRELSBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * self.dim # Initial samples
        self.trust_region_size = 2.0  # Initial trust region size
        self.exploration_factor = 2.0 # Initial exploration factor

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        samples = sampler.random(n=n_points)
        return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, nu=2.5)
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

        # Lower Confidence Bound
        lcb = mu - self.exploration_factor * sigma
        return lcb

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Optimize the acquisition function within the trust region using L-BFGS-B
        best_idx = np.argmin(self.y)
        best_x = self.X[best_idx]
        
        x_starts = best_x + np.random.normal(0, 0.1, size=(batch_size, self.dim))
        x_starts = np.clip(x_starts, self.bounds[0], self.bounds[1])

        candidates = []
        values = []
        for x_start in x_starts:
            # Define trust region bounds
            lower_bound = np.maximum(x_start - self.trust_region_size / 2, self.bounds[0])
            upper_bound = np.minimum(x_start + self.trust_region_size / 2, self.bounds[1])
            
            res = minimize(lambda x: self._acquisition_function(x.reshape(1, -1)),
                           x_start,
                           bounds=np.array([lower_bound, upper_bound]).T,
                           method="L-BFGS-B")
            candidates.append(res.x)
            values.append(res.fun)
        
        return np.array(candidates)

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
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        self.model = self._fit_model(self.X, self.y)
        
        while self.n_evals < self.budget:
            # Optimization
            batch_size = 1
            X_next = self._select_next_points(batch_size)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

            # Adaptive trust region adjustment
            y_pred, sigma = self.model.predict(X_next, return_std=True)
            y_pred = y_pred.reshape(-1, 1)
            
            # Agreement between prediction and actual value
            agreement = np.abs(y_pred - y_next) / sigma.reshape(-1, 1)
            
            if np.mean(agreement) < 1.0:
                self.trust_region_size *= 1.1  # Increase trust region if model is accurate
            else:
                self.trust_region_size *= 0.9  # Decrease trust region if model is inaccurate
            
            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0) # Clip trust region size

            # Dynamic exploration factor adjustment
            self.exploration_factor = 1.0 + (self.budget - self.n_evals) / self.budget # Reduce exploration over time
            
            # Local search around the best point
            best_idx = np.argmin(self.y)
            best_x = self.X[best_idx]
            
            # Dynamically adjust the number of local search steps based on the remaining budget
            remaining_budget = self.budget - self.n_evals
            n_local_steps = min(5, remaining_budget) # Reduce the number of steps as budget decreases
            
            X_local = best_x + np.random.normal(0, 0.1, size=(n_local_steps, self.dim)) # Gaussian mutation
            X_local = np.clip(X_local, self.bounds[0], self.bounds[1])  # Clip to bounds
            y_local = self._evaluate_points(func, X_local)
            self._update_eval_points(X_local, y_local)
            
            self.model = self._fit_model(self.X, self.y) # Refit the model with new data
            
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        return best_y, best_x

```
The algorithm ATRELSBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1766 with standard deviation 0.1034.

took 32.43 seconds to run.

Combine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it. If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.




Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

