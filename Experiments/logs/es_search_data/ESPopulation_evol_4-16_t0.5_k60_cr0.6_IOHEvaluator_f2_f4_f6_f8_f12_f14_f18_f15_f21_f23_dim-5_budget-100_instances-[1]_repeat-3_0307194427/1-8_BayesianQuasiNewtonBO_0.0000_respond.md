# Description
**BayesianQuasiNewtonBO (BQNB) - Budget-Aware Refinement:** This algorithm refines the original BQNB by replacing the direct call to the objective function `func` within the BFGS local optimization step with the Gaussian Process (GP) surrogate model. The BFGS optimizer now uses the GP's `predict` method to estimate function values, avoiding exceeding the evaluation budget. The acquisition function remains Expected Improvement (EI). The GP is still updated periodically to balance computational cost and model accuracy. Latin Hypercube Sampling is used for initial exploration.

# Justification
The primary issue with the previous `BayesianQuasiNewtonBO` was the `OverBudgetException`, caused by the `minimize` function calling `func` directly during the BFGS optimization. To address this, the local optimization step now utilizes the Gaussian Process surrogate model to estimate the function values needed by BFGS. This ensures that the budget is not exceeded, as the surrogate model predictions do not count as function evaluations. By using the GP as a proxy for the true objective function during the local search, we can still leverage the gradient-based refinement capabilities of BFGS without violating the budget constraint. This approach balances exploration (through EI and initial sampling) and exploitation (through BFGS refinement guided by the GP).

# Code
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

        # Local optimization using BFGS, using GP to estimate function values
        refined_points = []
        for point in selected_points:
            def gp_objective(x):
                return self.gp.predict(x.reshape(1, -1))[0][0]

            res = minimize(gp_objective, point, method='BFGS', bounds=list(zip(self.bounds[0], self.bounds[1])))
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
## Error
 Traceback (most recent call last):
  File "<BayesianQuasiNewtonBO>", line 122, in __call__
 122->             next_X = self._select_next_points(batch_size, func)
  File "<BayesianQuasiNewtonBO>", line 75, in _select_next_points
  75->             res = minimize(gp_objective, point, method='BFGS', bounds=list(zip(self.bounds[0], self.bounds[1])))
  File "<BayesianQuasiNewtonBO>", line 73, in gp_objective
  71 |         for point in selected_points:
  72 |             def gp_objective(x):
  73->                 return self.gp.predict(x.reshape(1, -1))[0][0]
  74 | 
  75 |             res = minimize(gp_objective, point, method='BFGS', bounds=list(zip(self.bounds[0], self.bounds[1])))
IndexError: invalid index to scalar variable.
