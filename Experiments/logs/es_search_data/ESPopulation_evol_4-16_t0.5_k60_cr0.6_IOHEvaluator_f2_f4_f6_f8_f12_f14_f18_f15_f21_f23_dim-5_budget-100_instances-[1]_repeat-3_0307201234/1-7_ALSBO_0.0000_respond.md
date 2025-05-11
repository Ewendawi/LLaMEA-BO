# Description
**Adaptive Local Search Bayesian Optimization (ALSBO):** This algorithm builds upon EHBBO by introducing an adaptive local search strategy. The key enhancements include: (1) Dynamically adjusting the local search radius based on the function's landscape and the optimization progress. A larger radius is used initially to explore more broadly, and it shrinks as the algorithm converges to refine the solution. (2) Employing a more sophisticated local search method, specifically the BFGS optimizer, instead of simple Gaussian mutation. This allows for a more efficient and targeted exploration of the local neighborhood. (3) Adapting the number of initial samples based on dimensionality. (4) Using a more robust kernel with optimized hyperparameters.

# Justification
The original EHBBO uses a fixed local search radius and a simple Gaussian mutation, which might be inefficient for complex functions. Adapting the local search radius allows for a more flexible exploration strategy, adapting to the function's landscape. BFGS is a more efficient local optimizer than simple Gaussian mutation, leading to faster convergence. Adapting the number of initial samples based on dimensionality ensures sufficient initial exploration, especially in high-dimensional spaces. Using a more robust kernel and optimizing its hyperparameters improves the accuracy of the GP surrogate model. These changes aim to improve both exploration and exploitation, leading to better performance on the BBOB test suite.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize

class ALSBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 5 * self.dim  # Increased initial samples based on dimensionality
        self.local_search_radius = 1.0 # Initial local search radius
        self.radius_decay = 0.95 # Decay factor for the local search radius

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        samples = sampler.random(n=n_points)
        return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * \
                 Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=1.5) # Matern kernel
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement
        best = np.min(self.y)
        imp = mu - best
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # avoid division by zero
        return -ei  # we want to maximize EI

    def _select_next_points(self, batch_size):
        # Optimize the acquisition function using L-BFGS-B
        x_starts = self._sample_points(batch_size)
        candidates = []
        values = []
        for x_start in x_starts:
            res = minimize(lambda x: self._acquisition_function(x.reshape(1, -1)),
                           x_start,
                           bounds=[(-5, 5)] * self.dim,
                           method="L-BFGS-B")
            candidates.append(res.x)
            values.append(res.fun)

        return np.array(candidates)

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
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

            # Local search around the best point
            best_idx = np.argmin(self.y)
            best_x = self.X[best_idx]

            # Local search using BFGS
            def local_obj(x):
                return func(x)  # Use the original function for local search

            res = minimize(local_obj, best_x,
                           bounds=[(-5, 5)] * self.dim,
                           method="BFGS")
            x_local = res.x
            y_local = func(x_local)  # Evaluate the result of local search
            self._update_eval_points(x_local.reshape(1, -1), np.array([[y_local]]))
            self.n_evals +=1

            self.model = self._fit_model(self.X, self.y) # Refit the model with new data

            # Decay the local search radius
            self.local_search_radius *= self.radius_decay

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<ALSBO>", line 101, in __call__
 101->             res = minimize(local_obj, best_x,
  File "<ALSBO>", line 99, in local_obj
  97 |             # Local search using BFGS
  98 |             def local_obj(x):
  99->                 return func(x)  # Use the original function for local search
 100 | 
 101 |             res = minimize(local_obj, best_x,
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
