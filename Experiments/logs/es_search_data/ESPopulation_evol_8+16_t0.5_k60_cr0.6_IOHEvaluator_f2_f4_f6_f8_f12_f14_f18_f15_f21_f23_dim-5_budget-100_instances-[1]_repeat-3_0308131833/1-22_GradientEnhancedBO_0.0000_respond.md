# Description
**GradientEnhancedBO (GEBO) with Gradient Approximation and Adaptive Exploration:** This enhanced version of GEBO focuses on approximating gradients efficiently and adaptively adjusting the exploration-exploitation balance. It uses a finite difference method to approximate gradients and incorporates them into the acquisition function. The gradient weight is dynamically adjusted based on the optimization progress and the estimated uncertainty of the GP model. A local search is performed around the best-observed solution to refine the search.

# Justification
The key improvements are:

1.  **Gradient Approximation:** Instead of relying on true gradient information (which is unavailable in black-box optimization), a finite difference method is used to approximate the gradients. This provides a computationally feasible way to incorporate gradient information into the optimization process.
2.  **Adaptive Gradient Weight:** The weight given to the gradient term in the acquisition function is dynamically adjusted. Initially, a higher weight is used to encourage exploration based on gradient information. As the optimization progresses and the GP model becomes more accurate, the gradient weight is reduced to favor exploitation. This adaptive strategy helps to balance exploration and exploitation effectively. The adjustment is based on the variance predicted by the GP. High variance indicates uncertainty, justifying more gradient-based exploration.
3.  **Local Search:** A local search is performed around the best-observed solution to refine the search. This helps to improve the convergence of the algorithm by exploiting the local structure of the objective function.
4.  **Efficient Batch Selection:** The batch selection strategy is optimized by generating a larger pool of candidate points and selecting the best ones based on the acquisition function. This helps to improve the efficiency of the optimization process.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

class GradientEnhancedBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.gp = None
        self.gradient_weight = 0.1
        self.best_x = None
        self.best_y = np.inf
        self.finite_diff_step = 0.1 # Step size for finite difference gradient approximation

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)
        gp.fit(X, y)
        return gp

    def _approximate_gradients(self, func, X):
        # Approximate gradients using finite differences
        gradients = np.zeros_like(X)
        for i in range(len(X)):
            for j in range(self.dim):
                x_plus = X[i].copy()
                x_minus = X[i].copy()
                x_plus[j] += self.finite_diff_step
                x_minus[j] -= self.finite_diff_step

                # Ensure the perturbed points are within bounds
                x_plus[j] = np.clip(x_plus[j], self.bounds[0, j], self.bounds[1, j])
                x_minus[j] = np.clip(x_minus[j], self.bounds[0, j], self.bounds[1, j])

                y_plus = func(x_plus)
                y_minus = func(x_minus)
                gradients[i, j] = (y_plus - y_minus) / (2 * self.finite_diff_step)
        return gradients

    def _acquisition_function(self, func, X):
        if self.gp is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.gp.predict(X, return_std=True)
        acquisition = mu - 0.1 * sigma

        # Adaptive gradient weight
        if self.X is not None:
            mean_variance = np.mean(sigma)
            self.gradient_weight = np.clip(mean_variance, 0.01, 0.5) # Adjust range as needed

        # Incorporate gradient information
        gradients = self._approximate_gradients(func, X)
        acquisition += self.gradient_weight * np.linalg.norm(gradients, axis=1).reshape(-1, 1)

        return acquisition.reshape(-1, 1)

    def _select_next_points(self, func, batch_size):
        x_tries = self._sample_points(batch_size * 10)
        acq_values = self._acquisition_function(func, x_tries)
        indices = np.argsort(acq_values.flatten())[::-1][:batch_size]
        return x_tries[indices]

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
        
        best_index = np.argmin(self.y)
        if self.y[best_index, 0] < self.best_y:
            self.best_x = self.X[best_index]
            self.best_y = self.y[best_index, 0]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        while self.n_evals < self.budget:
            self.gp = self._fit_model(self.X, self.y)

            batch_size = min(10, self.budget - self.n_evals)
            next_X = self._select_next_points(func, batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Local search around the best solution
            if self.n_evals + 5 <= self.budget:
                x_local_tries = self._sample_points(5) + self.best_x # Sample around best_x
                x_local_tries = np.clip(x_local_tries, self.bounds[0], self.bounds[1])
                y_local_tries = self._evaluate_points(func, x_local_tries)
                self._update_eval_points(x_local_tries, y_local_tries)

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<GradientEnhancedBO>", line 105, in __call__
 105->             next_X = self._select_next_points(func, batch_size)
  File "<GradientEnhancedBO>", line 74, in _select_next_points
  74->         acq_values = self._acquisition_function(func, x_tries)
  File "<GradientEnhancedBO>", line 67, in _acquisition_function
  67->         gradients = self._approximate_gradients(func, X)
  File "<GradientEnhancedBO>", line 50, in _approximate_gradients
  48 | 
  49 |                 y_plus = func(x_plus)
  50->                 y_minus = func(x_minus)
  51 |                 gradients[i, j] = (y_plus - y_minus) / (2 * self.finite_diff_step)
  52 |         return gradients
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
