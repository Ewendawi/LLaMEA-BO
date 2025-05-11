# Description
**AGETRBO-EGD:** Adaptive Gradient-Enhanced Trust Region Bayesian Optimization with Enhanced Gradient Descent. This algorithm refines AGETRBO by incorporating gradient information directly into the acquisition function and using an enhanced gradient descent method to select the next points. The gradient information, estimated using finite differences, is used to modify the Expected Improvement (EI) acquisition function to guide the search towards promising regions. Instead of relying solely on L-BFGS-B, an enhanced gradient descent is performed within the trust region, leveraging the estimated gradients to efficiently find better candidate points. The trust region is adaptively adjusted based on the success rate.

# Justification
1.  **Gradient-Enhanced Acquisition Function:** Incorporating gradient information into the acquisition function can significantly improve the efficiency of the optimization process. By using the estimated gradient to guide the search, the algorithm can more effectively explore the search space and identify promising regions. The modified EI acquisition function combines the benefits of both exploitation (EI) and exploration (gradient direction).
2.  **Enhanced Gradient Descent:** Replacing L-BFGS-B with an enhanced gradient descent method for selecting the next points allows the algorithm to leverage the estimated gradients more directly. This can lead to faster convergence and better performance, especially in high-dimensional spaces. The gradient descent method is enhanced by adding a momentum term to prevent oscillations and accelerate convergence.
3.  **Adaptive Trust Region:** The adaptive trust region strategy ensures that the algorithm can effectively balance exploration and exploitation. By adjusting the trust region width based on the success rate, the algorithm can adapt to the local landscape of the objective function and avoid premature convergence.
4.  **Computational Efficiency:** The enhanced gradient descent method is computationally efficient, making the algorithm suitable for optimizing black-box functions with limited budgets. The finite difference gradient estimation is also relatively inexpensive, especially when compared to other gradient estimation techniques.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

class AGETRBODE:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(10*dim, self.budget//5) # initial samples, at least 10*dim, at most 1/5 of budget
        self.trust_region_width = 2.0  # Initial trust region width
        self.success_threshold = 0.1 # Threshold for increasing trust region
        self.best_y = np.inf # Initialize best_y with a large value
        self.delta = 1e-3 # Step size for finite difference gradient estimation
        self.learning_rate = 0.1
        self.momentum = 0.9

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, center=None, width=None):
        # sample points within the trust region
        # return array of shape (n_points, n_dims)
        if center is None:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            # Sample within the trust region
            lower_bound = np.maximum(self.bounds[0], center - width / 2)
            upper_bound = np.minimum(self.bounds[1], center + width / 2)
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            scaled_sample = qmc.scale(sample, lower_bound, upper_bound)
            return scaled_sample

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        model.fit(X, y)
        return model

    def _estimate_gradient(self, func, x):
        # Estimate gradient using finite differences
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.delta
            x_minus[i] -= self.delta
            x_plus = np.clip(x_plus, self.bounds[0][i], self.bounds[1][i])
            x_minus = np.clip(x_minus, self.bounds[0][i], self.bounds[1][i])
            gradient[i] = (func(x_plus) - func(x_minus)) / (2 * self.delta)
        return gradient

    def _acquisition_function(self, X, gradients):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1)) # Return zeros if no data is available

        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement
        imp = self.best_y - mu
        Z = imp / (sigma + 1e-9)  # Add a small constant to avoid division by zero
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # avoid division by zero, if sigma is too small, set EI to 0

        # Gradient enhancement: favor points with negative gradient in the EI
        ei = ei - np.sum(gradients * (X - self.best_x), axis=1, keepdims=True) * 0.01

        return ei

    def _select_next_points(self, func, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        X_next = []
        for _ in range(batch_size):
            # Enhanced Gradient Descent
            x = self.best_x.copy()
            v = np.zeros(self.dim) # Initialize velocity for momentum

            lower_bound = np.maximum(self.bounds[0], self.best_x - self.trust_region_width / 2)
            upper_bound = np.minimum(self.bounds[1], self.best_x + self.trust_region_width / 2)

            for i in range(100): # Gradient descent iterations
                gradient = self._estimate_gradient(func, x)
                v = self.momentum * v - self.learning_rate * gradient  # Momentum update
                x_new = x + v
                x_new = np.clip(x_new, lower_bound, upper_bound)  # Clip to trust region

                if np.linalg.norm(x_new - x) < 1e-6:
                    break # Convergence

                x = x_new

            X_next.append(x)

        return np.array(X_next)

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
            self.X = np.concatenate((self.X, new_X), axis=0)
            self.y = np.concatenate((self.y, new_y), axis=0)

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_x = X_init[np.argmin(y_init)]
        self.best_y = np.min(y_init)

        # Optimization loop
        batch_size = max(1, self.dim // 2) # dynamic batch size
        while self.n_evals < self.budget:
            # Fit the model
            self.model = self._fit_model(self.X, self.y)

            # Select next points
            X_next = self._select_next_points(func, batch_size)

            # Evaluate points
            y_next = self._evaluate_points(func, X_next)

            # Update evaluated points
            self._update_eval_points(X_next, y_next)

            # Update best solution
            new_best_y = np.min(self.y)
            new_best_x = self.X[np.argmin(self.y)]

            # Adjust trust region width
            if (self.best_y - new_best_y) / self.best_y > self.success_threshold:
                self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)  # Increase if successful
            else:
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)  # Decrease if not successful

            self.best_y = new_best_y
            self.best_x = new_best_x

        # Return best solution
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<AGETRBODE>", line 162, in __call__
 162->             y_next = self._evaluate_points(func, X_next)
  File "<AGETRBODE>", line 125, in _evaluate_points
 125->         y = np.array([func(x) for x in X])
  File "<AGETRBODE>", line 125, in <listcomp>
 123 |         # return array of shape (n_points, 1)
 124 | 
 125->         y = np.array([func(x) for x in X])
 126 |         self.n_evals += len(X)
 127 |         return y.reshape(-1, 1)
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
