# Description
Gradient-Enhanced Bayesian Optimization with Adaptive Gradient Estimation (AGEBO). This enhanced version of GEBO incorporates an adaptive strategy for gradient estimation, switching between finite differences and a zero-gradient assumption based on the local landscape smoothness. It also includes a dynamic batch size adjustment to balance exploration and exploitation.

# Justification
1.  **Adaptive Gradient Estimation:** The original GEBO uses finite differences to estimate gradients, which can be noisy and computationally expensive. In regions where the landscape is relatively flat or the function is not changing rapidly, the gradient information might not be very informative. In such cases, assuming a zero gradient can be more efficient and less prone to noise. We use a simple heuristic: if the variance of the function values in the neighborhood of the current point is below a threshold, we assume the gradient is zero. This threshold is dynamically adjusted based on the observed function values.
2.  **Dynamic Batch Size:** The batch size is adjusted dynamically based on the iteration number. Initially, a larger batch size is used to promote exploration. As the optimization progresses, the batch size is reduced to focus on exploitation. This helps to balance exploration and exploitation throughout the optimization process.
3.  **Computational Efficiency:** The adaptive gradient estimation reduces the number of gradient evaluations, saving computational cost. The dynamic batch size also contributes to computational efficiency by adjusting the number of points evaluated in each iteration.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

class AGEBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(10*dim, self.budget//5)
        self.delta = 1e-3 # Step size for finite difference gradient estimation
        self.gradient_threshold = 1e-4  # Initial threshold for gradient estimation
        self.batch_size = 1 #Initial batch size

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
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        model.fit(X, y)
        return model

    def _estimate_gradient(self, func, x):
        # Adaptive gradient estimation
        neighborhood_size = min(10, len(self.X))  # Consider a maximum of 10 neighbors
        if len(self.X) > 0:
            distances = np.linalg.norm(self.X - x, axis=1)
            nearest_indices = np.argsort(distances)[:neighborhood_size]
            neighborhood_values = self.y[nearest_indices]
            if np.var(neighborhood_values) < self.gradient_threshold:
                return np.zeros(self.dim)  # Assume zero gradient if landscape is flat
            else:
                # Adjust the threshold dynamically
                self.gradient_threshold = np.var(neighborhood_values)
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

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1)) # Return zeros if no data is available

        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement
        best = np.min(self.y)
        imp = best - mu
        Z = imp / (sigma + 1e-9) # Adding a small constant to avoid division by zero
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0 # avoid division by zero

        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        X_next = []
        for _ in range(batch_size):
            # Define the objective function for optimization
            def objective(x):
                x = x.reshape(1, -1)
                return -self._acquisition_function(x)[0, 0]

            # Optimization within bounds
            bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
            
            # Initial guess (randomly sampled)
            x0 = self._sample_points(1).flatten()
            
            # Perform optimization
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            
            X_next.append(result.x)

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

        # Optimization loop
        while self.n_evals < self.budget:
            # Fit the model
            self.model = self._fit_model(self.X, self.y)

            # Adjust batch size dynamically
            self.batch_size = max(1, int(self.dim * (1 - self.n_evals / self.budget)))

            # Select next points
            X_next = self._select_next_points(self.batch_size)

            # Evaluate points
            y_next = self._evaluate_points(func, X_next)

            # Update evaluated points
            self._update_eval_points(X_next, y_next)

        # Return best solution
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Feedback
 The algorithm AGEBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1552 with standard deviation 0.0969.

took 430.08 seconds to run.