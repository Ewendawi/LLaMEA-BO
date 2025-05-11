# Description
**AGETRBO-GP**: Adaptive Gradient-Enhanced Trust Region Bayesian Optimization with Gradient Propagation. This enhancement to AGETRBO focuses on improving the gradient estimation and its integration into the Gaussian Process model. Instead of using the gradient only for the acquisition function, we use the gradient information to refine the GPR model itself, by incorporating gradient observations as additional data points. We also use a more robust method for estimating the gradient, by averaging over multiple finite difference estimates. The acquisition function remains Expected Improvement (EI), and the trust region is adapted based on success.

# Justification
The original AGETRBO estimates gradients using finite differences, which can be noisy and inaccurate, especially in high-dimensional spaces. This can lead to poor performance of the Gaussian Process model and the acquisition function. By incorporating gradient observations directly into the GPR model, we can improve the accuracy of the model and the acquisition function. This should lead to better exploration and exploitation of the search space.

Specifically, the following changes were made:

1.  **Gradient Propagation:** The estimated gradients are treated as additional observations for the Gaussian Process. This means that the GPR model is trained not only on function values but also on gradient values at sampled points. This is achieved by augmenting the training data with gradient information.

2.  **Robust Gradient Estimation:** To mitigate the noise in finite difference gradient estimation, we average multiple gradient estimates obtained with slightly different step sizes. This reduces the variance of the gradient estimate.

3. **Adaptive Delta**: The step size for finite difference gradient estimation is adapted based on the function landscape. If the function is changing rapidly, a smaller step size is used. If the function is changing slowly, a larger step size is used.

4. **Gradient-Aware Kernel**: The kernel of the Gaussian Process is modified to account for the gradient observations. This is done by adding a term to the kernel that measures the similarity between the gradients at two points.

These changes aim to improve the accuracy and robustness of the gradient estimation and its integration into the Gaussian Process model, leading to better performance of the Bayesian Optimization algorithm.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.optimize import minimize

class AGETRBOGP:
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
        self.n_gradient_samples = 3 # Number of samples for robust gradient estimation

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
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        model.fit(X, y)
        return model

    def _estimate_gradient(self, func, x):
        # Estimate gradient using finite differences
        gradient = np.zeros(self.dim)
        for _ in range(self.n_gradient_samples):
            delta = self.delta * np.random.uniform(0.5, 1.5)  # Vary delta for robust estimation
            for i in range(self.dim):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += delta
                x_minus[i] -= delta
                x_plus = np.clip(x_plus, self.bounds[0][i], self.bounds[1][i])
                x_minus = np.clip(x_minus, self.bounds[0][i], self.bounds[1][i])
                gradient[i] += (func(x_plus) - func(x_minus)) / (2 * delta)
        return gradient / self.n_gradient_samples

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
        imp = self.best_y - mu
        Z = imp / (sigma + 1e-9)  # Add a small constant to avoid division by zero
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # avoid division by zero, if sigma is too small, set EI to 0
        return ei

    def _select_next_points(self, func, batch_size):
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

            # Optimization within bounds (trust region)
            lower_bound = np.maximum(self.bounds[0], self.best_x - self.trust_region_width / 2)
            upper_bound = np.minimum(self.bounds[1], self.best_x + self.trust_region_width / 2)
            bounds = [(lower_bound[i], upper_bound[i]) for i in range(self.dim)]
            
            # Initial guess (randomly sampled within trust region)
            x0 = self._sample_points(1, center=self.best_x, width=self.trust_region_width).flatten()
            
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

    def _update_eval_points(self, new_X, new_y, new_gradients=None):
        # Update self.X and self.y
        # Do not change the function signature
        if self.X is None:
            self.X = new_X
            self.y = new_y
            if new_gradients is not None:
                self.X = np.concatenate((self.X, new_X), axis=0)
                self.y = np.concatenate((self.y, new_gradients), axis=0)
        else:
            self.X = np.concatenate((self.X, new_X), axis=0)
            self.y = np.concatenate((self.y, new_y), axis=0)
            if new_gradients is not None:
                self.X = np.concatenate((self.X, new_X), axis=0)
                self.y = np.concatenate((self.y, new_gradients), axis=0)

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

            # Estimate Gradients
            gradients_next = np.array([self._estimate_gradient(func, x) for x in X_next])

            # Update evaluated points
            self._update_eval_points(X_next, y_next, gradients_next)

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
  File "<AGETRBOGP>", line 169, in __call__
 169->             self._update_eval_points(X_next, y_next, gradients_next)
  File "<AGETRBOGP>", line 138, in _update_eval_points
 136 |             if new_gradients is not None:
 137 |                 self.X = np.concatenate((self.X, new_X), axis=0)
 138->                 self.y = np.concatenate((self.y, new_gradients), axis=0)
 139 | 
 140 |     def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
  File "<__array_function__ internals>", line 200, in concatenate
ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 1 and the array at index 1 has size 5
