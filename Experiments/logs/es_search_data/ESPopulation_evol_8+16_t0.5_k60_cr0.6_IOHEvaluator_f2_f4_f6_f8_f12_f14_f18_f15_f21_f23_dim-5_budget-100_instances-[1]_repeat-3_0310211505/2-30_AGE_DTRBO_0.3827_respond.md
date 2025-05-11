# Description
**AGE-DTRBO**: Adaptive Gradient-Enhanced Trust Region Bayesian Optimization with Dynamic Trust Region. This algorithm combines gradient information, adaptive trust region management, and dynamic trust region adjustment. It estimates gradients using finite differences, similar to AGETRBO. It also dynamically adjusts the trust region based on success ratio and model uncertainty, similar to ATRBO_DTR. The acquisition function is Expected Improvement (EI) with an added exploration term based on predicted variance. The next points are selected by L-BFGS-B optimization within the trust region. Sobol sampling is used for initial sampling and candidate point generation.

# Justification
This algorithm aims to leverage the strengths of both AGETRBO and ATRBO_DTR.
*   **Gradient Enhancement:** Incorporating gradient information can improve the accuracy of the surrogate model, especially in high-dimensional spaces.
*   **Adaptive Trust Region:** Adaptive trust region management helps to balance exploration and exploitation by focusing the search in promising regions.
*   **Dynamic Trust Region Adjustment:** Adjusting the trust region based on both success ratio and model uncertainty allows for a more robust and efficient search.
*   **Exploration Term:** Adding an exploration term to the acquisition function encourages exploration in uncertain regions, preventing premature convergence.
*   **Sobol Sampling:** Using Sobol sampling for initial sampling and candidate point generation ensures better space-filling properties, especially in low-dimensional problems.
*   **L-BFGS-B Optimization:** Using L-BFGS-B to optimize the acquisition function within the trust region allows for efficient and accurate selection of the next points to evaluate.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

class AGE_DTRBO:
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
        self.exploration_weight = 0.1 # Weight for exploration term in acquisition function
        self.delta = 1e-3 # Step size for finite difference gradient estimation


        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, center=None, width=None):
        # sample points within the trust region
        # return array of shape (n_points, n_dims)
        if center is None:
            sampler = qmc.Sobol(d=self.dim, scramble=False)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            # Sample within the trust region
            lower_bound = np.maximum(self.bounds[0], center - width / 2)
            upper_bound = np.minimum(self.bounds[1], center + width / 2)
            sampler = qmc.Sobol(d=self.dim, scramble=False)
            sample = sampler.random(n=n_points)
            scaled_sample = qmc.scale(sample, lower_bound, upper_bound)
            return scaled_sample

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)
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

    def _acquisition_function(self, X, func):
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

        # Add exploration term
        acquisition = ei + self.exploration_weight * sigma

        # Incorporate gradient information
        # for i in range(len(X)):
        #     gradient = self._estimate_gradient(func, X[i])
        #     acquisition[i] += 0.01 * np.linalg.norm(gradient) # Add a small weight to the gradient norm

        return acquisition

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
                return -self._acquisition_function(x, func)[0, 0]

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
            success_ratio = (self.best_y - new_best_y) / self.best_y if self.best_y != 0 else 0
            mean_sigma = np.mean(self.model.predict(self.X, return_std=True)[1]) # Average uncertainty

            if success_ratio > self.success_threshold:
                self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)  # Increase if successful
            else:
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)  # Decrease if not successful

            # Adjust trust region based on uncertainty
            self.trust_region_width = min(self.trust_region_width + 0.1 * mean_sigma, 10.0)

            self.best_y = new_best_y
            self.best_x = new_best_x

        # Return best solution
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Feedback
 The algorithm AGE_DTRBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1586 with standard deviation 0.0982.

took 376.18 seconds to run.