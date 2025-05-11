# Description
Gradient-Enhanced Surrogate Model Averaging Bayesian Optimization (GESMABO) is a Bayesian optimization algorithm that combines the strengths of Gradient-Enhanced Bayesian Optimization (GEBO) and Surrogate Model Averaging Bayesian Optimization (SMABO). It leverages gradient information via finite differences to improve the accuracy of the surrogate model and uses an ensemble of Gaussian Process Regression (GPR) models with different kernels to enhance robustness and exploration. The acquisition function is based on Expected Improvement (EI), modified to incorporate gradient information and averaged across the ensemble of surrogate models. To enhance exploration, a dynamic temperature parameter is introduced, influencing the exploration-exploitation trade-off. Local optimization is used to select the next points.

# Justification
The algorithm combines the strengths of GEBO and SMABO. GEBO uses gradient information to improve the accuracy of the surrogate model, which can lead to faster convergence. SMABO uses an ensemble of surrogate models to improve robustness and exploration, which can help to avoid getting stuck in local optima. The combination of these two techniques should lead to a more efficient and robust Bayesian optimization algorithm.

Specifically:

*   **Gradient Enhancement:** Incorporating gradient information, even if estimated via finite differences, can significantly improve the surrogate model's accuracy, especially in higher dimensions. This is beneficial for navigating complex landscapes more efficiently.
*   **Surrogate Model Averaging:** Averaging predictions from multiple GPR models with different kernels (RBF and Matern) provides a more robust estimate of the function surface and its uncertainty. This reduces the risk of overfitting to a single kernel and improves generalization.
*   **Dynamic Temperature:** The dynamic temperature parameter in the EI acquisition function allows for adaptive control of the exploration-exploitation trade-off. Starting with a higher temperature promotes exploration in the initial stages, while gradually decreasing it focuses on exploitation as the optimization progresses.
*   **Local Optimization:** Using local optimization (L-BFGS-B) to maximize the acquisition function allows for a more precise selection of the next points to evaluate, leading to faster convergence.
*   **Robust EI Calculation:** Clipping the EI calculation and adding a small constant to the denominator prevents `NaN` errors and ensures numerical stability.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from scipy.optimize import minimize

class GESMABO:
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
        self.temperature = 1.0 # Initial temperature for exploration
        self.temperature_decay = 0.95 # Decay rate for the temperature

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
        
        # Define kernels
        kernel_rbf = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        kernel_matern = ConstantKernel(1.0) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        # Initialize models
        gp_rbf = GaussianProcessRegressor(kernel=kernel_rbf, n_restarts_optimizer=5)
        gp_matern = GaussianProcessRegressor(kernel=kernel_matern, n_restarts_optimizer=5)
        
        # Fit models
        gp_rbf.fit(X, y)
        gp_matern.fit(X, y)
        
        return gp_rbf, gp_matern

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

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1)) # Return zeros if no data is available

        mu_rbf, sigma_rbf = self.model_rbf.predict(X, return_std=True)
        mu_matern, sigma_matern = self.model_matern.predict(X, return_std=True)

        mu_rbf = mu_rbf.reshape(-1, 1)
        sigma_rbf = sigma_rbf.reshape(-1, 1)
        mu_matern = mu_matern.reshape(-1, 1)
        sigma_matern = sigma_matern.reshape(-1, 1)
        
        # Average the predictions
        mu = (mu_rbf + mu_matern) / 2.0
        sigma = (sigma_rbf + sigma_matern) / 2.0

        # Expected Improvement with temperature
        best = np.min(self.y)
        imp = best - mu
        Z = imp / (self.temperature * sigma + 1e-9)  # Adding a small constant to avoid division by zero
        ei = imp * norm.cdf(Z) + self.temperature * sigma * norm.pdf(Z)
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
        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            # Fit the model
            self.model_rbf, self.model_matern = self._fit_model(self.X, self.y)

            # Select next points
            X_next = self._select_next_points(batch_size)

            # Evaluate points
            y_next = self._evaluate_points(func, X_next)

            # Update evaluated points
            self._update_eval_points(X_next, y_next)

            # Update temperature
            self.temperature *= self.temperature_decay

        # Return best solution
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Feedback
 The algorithm GESMABO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1571 with standard deviation 0.0999.

took 723.53 seconds to run.