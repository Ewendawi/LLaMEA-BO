# Description
**Adaptive Temperature Bayesian Optimization (ATBO):** This algorithm refines the EHBBO algorithm by introducing an adaptive temperature scaling mechanism for Thompson Sampling, which dynamically adjusts the exploration-exploitation balance based on the optimization progress and function landscape. It also incorporates a more robust local search strategy using multiple restarts and a slightly modified acquisition function to enhance local refinement. The initial sampling is also improved by using a Sobol sequence instead of Latin Hypercube sampling.

# Justification
The key improvements are:

1.  **Adaptive Temperature Scaling:** The original EHBBO uses a temperature that depends only on the best-seen function value. This can be too simplistic. The ATBO algorithm refines the temperature scaling by considering the variance of the GP predictions. This allows for more exploration in regions where the model is uncertain and more exploitation in regions where the model is confident.

2.  **Robust Local Search:** The original EHBBO performs local search from `batch_size` starting points. The ATBO enhances this by performing multiple local searches from each starting point. This helps to escape local optima and find better solutions. A tiny change to the acquisition function during the local search phase encourages exploitation.

3.  **Sobol Initial Sampling**: Sobol sequences are known to have better space-filling properties than Latin Hypercube sampling, especially in higher dimensions. Using a Sobol sequence for initial sampling can lead to a better initial model and faster convergence.

4. **Kernel tuning:** The length_scale_bounds of the Matern kernel is tuned to (1e-2, 1e2) to allow for greater flexibility in the kernel.

These changes aim to improve the exploration-exploitation balance and enhance the robustness of the optimization process, leading to better performance on the BBOB test suite.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize

class ATBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * (dim + 1) # Number of initial samples, increased for higher dimensions
        self.gp = None
        self.best_y = float('inf')
        self.best_x = None

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.Sobol(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)  # Matern kernel
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X, xi=0.01):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)

        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)

        # Adaptive Thompson Sampling with dynamic temperature
        temperature = (1.0 / (1.0 + np.exp(- (self.best_y + 1e-9)))) * np.mean(sigma) # Temperature decreases as best_y decreases and sigma decreases
        samples = np.random.normal(mu, temperature * sigma)
        return samples.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Optimize the acquisition function using a local search strategy
        x_starts = self._sample_points(batch_size)
        x_next = []
        for x_start in x_starts:
            best_x = None
            best_acq = float('inf')
            for _ in range(3): # Multiple restarts for robustness
                res = minimize(lambda x: -self._acquisition_function(x.reshape(1, -1))[0,0] - 1e-3 * np.linalg.norm(x - x_start), # add penalty to encourage exploitation
                               x_start, 
                               bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)],
                               method='L-BFGS-B')
                if -self._acquisition_function(res.x.reshape(1, -1))[0,0] < best_acq:
                    best_acq = -self._acquisition_function(res.x.reshape(1, -1))[0,0]
                    best_x = res.x
                x_start = self._sample_points(1).flatten() # Re-initialize x_start
            x_next.append(best_x)
        return np.array(x_next)

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
        
        # Update best seen value
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]
    
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
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Select next points by acquisition function
            batch_size = min(self.budget - self.n_evals, max(1, self.dim // 2)) # Adaptive batch size
            X_next = self._select_next_points(batch_size)

            # Evaluate the points
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ATBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1405 with standard deviation 0.1095.

took 545.24 seconds to run.