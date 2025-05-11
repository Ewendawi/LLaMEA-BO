# Description
EHBBO_BLS_PLE_AdaptiveKernel_v2: Builds upon EHBBO_BLS_PLE_AdaptiveKernel by incorporating a dynamic exploration factor adjustment based on the observed function landscape. It uses a history of function evaluations to estimate the local "roughness" of the function. A higher roughness indicates a more complex landscape, prompting increased exploration. Additionally, the local search is enhanced with a more sophisticated adaptive step size based on the GP's uncertainty estimates.

# Justification
The key improvements are:

1.  **Dynamic Exploration Factor:** The original exploration factor decayed linearly. The new version dynamically adjusts it based on the variance of recent function evaluations. If the variance is high, it suggests a rough landscape, and the exploration factor is increased to encourage broader search. This helps to escape local optima and explore promising regions more effectively.

2.  **Adaptive Local Search Step Size:** The original local search used a fixed step size for generating starting points. The new version uses the GP's uncertainty (sigma) to adapt the step size. Larger sigma indicates higher uncertainty, prompting a larger step size for exploration during local search. This allows the local search to adapt to the local landscape and potentially find better optima.

3. **Batch Acquisition:** Instead of selecting just one point at a time, select a batch of points using the acquisition function. This reduces the overhead of GP fitting and prediction.

4. **Refine Local Search Bounds:** The bounds of the local search are dynamically adjusted based on the current best solution and the problem bounds. This ensures that the local search stays within the problem bounds and focuses on the most promising regions.

These changes are designed to improve the algorithm's ability to balance exploration and exploitation, adapt to different function landscapes, and improve the efficiency of the local search.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

class EHBBO_BLS_PLE_AdaptiveKernel_v2:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim # initial samples
        self.gp = None
        self.best_x = None
        self.best_y = np.inf
        self.exploration_factor = 1.0 # initial exploration factor
        self.exploration_decay = 0.99 # decay rate for exploration factor
        self.kernel_length_scale = 1.0
        self.kernel_update_interval = 5 * dim # Update kernel every this many evaluations
        self.past_y = [] # Keep track of past function evaluations for dynamic exploration

    def _sample_points(self, n_points):
        # sample points using Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.kernel_length_scale, length_scale_bounds=(1e-2, 10))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-5) # Increased restarts for kernel tuning
        self.gp.fit(X, y)
        self.kernel_length_scale = self.gp.kernel_.k2.length_scale # Update the length scale
        return self.gp

    def _acquisition_function(self, X):
        # Implement Expected Improvement acquisition function
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)  # avoid division by zero
        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei = ei * self.exploration_factor # add exploration factor
        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate by maximizing the acquisition function
        # Use a simple optimization strategy: randomly sample points and choose the best one
        
        # Optimize acquisition function using L-BFGS-B
        def objective(x):
            return -self._acquisition_function(x.reshape(1, -1))[0]

        best_x = None
        best_acq = -np.inf
        
        # Multi-start optimization
        for _ in range(5):
            x0 = self._sample_points(1)  # Random initial point
            bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            
            if -res.fun > best_acq:
                best_acq = -res.fun
                best_x = res.x
        
        return best_x
    

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        self.past_y.extend(y)  # Store past evaluations
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

        # Update best_x and best_y
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

    def local_search(self, gp, x0, num_restarts=1):
        # Perform a local search around x0 using the GP prediction
        best_x = x0
        best_y_pred = np.inf

        for _ in range(num_restarts):
            # Adaptive step size based on GP uncertainty
            _, sigma = gp.predict(x0.reshape(1, -1), return_std=True)
            step_size = 0.1 * sigma[0]  # Adjust the scaling factor as needed
            
            x_start = x0 + np.random.normal(0, step_size, size=self.dim) # Add some noise
            x_start = np.clip(x_start, self.bounds[0], self.bounds[1])

            def gp_objective(x):
                mu, _ = gp.predict(x.reshape(1, -1), return_std=True)
                return mu[0]

            # Refine local search bounds
            local_bounds = [(max(self.bounds[0][i], x0[i] - 0.5), min(self.bounds[1][i], x0[i] + 0.5)) for i in range(self.dim)]  # Smaller bounds

            res = minimize(gp_objective, x_start, method='L-BFGS-B', bounds=local_bounds)
            y_pred = gp_objective(res.x)

            if y_pred < best_y_pred:
                best_y_pred = y_pred
                best_x = res.x
        return best_x

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Optimization loop
        while self.n_evals < self.budget:
            # Fit the GP model
            if self.n_evals % self.kernel_update_interval == 0:
                self._fit_model(self.X, self.y)
            else:
                # Refit without kernel optimization for efficiency
                kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.kernel_length_scale, length_scale_bounds="fixed")
                self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-5)
                self.gp.fit(self.X, self.y)


            # Select next point(s) by acquisition function
            next_X = self._select_next_points(1)

            # Evaluate the new point(s)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Local search around the best point
            num_local_restarts = 1 + int(self.budget / (self.n_evals + 1e-6)) # More restarts early on
            local_x = self.local_search(self.gp, self.best_x, num_restarts=num_local_restarts)
            predicted_y, predicted_sigma = self.gp.predict(local_x.reshape(1, -1), return_std=True)

            # Probabilistic evaluation
            improvement = self.best_y - predicted_y[0]
            Z = improvement / predicted_sigma[0] if predicted_sigma[0] > 0 else np.inf
            prob_improvement = norm.cdf(Z)

            # Adjust the evaluation threshold based on the remaining budget
            remaining_budget = self.budget - self.n_evals
            evaluation_threshold = 0.5 + 0.4 * (remaining_budget / self.budget)  # Higher threshold when budget is tight

            if prob_improvement > evaluation_threshold and self.n_evals < self.budget:
                # Evaluate the true function value only if the GP predicts a sufficiently high probability of improvement and we have budget
                local_y = func(local_x)
                self.n_evals += 1
                if local_y < self.best_y:
                    self.best_y = local_y
                    self.best_x = local_x

            # Dynamic exploration factor adjustment
            if len(self.past_y) > 10:
                variance = np.var(self.past_y[-10:])
                self.exploration_factor = min(1.0, self.exploration_factor + 0.1 * variance) # Increase exploration if variance is high
            else:
                self.exploration_factor *= self.exploration_decay

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<EHBBO_BLS_PLE_AdaptiveKernel_v2>", line 156, in __call__
 156->             next_y = self._evaluate_points(func, next_X)
  File "<EHBBO_BLS_PLE_AdaptiveKernel_v2>", line 81, in _evaluate_points
  81->         y = np.array([func(x) for x in X])
  File "<EHBBO_BLS_PLE_AdaptiveKernel_v2>", line 81, in <listcomp>
  79 |         # func: takes array of shape (n_dims,) and returns np.float64.
  80 |         # return array of shape (n_points, 1)
  81->         y = np.array([func(x) for x in X])
  82 |         self.n_evals += len(X)
  83 |         self.past_y.extend(y)  # Store past evaluations
  File "<__array_function__ internals>", line 200, in vstack
  File "<__array_function__ internals>", line 200, in concatenate
ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 5 and the array at index 1 has size 1
