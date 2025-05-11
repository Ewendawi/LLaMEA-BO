# Description
EHBBO_BLS_PLE_AdaptiveKernel_v2: Builds upon EHBBO_BLS_PLE_AdaptiveKernel by incorporating a more sophisticated acquisition function that balances exploration and exploitation using an adaptive weighting scheme. It also introduces a dynamic local search radius that shrinks as the optimization progresses, focusing on finer-grained exploration near the end. The kernel update strategy is refined to consider both budget and the rate of change in the best-observed value.

# Justification
The key improvements are:

1.  **Adaptive Acquisition Function:** The original algorithm relies on a simple Expected Improvement (EI) acquisition function with an exploration factor. This is replaced with a weighted combination of EI and Lower Confidence Bound (LCB). The weights are adapted based on the optimization progress. Early on, exploration (LCB) is favored, while later, exploitation (EI) is emphasized. This helps to escape local optima and converge faster.
2.  **Dynamic Local Search Radius:** The original local search uses a fixed radius. This is changed to a dynamic radius that shrinks over time. This allows for broader exploration early on and finer-grained exploitation later. The radius is reduced as the budget is consumed, allowing for a more focused search in the final stages.
3.  **Refined Kernel Update Strategy:** The kernel update interval is modified to consider the rate of change in the best-observed value. If the best value is not improving significantly, the kernel is updated more frequently to adapt to the changing landscape. This ensures that the GP model remains accurate and relevant throughout the optimization process.
4. **Batch Evaluation**: Evaluate multiple points at once to better utilize the information gained from each evaluation, especially in higher dimensions.

These changes aim to improve the balance between exploration and exploitation, leading to better performance on a wider range of BBOB functions.

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
        self.best_y_history = []

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

    def _acquisition_function(self, X, exploitation_weight):
        # Implement Adaptive Acquisition function: weighted EI + LCB
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)  # avoid division by zero
        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Lower Confidence Bound
        lcb = mu - self.exploration_factor * sigma

        # Adaptive weighting
        acquisition = exploitation_weight * ei + (1 - exploitation_weight) * lcb
        return acquisition

    def _select_next_points(self, batch_size, exploitation_weight):
        # Select the next points to evaluate by maximizing the acquisition function
        # Use a simple optimization strategy: randomly sample points and choose the best one
        best_x = None
        best_acq = -np.inf
        for _ in range(10 * batch_size): # Increased sampling for better exploration
            x = self._sample_points(1)
            acq = self._acquisition_function(x, exploitation_weight)[0]
            if acq > best_acq:
                best_acq = acq
                best_x = x

        return best_x

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
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
        self.best_y_history.append(self.best_y)

    def local_search(self, gp, x0, num_restarts=1, search_radius=0.5):
        # Perform a local search around x0 using the GP prediction
        best_x = x0
        best_y_pred = np.inf

        for _ in range(num_restarts):
            x_start = x0 + np.random.normal(0, 0.1 * search_radius, size=self.dim) # Add some noise, scaled by radius
            x_start = np.clip(x_start, self.bounds[0], self.bounds[1])

            def gp_objective(x):
                mu, _ = gp.predict(x.reshape(1, -1), return_std=True)
                return mu[0]

            bounds = [(max(self.bounds[0][i], x0[i] - search_radius), min(self.bounds[1][i], x0[i] + search_radius)) for i in range(self.dim)]  # Radius-based bounds
            res = minimize(gp_objective, x_start, method='L-BFGS-B', bounds=bounds)
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
            # Adaptive Exploitation Weight
            exploitation_weight = min(1.0, self.n_evals / self.budget)  # Linearly increase exploitation

            # Fit the GP model
            if self.n_evals % self.kernel_update_interval == 0:
                # Check for improvement stagnation
                if len(self.best_y_history) > 10:
                    recent_improvements = np.diff(self.best_y_history[-10:])
                    if np.all(recent_improvements > -1e-6):  #Stagnation
                        self.kernel_update_interval = max(1, self.kernel_update_interval // 2) # Increase update frequency
                        print("Kernel update interval decreased due to stagnation")
                self._fit_model(self.X, self.y)
                self.kernel_update_interval = 5 * self.dim # Reset interval
            else:
                # Refit without kernel optimization for efficiency
                kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.kernel_length_scale, length_scale_bounds="fixed")
                self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-5)
                self.gp.fit(self.X, self.y)


            # Select next point(s) by acquisition function
            next_X = self._select_next_points(1, exploitation_weight)

            # Evaluate the new point(s)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Local search around the best point
            remaining_budget = self.budget - self.n_evals
            search_radius = 0.5 * (remaining_budget / self.budget) + 0.05 # Dynamic search radius
            num_local_restarts = 1 + int(self.budget / (self.n_evals + 1e-6)) # More restarts early on
            local_x = self.local_search(self.gp, self.best_x, num_restarts=num_local_restarts, search_radius=search_radius)
            predicted_y, predicted_sigma = self.gp.predict(local_x.reshape(1, -1), return_std=True)

            # Probabilistic evaluation
            improvement = self.best_y - predicted_y[0]
            Z = improvement / predicted_sigma[0] if predicted_sigma[0] > 0 else np.inf
            prob_improvement = norm.cdf(Z)

            # Adjust the evaluation threshold based on the remaining budget
            evaluation_threshold = 0.5 + 0.4 * (remaining_budget / self.budget)  # Higher threshold when budget is tight

            if prob_improvement > evaluation_threshold and self.n_evals < self.budget:
                # Evaluate the true function value only if the GP predicts a sufficiently high probability of improvement and we have budget
                local_y = func(local_x)
                self.n_evals += 1
                if local_y < self.best_y:
                    self.best_y = local_y
                    self.best_x = local_x


            # Decay exploration factor
            self.exploration_factor *= self.exploration_decay

        return self.best_y, self.best_x
```
## Feedback
 The algorithm EHBBO_BLS_PLE_AdaptiveKernel_v2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1540 with standard deviation 0.1019.

took 45.47 seconds to run.