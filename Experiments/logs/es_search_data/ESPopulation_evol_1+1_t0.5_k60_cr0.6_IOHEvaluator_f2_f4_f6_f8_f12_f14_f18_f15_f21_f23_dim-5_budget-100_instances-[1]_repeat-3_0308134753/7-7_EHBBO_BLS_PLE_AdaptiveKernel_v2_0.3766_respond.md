# Description
EHBBO_BLS_PLE_AdaptiveKernel_v2: Builds upon the previous EHBBO_BLS_PLE_AdaptiveKernel by incorporating a more sophisticated acquisition function balancing exploration and exploitation, and by adaptively adjusting the local search radius based on the optimization progress. The acquisition function now uses a weighted combination of Expected Improvement (EI) and Upper Confidence Bound (UCB). The local search radius shrinks as the optimization progresses, focusing on finer-grained exploration near the best-known solution later in the optimization.

# Justification
The key improvements focus on refining the exploration-exploitation balance and local search strategy.

1.  **Weighted EI-UCB Acquisition:** Combining EI and UCB allows for a more flexible control over exploration and exploitation. EI favors regions with high potential improvement, while UCB encourages exploration in areas with high uncertainty. The weights are adaptively adjusted to shift the focus from exploration to exploitation over time.

2.  **Adaptive Local Search Radius:** Starting with a larger local search radius allows for broader exploration initially. As the optimization progresses and the algorithm converges towards a promising region, the search radius is reduced to refine the solution. This approach helps to avoid getting stuck in local optima early on and allows for finer-grained optimization later.

3.  **Budget-Aware Exploration Decay:** The exploration decay is made budget-aware, ensuring that exploration is prioritized when the budget is abundant and exploitation is emphasized when the budget is scarce. This is achieved by linking the decay rate to the remaining budget.

4. **Simplified Kernel Refitting:** The kernel refitting strategy is simplified to reduce computational overhead. Instead of refitting the kernel every `kernel_update_interval`, it's only refitted if the number of evaluations is a multiple of `kernel_update_interval` and the budget is not almost exhausted.

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
        self.ei_weight = 0.5 # Weight for Expected Improvement in acquisition
        self.ucb_weight = 0.5 # Weight for Upper Confidence Bound in acquisition


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
        # Implement a weighted combination of Expected Improvement and UCB
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)  # avoid division by zero
        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ucb = mu + self.exploration_factor * sigma  # Upper Confidence Bound

        # Adaptive weights for EI and UCB
        remaining_budget = self.budget - self.n_evals
        exploitation_factor = 1 - (remaining_budget / self.budget)
        self.ei_weight = 0.5 + 0.5 * exploitation_factor
        self.ucb_weight = 1 - self.ei_weight

        acq = self.ei_weight * ei + self.ucb_weight * ucb
        return acq

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate by maximizing the acquisition function
        # Use a simple optimization strategy: randomly sample points and choose the best one
        best_x = None
        best_acq = -np.inf
        for _ in range(10 * batch_size): # Increased sampling for better exploration
            x = self._sample_points(1)
            acq = self._acquisition_function(x)[0]
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

    def local_search(self, gp, x0, num_restarts=1):
        # Perform a local search around x0 using the GP prediction
        best_x = x0
        best_y_pred = np.inf

        # Adaptive local search radius
        remaining_budget = self.budget - self.n_evals
        search_radius = 0.5 + 4.5 * (remaining_budget / self.budget) # Radius shrinks as budget decreases

        for _ in range(num_restarts):
            x_start = x0 + np.random.normal(0, 0.1, size=self.dim) # Add some noise
            x_start = np.clip(x_start, self.bounds[0], self.bounds[1])

            def gp_objective(x):
                mu, _ = gp.predict(x.reshape(1, -1), return_std=True)
                return mu[0]

            bounds = [(max(self.bounds[0][i], x0[i] - search_radius), min(self.bounds[1][i], x0[i] + search_radius)) for i in range(self.dim)]  # Adaptive bounds
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
            # Fit the GP model
            if self.n_evals % self.kernel_update_interval == 0 and self.n_evals < 0.95 * self.budget:
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


            # Decay exploration factor
            self.exploration_factor *= self.exploration_decay

        return self.best_y, self.best_x
```
## Feedback
 The algorithm EHBBO_BLS_PLE_AdaptiveKernel_v2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1524 with standard deviation 0.0969.

took 86.17 seconds to run.