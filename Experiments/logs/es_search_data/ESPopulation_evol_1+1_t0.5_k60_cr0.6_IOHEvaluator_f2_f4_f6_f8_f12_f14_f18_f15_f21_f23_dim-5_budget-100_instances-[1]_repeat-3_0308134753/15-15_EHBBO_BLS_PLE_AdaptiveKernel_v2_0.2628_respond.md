# Description
EHBBO_BLS_PLE_AdaptiveKernel_v2: Builds upon EHBBO_BLS_PLE_AdaptiveKernel by incorporating a more robust acquisition function that balances exploration and exploitation using a dynamic weighting scheme. It also introduces a more efficient local search strategy that adapts the search radius based on the GP's uncertainty. Finally, it includes a mechanism to re-evaluate the best point periodically to mitigate the effects of GP drift.

# Justification
The key improvements are:

1.  **Dynamic Acquisition Function:** The original algorithm used a simple exploration factor. This is replaced with a dynamic weighting between Expected Improvement (EI) and a Lower Confidence Bound (LCB) to better balance exploration and exploitation. The weight is adjusted based on the optimization progress.

2.  **Adaptive Local Search Radius:** The local search radius is now dynamically adjusted based on the GP's predicted uncertainty (sigma). This allows for more focused local search in regions where the GP is confident and broader search in regions where the GP is uncertain.

3.  **Periodic Best Point Re-evaluation:** The best point found so far is periodically re-evaluated using the true function. This helps to correct for GP drift and ensures that the algorithm doesn't get stuck in a false local optimum due to inaccuracies in the GP model.

4.  **Batch Evaluation:** Instead of evaluating one point at a time, the algorithm selects and evaluates a small batch of points. This can improve efficiency, especially in parallel environments.

These changes aim to improve the algorithm's ability to find the global optimum, especially in challenging black-box optimization problems. The dynamic acquisition function and adaptive local search radius enhance exploration and exploitation, while the periodic re-evaluation mitigates GP drift.

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
        self.batch_size = min(4, dim) # Evaluate a small batch of points
        self.reevaluation_interval = 10 * dim # Re-evaluate best point periodically

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
        # Implement Dynamic Acquisition Function: Weighted EI and LCB
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)  # avoid division by zero
        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        lcb = mu - self.exploration_factor * sigma # Lower Confidence Bound

        # Dynamic weighting between EI and LCB
        progress = min(1.0, self.n_evals / self.budget)
        weight = 0.5 + 0.5 * progress # Linearly increase weight of LCB
        acq = weight * ei + (1 - weight) * lcb
        return acq

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate by maximizing the acquisition function
        # Use a simple optimization strategy: randomly sample points and choose the best one
        candidates = self._sample_points(10 * batch_size)
        acq_values = self._acquisition_function(candidates)
        indices = np.argsort(acq_values)[-batch_size:] # Select top batch_size points
        return candidates[indices]

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

        for _ in range(num_restarts):
            x_start = x0 + np.random.normal(0, 0.1, size=self.dim) # Add some noise
            x_start = np.clip(x_start, self.bounds[0], self.bounds[1])

            def gp_objective(x):
                mu, _ = gp.predict(x.reshape(1, -1), return_std=True)
                return mu[0]

            # Adaptive local search radius
            _, sigma = gp.predict(x0.reshape(1, -1), return_std=True)
            search_radius = min(0.5 + sigma[0], 1.0) # Radius increases with uncertainty
            bounds = [(max(self.bounds[0][i], x0[i] - search_radius), min(self.bounds[1][i], x0[i] + search_radius)) for i in range(self.dim)]  # Smaller bounds
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
            if self.n_evals % self.kernel_update_interval == 0:
                self._fit_model(self.X, self.y)
            else:
                # Refit without kernel optimization for efficiency
                kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.kernel_length_scale, length_scale_bounds="fixed")
                self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-5)
                self.gp.fit(self.X, self.y)

            # Select next point(s) by acquisition function
            next_X = self._select_next_points(self.batch_size)

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

            # Periodic re-evaluation of the best point
            if self.n_evals % self.reevaluation_interval == 0 and self.n_evals < self.budget:
                best_y_reevaluated = func(self.best_x)
                self.n_evals += 1
                if best_y_reevaluated < self.best_y:
                    self.best_y = best_y_reevaluated


            # Decay exploration factor
            self.exploration_factor *= self.exploration_decay

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<EHBBO_BLS_PLE_AdaptiveKernel_v2>", line 141, in __call__
 141->             next_y = self._evaluate_points(func, next_X)
  File "<EHBBO_BLS_PLE_AdaptiveKernel_v2>", line 71, in _evaluate_points
  71->         y = np.array([func(x) for x in X])
  File "<EHBBO_BLS_PLE_AdaptiveKernel_v2>", line 71, in <listcomp>
  69 |         # func: takes array of shape (n_dims,) and returns np.float64.
  70 |         # return array of shape (n_points, 1)
  71->         y = np.array([func(x) for x in X])
  72 |         self.n_evals += len(X)
  73 |         return y.reshape(-1, 1)
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
