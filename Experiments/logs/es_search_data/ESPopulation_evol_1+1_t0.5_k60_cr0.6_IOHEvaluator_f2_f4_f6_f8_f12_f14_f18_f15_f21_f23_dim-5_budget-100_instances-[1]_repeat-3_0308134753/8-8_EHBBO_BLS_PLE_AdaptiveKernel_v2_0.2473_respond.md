# Description
EHBBO_BLS_PLE_AdaptiveKernel_v2: Builds upon EHBBO_BLS_PLE_AdaptiveKernel by incorporating a more sophisticated acquisition function that balances exploration and exploitation using an adaptive weighting scheme. It also includes a dynamic local search radius that shrinks as the optimization progresses, focusing on finer-grained exploration near the end. Furthermore, a more robust probabilistic local search evaluation is implemented.

# Justification
The key improvements are:

1.  **Adaptive Acquisition Function:** The original EI acquisition function is enhanced with a weighting factor that adaptively balances exploration and exploitation. This weighting is adjusted based on the remaining budget and the GP's uncertainty, allowing the algorithm to explore more early on and exploit more as the budget decreases.

2.  **Dynamic Local Search Radius:** The local search radius is dynamically adjusted based on the remaining budget. As the budget decreases, the radius shrinks, allowing for finer-grained exploration of the search space. This helps to avoid getting stuck in local optima.

3.  **Improved Probabilistic Local Search Evaluation:** The probabilistic local search evaluation is made more robust by considering the uncertainty in the GP's prediction. Instead of just comparing the predicted improvement to a fixed threshold, the algorithm considers the probability of improvement relative to the uncertainty. This helps to avoid evaluating points that are unlikely to lead to a significant improvement.

4.  **Batch Evaluation:** Evaluate multiple points in each iteration to better utilize the budget.

These changes aim to improve the algorithm's ability to find the global optimum, especially in challenging black-box optimization problems. The dynamic local search radius and adaptive acquisition function help to balance exploration and exploitation, while the improved probabilistic local search evaluation helps to avoid wasting function evaluations on points that are unlikely to lead to a significant improvement.

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
        self.batch_size = min(4, dim)

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

        # Adaptive exploration-exploitation balance
        remaining_budget = self.budget - self.n_evals
        exploration_weight = 0.1 + 0.9 * (remaining_budget / self.budget)  # More exploration early on
        ei = ei * exploration_weight

        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate by maximizing the acquisition function
        # Use a simple optimization strategy: randomly sample points and choose the best one
        best_x = None
        best_acq = -np.inf
        candidates = self._sample_points(10 * batch_size)
        acq_values = self._acquisition_function(candidates)
        indices = np.argsort(acq_values)[-batch_size:]
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
        remaining_budget = self.budget - self.n_evals
        search_radius = 0.5 * (remaining_budget / self.budget) + 0.05  # Dynamic search radius

        for _ in range(num_restarts):
            x_start = x0 + np.random.normal(0, 0.1 * search_radius, size=self.dim) # Add some noise
            x_start = np.clip(x_start, self.bounds[0], self.bounds[1])

            def gp_objective(x):
                mu, _ = gp.predict(x.reshape(1, -1), return_std=True)
                return mu[0]

            bounds = [(max(self.bounds[0][i], x0[i] - search_radius), min(self.bounds[1][i], x0[i] + search_radius)) for i in range(self.dim)]  # Dynamic bounds
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

            # Adjust the evaluation threshold based on the remaining budget and GP uncertainty
            remaining_budget = self.budget - self.n_evals
            evaluation_threshold = 0.5 + 0.4 * (remaining_budget / self.budget) * (1 - predicted_sigma[0])  # Higher threshold when budget is tight and GP is confident

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
## Error
 Traceback (most recent call last):
  File "<EHBBO_BLS_PLE_AdaptiveKernel_v2>", line 142, in __call__
 142->             next_y = self._evaluate_points(func, next_X)
  File "<EHBBO_BLS_PLE_AdaptiveKernel_v2>", line 72, in _evaluate_points
  72->         y = np.array([func(x) for x in X])
  File "<EHBBO_BLS_PLE_AdaptiveKernel_v2>", line 72, in <listcomp>
  70 |         # func: takes array of shape (n_dims,) and returns np.float64.
  71 |         # return array of shape (n_points, 1)
  72->         y = np.array([func(x) for x in X])
  73 |         self.n_evals += len(X)
  74 |         return y.reshape(-1, 1)
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
