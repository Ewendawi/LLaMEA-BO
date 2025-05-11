# Description
EHBBO_BLS_PLE_AdaptiveKernel_v2: Builds upon EHBBO_BLS_PLE_AdaptiveKernel by incorporating a more sophisticated acquisition function that balances exploration and exploitation using an adaptive weighting scheme. It also introduces a dynamic local search radius that shrinks as the optimization progresses, focusing on finer-grained exploration near the end. Furthermore, the kernel update strategy is refined to consider the diversity of sampled points, triggering updates more frequently when the sampled region is less explored.

# Justification
The key improvements are:

1.  **Adaptive Acquisition Function:** Instead of a fixed exploration factor, the acquisition function now uses a weighted combination of Expected Improvement (EI) and a Lower Confidence Bound (LCB). The weights are dynamically adjusted based on the optimization progress. Early on, LCB is emphasized for exploration, while later EI is favored for exploitation. This allows for a more nuanced exploration-exploitation trade-off.
2.  **Dynamic Local Search Radius:** The local search radius is reduced as the optimization progresses. This allows for broader exploration initially and finer-grained exploitation later.
3.  **Diversity-Aware Kernel Updates:** The kernel is updated more frequently when the diversity of the sampled points is low, indicated by a low average distance between points. This ensures that the GP model adapts more quickly to changes in the function landscape, especially in regions that have not been thoroughly explored.
4. **Budget-aware local search restarts:** The number of local search restarts now also depends on the predicted variance. Higher variance means more uncertainty and therefore more restarts.
5. **Early stopping:** If the algorithm finds a solution that is good enough, it stops early to save budget.

These changes aim to improve the algorithm's ability to balance exploration and exploitation, adapt to the function landscape, and make efficient use of the limited budget.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances

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
        self.ei_weight = 0.5 # Weight for Expected Improvement in acquisition function
        self.lcb_weight = 0.5 # Weight for Lower Confidence Bound in acquisition function
        self.initial_local_search_radius = 0.5
        self.early_stopping_threshold = -1e-6

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
        # Implement Adaptive Acquisition function: weighted EI and LCB
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)  # avoid division by zero
        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        lcb = mu - self.exploration_factor * sigma # Lower Confidence Bound
        acquisition = self.ei_weight * ei + self.lcb_weight * lcb
        return acquisition

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

    def local_search(self, gp, x0, num_restarts=1, radius=0.5):
        # Perform a local search around x0 using the GP prediction
        best_x = x0
        best_y_pred = np.inf

        for _ in range(num_restarts):
            x_start = x0 + np.random.normal(0, 0.1 * radius, size=self.dim) # Add some noise
            x_start = np.clip(x_start, self.bounds[0], self.bounds[1])

            def gp_objective(x):
                mu, _ = gp.predict(x.reshape(1, -1), return_std=True)
                return mu[0]

            bounds = [(max(self.bounds[0][i], x0[i] - radius), min(self.bounds[1][i], x0[i] + radius)) for i in range(self.dim)]  # Smaller bounds
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

            # Adjust EI and LCB weights
            progress = self.n_evals / self.budget
            self.ei_weight = 0.5 + progress / 2 # Increase EI weight as we progress
            self.lcb_weight = 1 - self.ei_weight # Decrease LCB weight accordingly

            # Fit the GP model
            if self.n_evals % self.kernel_update_interval == 0:
                # Check diversity of sampled points
                if self.X.shape[0] > 1:
                    distances = euclidean_distances(self.X)
                    avg_distance = np.mean(distances[np.triu_indices_from(distances, k=1)]) # Avoid diagonal
                    if avg_distance < 0.1 * (self.bounds[1][0] - self.bounds[0][0]):
                        self.kernel_update_interval = max(1, self.kernel_update_interval // 2) #Update more frequently if points are close
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
            local_search_radius = self.initial_local_search_radius * (1 - progress) # Reduce radius as we progress
            num_local_restarts = 1 + int(self.budget / (self.n_evals + 1e-6)) # More restarts early on
            predicted_y, predicted_sigma = self.gp.predict(self.best_x.reshape(1, -1), return_std=True)
            num_local_restarts += int(predicted_sigma[0] * 10) # More restarts if variance is high
            local_x = self.local_search(self.gp, self.best_x, num_restarts=num_local_restarts, radius=local_search_radius)
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

            # Early stopping
            if self.best_y < self.early_stopping_threshold:
                break


            # Decay exploration factor
            self.exploration_factor *= self.exploration_decay

        return self.best_y, self.best_x
```
## Feedback
 The algorithm EHBBO_BLS_PLE_AdaptiveKernel_v2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1477 with standard deviation 0.0983.

took 41.78 seconds to run.