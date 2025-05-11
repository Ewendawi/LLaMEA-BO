# Description
EHBBO_BLS_PLE_AdaptiveKernel_v4: This algorithm builds upon EHBBO_BLS_PLE_AdaptiveKernel_v3 by incorporating a more robust local search strategy using CMA-ES, replacing L-BFGS-B. It also introduces a dynamic adjustment of the exploration-exploitation trade-off (kappa) based on the function's landscape, adapting to both local and global search phases. Furthermore, the kernel update frequency is made adaptive based on the rate of change of the best-found value, allowing for more frequent updates when the function landscape is changing rapidly and less frequent updates when convergence is slow.

# Justification
1.  **CMA-ES for Local Search:** CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a more robust and effective local search algorithm compared to L-BFGS-B, particularly in high-dimensional and non-convex landscapes. Replacing L-BFGS-B with CMA-ES in the `local_search` function can lead to better exploitation of promising regions.
2.  **Adaptive Kappa:** Dynamically adjusting the exploration-exploitation parameter `kappa` allows the algorithm to adapt to different phases of the optimization. A larger `kappa` encourages exploration early on, while a smaller `kappa` promotes exploitation as the algorithm converges. The proposed update adjusts `kappa` based on the variance of the GP predictions. If the variance is high, the algorithm increases `kappa` to explore more. If the variance is low, the algorithm decreases `kappa` to exploit the current best region.
3.  **Adaptive Kernel Update Frequency:** The frequency of kernel updates is crucial for balancing model accuracy and computational cost. Updating the kernel too frequently can be computationally expensive, while updating it too infrequently can lead to a poorly fitted GP model. By adapting the kernel update frequency based on the rate of change of the best-found value, the algorithm can dynamically adjust the update frequency to maintain a good balance between accuracy and cost. This is implemented by tracking the recent best values and updating the kernel more frequently if the best value is changing rapidly.
4.  **Simplified Initial Sampling Adjustment:** The logic for adjusting the initial sampling size has been simplified.
5. **Early Stopping**: Add an early stopping condition based on the lack of improvement.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
import cma

class EHBBO_BLS_PLE_AdaptiveKernel_v4:
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
        self.kappa = 2.0  # UCB exploration-exploitation parameter
        self.kappa_decay = 0.995
        self.min_kappa = 0.1
        self.initial_sample_multiplier = 1.0
        self.best_y_history = []
        self.kernel_update_counter = 0
        self.no_improvement_streak = 0
        self.max_no_improvement = 50

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
        # Implement Upper Confidence Bound acquisition function
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu - self.kappa * sigma

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate by maximizing the acquisition function
        # Use a multi-start local search strategy
        best_x = None
        best_acq = np.inf
        num_starts = 5 * batch_size # Increased number of starts
        for _ in range(num_starts):
            x_start = self._sample_points(1)
            x_start = x_start.flatten()

            def acquisition_objective(x):
                return self._acquisition_function(x.reshape(1, -1))[0]

            res = minimize(acquisition_objective, x_start, method='L-BFGS-B', bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)])
            acq = res.fun

            if acq < best_acq:
                best_acq = acq
                best_x = res.x

        return best_x.reshape(1, -1)

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
            self.no_improvement_streak = 0
        else:
            self.no_improvement_streak += 1

    def local_search(self, gp, x0, sigma):
        # Perform a local search around x0 using CMA-ES
        def gp_objective(x):
            mu, _ = gp.predict(x.reshape(1, -1), return_std=True)
            return mu[0]

        es = cma.purecma.CMAES(x0, sigma, bounds=[self.bounds[0], self.bounds[1]])
        options = {'verb_disp':0, 'maxiter': 50}
        es.optimize(gp_objective, options=options)
        return es.best.x

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial exploration
        n_init_dynamic = int(self.initial_sample_multiplier * self.n_init)
        initial_X = self._sample_points(n_init_dynamic)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        self.best_y_history.append(self.best_y)

        # Optimization loop
        while self.n_evals < self.budget and self.no_improvement_streak < self.max_no_improvement:
            # Fit the GP model
            if self.kernel_update_counter >= self.kernel_update_interval:
                self._fit_model(self.X, self.y)
                self.kernel_update_counter = 0
                self.kernel_update_interval = 5 * self.dim
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
            self.best_y_history.append(self.best_y)

            # Local search around the best point
            predicted_y, predicted_sigma = self.gp.predict(self.best_x.reshape(1, -1), return_std=True)
            local_x = self.local_search(self.gp, self.best_x, predicted_sigma[0])
            predicted_y_local, predicted_sigma_local = self.gp.predict(local_x.reshape(1, -1), return_std=True)


            # Probabilistic evaluation
            improvement = self.best_y - predicted_y_local[0]
            Z = improvement / predicted_sigma_local[0] if predicted_sigma_local[0] > 0 else np.inf
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
                    self.no_improvement_streak = 0
            else:
                self.no_improvement_streak += 1


            # Decay exploration factor and kappa
            self.exploration_factor *= self.exploration_decay
            # Adaptive Kappa update
            if len(self.y) > 1:
                variance = np.var(self.gp.predict(self.X, return_std=True)[1])
                if variance > 0.1:
                    self.kappa = min(2.0, self.kappa * 1.05)  # Increase exploration
                else:
                    self.kappa = max(self.min_kappa, self.kappa * 0.95)  # Increase exploitation


            #Adaptive Kernel Update
            if len(self.best_y_history) > 2:
                rate_of_change = abs(self.best_y_history[-1] - self.best_y_history[-2])
                if rate_of_change > 1e-3:
                    self.kernel_update_interval = max(1, int(self.kernel_update_interval * 0.8)) # Update more frequently if changing rapidly
                else:
                    self.kernel_update_interval = min(5 * self.dim, int(self.kernel_update_interval * 1.2)) # Update less frequently if converging

            self.kernel_update_counter += 1

            # Adjust initial sampling size
            if self.n_evals > self.n_init and abs(self.y[-1] - self.y[-self.n_init]) < 1e-3:
                self.initial_sample_multiplier = min(2.0, self.initial_sample_multiplier * 1.05)
            else:
                self.initial_sample_multiplier = max(1.0, self.initial_sample_multiplier * 0.95)

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<EHBBO_BLS_PLE_AdaptiveKernel_v4>", line 152, in __call__
 152->             local_x = self.local_search(self.gp, self.best_x, predicted_sigma[0])
  File "<EHBBO_BLS_PLE_AdaptiveKernel_v4>", line 110, in local_search
 108 |             return mu[0]
 109 | 
 110->         es = cma.purecma.CMAES(x0, sigma, bounds=[self.bounds[0], self.bounds[1]])
 111 |         options = {'verb_disp':0, 'maxiter': 50}
 112 |         es.optimize(gp_objective, options=options)
TypeError: CMAES.__init__() got an unexpected keyword argument 'bounds'
