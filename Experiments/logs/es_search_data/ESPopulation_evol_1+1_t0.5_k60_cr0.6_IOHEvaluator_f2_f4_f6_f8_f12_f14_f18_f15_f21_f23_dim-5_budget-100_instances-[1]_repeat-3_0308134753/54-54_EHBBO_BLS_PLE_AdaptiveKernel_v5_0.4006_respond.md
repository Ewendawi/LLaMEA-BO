# Description
EHBBO_BLS_PLE_AdaptiveKernel_v5: This algorithm builds upon EHBBO_BLS_PLE_AdaptiveKernel_v4, further refining the balance between exploration and exploitation, and improving the efficiency of kernel updates. Key improvements include: 1) Introducing a dynamic lower bound for the UCB exploration parameter (kappa) based on the uncertainty of the GP. 2) Implementing a more adaptive kernel update strategy based on the change in the best observed value. 3) Adding a mechanism to periodically re-evaluate the best point to mitigate the impact of GP inaccuracies. 4) Introduce a batch size > 1 for selecting next points to evaluate.

# Justification
The key improvements in this version aim to address the limitations of the previous version by:

1.  **Dynamic Kappa Lower Bound:** The previous version had a fixed minimum value for kappa, which might be too aggressive in the later stages of optimization when the GP is more confident. By making the lower bound dynamic and dependent on the GP's uncertainty, we can ensure sufficient exploration even when the GP is confident.
2.  **Adaptive Kernel Updates based on Best Value Change:** The previous kernel update strategy was based on the last two evaluations, which could be noisy. This version uses the change in the best observed value to trigger kernel updates, ensuring that the kernel is updated when the optimization is making significant progress.
3.  **Periodic Best Point Re-evaluation:** The GP's predictions can become inaccurate over time, especially in high-dimensional spaces. Periodically re-evaluating the best point helps to correct for these inaccuracies and ensure that the optimization is not stuck in a local optimum due to a faulty GP prediction.
4. **Batch Size > 1**: Evaluating multiple points in parallel can improve the efficiency of the algorithm, especially when the function evaluation is expensive.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import norm

class EHBBO_BLS_PLE_AdaptiveKernel_v5:
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
        self.ei_weight = 0.5 # Weight for Expected Improvement
        self.ei_weight_decay = 0.99
        self.min_length_scale = 1e-3
        self.best_reval_interval = 10 * dim
        self.last_best_reval = 0
        self.batch_size = min(4, dim)

    def _sample_points(self, n_points):
        # sample points using Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.kernel_length_scale, length_scale_bounds=(self.min_length_scale, 10))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-5) # Increased restarts for kernel tuning
        self.gp.fit(X, y)
        self.kernel_length_scale = self.gp.kernel_.k2.length_scale # Update the length scale
        return self.gp

    def _acquisition_function(self, X):
        # Implement Upper Confidence Bound acquisition function
        mu, sigma = self.gp.predict(X, return_std=True)
        # Dynamic min_kappa based on uncertainty
        min_kappa_dynamic = self.min_kappa + 0.5 * np.mean(sigma)
        kappa = max(min_kappa_dynamic, self.kappa)
        ucb = mu - kappa * sigma

        # Calculate Expected Improvement
        improvement = self.best_y - mu
        Z = improvement / sigma if sigma > 0 else np.inf
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Weighted combination of UCB and EI
        return self.ei_weight * ei + (1 - self.ei_weight) * ucb

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate by maximizing the acquisition function
        # Use a multi-start local search strategy
        best_x_batch = []
        best_acq_batch = []
        for _ in range(batch_size):
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
            best_x_batch.append(best_x.reshape(1, -1))
            best_acq_batch.append(best_acq)

        return np.concatenate(best_x_batch, axis=0)

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

    def local_search(self, gp, x0, sigma, num_restarts=1):
        # Perform a local search around x0 using the GP prediction
        best_x = x0
        best_y_pred = np.inf
        search_radius = 0.2 + sigma # Dynamic search radius

        # Calculate gradient of GP prediction
        def gp_objective(x):
            mu, _ = gp.predict(x.reshape(1, -1), return_std=True)
            return mu[0]

        def gp_objective_grad(x):
            return gp.predict(x.reshape(1, -1), return_std=True)[1]

        # Adaptive search radius based on gradient
        #try:
        #    gradient = gp_objective_grad(x0)
        #    gradient_norm = np.linalg.norm(gradient)
        #    search_radius = 0.2 + sigma + 0.1 * gradient_norm # Increase radius if gradient is high
        #except:
        #    pass


        for _ in range(num_restarts):
            x_start = x0 + np.random.normal(0, 0.1, size=self.dim) # Add some noise
            x_start = np.clip(x_start, self.bounds[0], self.bounds[1])


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
        n_init_dynamic = int(self.initial_sample_multiplier * self.n_init)
        initial_X = self._sample_points(n_init_dynamic)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Optimization loop
        while self.n_evals < self.budget:

            #Re-evaluate best point periodically
            if self.n_evals - self.last_best_reval >= self.best_reval_interval:
                best_y_reval = func(self.best_x)
                self.n_evals += 1
                if best_y_reval < self.best_y:
                    self.best_y = best_y_reval
                self.last_best_reval = self.n_evals

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
            predicted_y, predicted_sigma = self.gp.predict(self.best_x.reshape(1, -1), return_std=True)
            num_local_restarts = 1 + int(self.budget / (self.n_evals + 1e-6)) # More restarts early on
            local_x = self.local_search(self.gp, self.best_x, predicted_sigma[0], num_restarts=num_local_restarts)
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

            # Decay exploration factor and kappa
            self.exploration_factor *= self.exploration_decay
            self.kappa = max(self.min_kappa, self.kappa * self.kappa_decay)
            self.ei_weight *= self.ei_weight_decay

            #Adaptive Kernel Update based on best value change
            if len(self.y) > 1:
                best_improvement_ratio = abs(self.best_y - self.y[-1]) / abs(self.best_y) if abs(self.best_y) > 1e-6 else 0
                if best_improvement_ratio > 0.05:
                    self._fit_model(self.X, self.y)
                    self.kernel_update_interval = 5 * self.dim #reset update interval
                else:
                    self.kernel_update_interval = max(1, int(self.kernel_update_interval * 0.9)) #reduce update interval gradually

            # Adjust initial sampling size
            if self.n_evals > self.n_init and abs(self.y[-1] - self.y[-self.n_init]) < 1e-3:
                self.initial_sample_multiplier *= 1.1
            else:
                self.initial_sample_multiplier = max(1.0, self.initial_sample_multiplier * 0.95)

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<EHBBO_BLS_PLE_AdaptiveKernel_v5>", line 193, in __call__
 193->             next_y = self._evaluate_points(func, next_X)
  File "<EHBBO_BLS_PLE_AdaptiveKernel_v5>", line 100, in _evaluate_points
 100->         y = np.array([func(x) for x in X])
  File "<EHBBO_BLS_PLE_AdaptiveKernel_v5>", line 100, in <listcomp>
  98 |         # func: takes array of shape (n_dims,) and returns np.float64.
  99 |         # return array of shape (n_points, 1)
 100->         y = np.array([func(x) for x in X])
 101 |         self.n_evals += len(X)
 102 |         return y.reshape(-1, 1)
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
