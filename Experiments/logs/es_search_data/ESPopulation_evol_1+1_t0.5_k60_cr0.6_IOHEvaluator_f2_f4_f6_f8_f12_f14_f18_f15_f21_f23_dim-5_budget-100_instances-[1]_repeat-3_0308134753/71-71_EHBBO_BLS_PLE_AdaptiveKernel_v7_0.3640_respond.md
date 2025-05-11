# Description
EHBBO_BLS_PLE_AdaptiveKernel_v7: This algorithm builds upon EHBBO_BLS_PLE_AdaptiveKernel_v6, focusing on improving the efficiency and effectiveness of the local search and acquisition function. Key improvements include: 1) Implementing a more efficient local search strategy by reducing the number of local search restarts and adaptively adjusting the step size based on the function landscape and GP uncertainty. 2) Introducing a dynamic adjustment of the exploration-exploitation trade-off based on the diversity of the sampled points. 3) Adding a mechanism to detect and escape local optima by re-initializing the GP model with samples generated from a mixture of uniform sampling and sampling around the current best point. 4) Implementing a more computationally efficient kernel update strategy by using a moving average of the kernel length scale.

# Justification
The key components of the algorithm are justified as follows:
- Efficient Local Search: Reducing the number of local search restarts and adaptively adjusting the step size helps to reduce the computational cost of the local search while maintaining its effectiveness.
- Dynamic Exploration-Exploitation Trade-off: Adjusting the exploration-exploitation trade-off based on the diversity of the sampled points helps to balance exploration and exploitation, leading to better performance.
- Local Optima Escape: Re-initializing the GP model with samples generated from a mixture of uniform sampling and sampling around the current best point helps to escape local optima.
- Efficient Kernel Update: Using a moving average of the kernel length scale helps to reduce the computational cost of the kernel update while maintaining its accuracy.

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

class EHBBO_BLS_PLE_AdaptiveKernel_v7:
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
        self.kernel_length_scale_ema = 1.0 # Exponential moving average of kernel length scale
        self.kernel_length_scale_ema_alpha = 0.1 # EMA smoothing factor
        self.kernel_update_interval = 5 * dim # Update kernel every this many evaluations
        self.kappa = 2.0  # UCB exploration-exploitation parameter
        self.kappa_decay = 0.995
        self.min_kappa = 0.1
        self.initial_sample_multiplier = 1.0
        self.ei_weight = 0.5 # Weight for Expected Improvement
        self.ei_weight_decay = 0.99
        self.min_ei_weight = 0.1 # Minimum EI weight
        self.min_length_scale = 1e-3
        self.temperature = 1.0 # Temperature parameter for acquisition function
        self.temperature_decay = 0.995 # Decay rate for temperature
        self.ucb_ei_weight = 0.5 # Weight for UCB and EI
        self.ucb_ei_weight_decay = 0.99
        self.min_ucb_ei_weight = 0.1
        self.local_optima_threshold = 1e-4
        self.local_optima_count = 0
        self.local_optima_reset_interval = 10 * dim
        self.momentum = 0.1
        self.diversity_threshold = 0.1 # Threshold for diversity check
        self.diversity_bonus = 0.1 # Bonus for diverse samples

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
        self.kernel_length_scale_ema = self.kernel_length_scale_ema_alpha * self.kernel_length_scale + (1 - self.kernel_length_scale_ema_alpha) * self.kernel_length_scale_ema # Update EMA
        return self.gp

    def _acquisition_function(self, X):
        # Implement Upper Confidence Bound acquisition function
        mu, sigma = self.gp.predict(X, return_std=True)
        ucb = mu - self.kappa * sigma

        # Calculate Expected Improvement
        improvement = self.best_y - mu
        Z = improvement / sigma if sigma > 0 else np.inf
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Weighted combination of UCB and EI
        ucb_ei_w = max(self.ucb_ei_weight, self.min_ucb_ei_weight)
        acquisition = (ucb_ei_w * ei + (1 - ucb_ei_w) * ucb) / self.temperature

        # Diversity bonus
        if self.X is not None and len(self.X) > 0:
            min_dist = np.min(np.linalg.norm(X - self.X, axis=1))
            if min_dist > self.diversity_threshold:
                acquisition += self.diversity_bonus

        return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate by maximizing the acquisition function
        # Use a multi-start local search strategy
        best_x = None
        best_acq = np.inf
        num_starts = 3 * batch_size # Reduced number of starts
        for _ in range(num_starts):
            x_start = self._sample_points(1)
            x_start = x_start.flatten()

            def acquisition_objective(x):
                return -self._acquisition_function(x.reshape(1, -1))[0] # Negative sign for minimization

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

    def local_search(self, gp, x0, sigma, num_restarts=1):
        # Perform a local search around x0 using the GP prediction
        best_x = x0
        best_y_pred = np.inf
        search_radius = 0.2 + sigma # Dynamic search radius
        step_size = 0.1
        velocity = np.zeros_like(x0)

        # Calculate gradient of GP prediction
        def gp_objective(x):
            mu, _ = gp.predict(x.reshape(1, -1), return_std=True)
            return mu[0]

        # Adaptive search radius based on gradient
        for _ in range(num_restarts):
            x_start = x0 + np.random.normal(0, 0.05, size=self.dim) # Add some noise
            x_start = np.clip(x_start, self.bounds[0], self.bounds[1])
            x = x_start.copy()

            for i in range(5): # Reduced number of iterations
                # Calculate gradient
                mu, sigma_matrix = gp.predict(x.reshape(1, -1), return_std=True)
                try:
                    gradient = np.linalg.solve(sigma_matrix.reshape(1,1), (mu - gp_objective(x)) * (x - x0))
                except:
                    gradient = np.zeros_like(x)
                # Update velocity with momentum
                velocity = self.momentum * velocity - step_size * gradient.flatten()
                # Update position
                x = x + velocity
                x = np.clip(x, self.bounds[0], self.bounds[1])

                y_pred = gp_objective(x)

                if y_pred < best_y_pred:
                    best_y_pred = y_pred
                    best_x = x.copy()

                # Adaptive step size
                if i > 0 and y_pred > gp_objective(x - velocity):
                    step_size *= 0.5
                else:
                    step_size *= 1.05

                # Adapt step size based on GP uncertainty
                step_size *= (1 + sigma)

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
            # Fit the GP model
            if self.n_evals % self.kernel_update_interval == 0:
                self._fit_model(self.X, self.y)
            else:
                # Refit without kernel optimization for efficiency
                kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.kernel_length_scale_ema, length_scale_bounds="fixed")
                self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-5)
                self.gp.fit(self.X, self.y)


            # Select next point(s) by acquisition function
            next_X = self._select_next_points(1)

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
            self.ei_weight = max(self.min_ei_weight, self.ei_weight * self.ei_weight_decay)
            self.temperature *= self.temperature_decay
            self.ucb_ei_weight = max(self.min_ucb_ei_weight, self.ucb_ei_weight * self.ucb_ei_weight_decay)

            #Adaptive Kernel Update
            if len(self.y) > 1:
                improvement_ratio = abs(self.y[-1] - self.y[-2]) / abs(self.y[-2]) if abs(self.y[-2]) > 1e-6 else 0
                _, predicted_sigma = self.gp.predict(self.X[-1].reshape(1, -1), return_std=True)
                uncertainty = predicted_sigma[0]

                if improvement_ratio > 0.1 or uncertainty > 0.5:
                    self._fit_model(self.X, self.y)
                    self.kernel_update_interval = 5 * self.dim #reset update interval
                else:
                    self.kernel_update_interval = max(1, int(self.kernel_update_interval * 0.9)) #reduce update interval gradually

            # Adjust initial sampling size
            if self.n_evals > self.n_init and abs(self.y[-1] - self.y[-self.n_init]) < 1e-3:
                self.initial_sample_multiplier *= 1.1
                self.local_optima_count += 1
            else:
                self.initial_sample_multiplier = max(1.0, self.initial_sample_multiplier * 0.95)
                self.local_optima_count = 0

            # Local optima detection and escape
            if self.local_optima_count > self.local_optima_reset_interval:
                # Re-initialize GP model with diverse samples
                # Mixture of uniform sampling and sampling around the current best point
                n_uniform = int(0.5 * self.n_init)
                new_X_uniform = self._sample_points(n_uniform)
                new_X_around_best = np.random.normal(self.best_x, 0.1, size=(self.n_init - n_uniform, self.dim))
                new_X_around_best = np.clip(new_X_around_best, self.bounds[0], self.bounds[1])
                new_X = np.vstack((new_X_uniform, new_X_around_best))

                new_y = self._evaluate_points(func, new_X)
                self._update_eval_points(new_X, new_y)
                self._fit_model(self.X, self.y)
                self.local_optima_count = 0

        return self.best_y, self.best_x
```
## Feedback
 The algorithm EHBBO_BLS_PLE_AdaptiveKernel_v7 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1375 with standard deviation 0.0996.

took 422.22 seconds to run.