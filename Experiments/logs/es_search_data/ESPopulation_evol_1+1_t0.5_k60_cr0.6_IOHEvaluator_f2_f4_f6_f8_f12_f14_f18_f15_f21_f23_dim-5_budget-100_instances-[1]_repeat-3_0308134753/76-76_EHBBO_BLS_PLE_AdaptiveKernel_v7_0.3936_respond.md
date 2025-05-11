# Description
EHBBO_BLS_PLE_AdaptiveKernel_v7: This algorithm refines EHBBO_BLS_PLE_AdaptiveKernel_v6 by focusing on enhancing the exploration-exploitation balance and improving the efficiency of the local search. Key improvements include: 1) Introducing a dynamic batch size for selecting next points based on the optimization progress. 2) Implementing a more efficient local search strategy by using a combination of gradient-based and random search. 3) Refining the adaptive kernel update strategy by considering the gradient norm of the GP prediction. 4) Adding a mechanism to re-initialize the GP model with samples clustered around the current best point to intensify the search in promising regions.

# Justification
1.  **Dynamic Batch Size:** Adjusting the batch size allows for more exploration early on and more exploitation later in the optimization process. This is achieved by increasing the batch size as the budget decreases.
2.  **Efficient Local Search:** Combining gradient-based and random search in the local search strategy improves the efficiency of the search by leveraging the GP's gradient information while also exploring the neighborhood of the best point.
3.  **Refined Adaptive Kernel Update:** Incorporating the gradient norm into the kernel update strategy allows for a more informed decision about when to update the kernel. A large gradient norm indicates a rapidly changing function landscape, which may require a more frequent kernel update.
4.  **GP Re-initialization around Best Point:** Re-initializing the GP model with samples clustered around the current best point helps to intensify the search in promising regions and improve the convergence rate.

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
from sklearn.cluster import KMeans

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
        self.gradient_norm_threshold = 0.1

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
        ucb = mu - self.kappa * sigma

        # Calculate Expected Improvement
        improvement = self.best_y - mu
        Z = improvement / sigma if sigma > 0 else np.inf
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Weighted combination of UCB and EI
        ucb_ei_w = max(self.ucb_ei_weight, self.min_ucb_ei_weight)
        return (ucb_ei_w * ei + (1 - ucb_ei_w) * ucb) / self.temperature

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

            for i in range(5): # Reduced iterations for efficiency
                # Calculate gradient
                mu, sigma_matrix = gp.predict(x.reshape(1, -1), return_cov=True)
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

            # Random Search around best point
            for i in range(3):
                x_rand = x0 + np.random.normal(0, search_radius, size=self.dim)
                x_rand = np.clip(x_rand, self.bounds[0], self.bounds[1])
                y_pred_rand = gp_objective(x_rand)
                if y_pred_rand < best_y_pred:
                    best_y_pred = y_pred_rand
                    best_x = x_rand

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
                kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.kernel_length_scale, length_scale_bounds="fixed")
                self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-5)
                self.gp.fit(self.X, self.y)


            # Select next point(s) by acquisition function
            remaining_budget = self.budget - self.n_evals
            batch_size = max(1, int(2 * (1 - remaining_budget / self.budget)))  # Dynamic batch size
            next_X = self._select_next_points(batch_size)

            # Evaluate the new point(s)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Local search around the best point
            predicted_y, predicted_sigma = self.gp.predict(self.best_x.reshape(1, -1), return_std=True)
            num_local_restarts = 2 + int(self.budget / (self.n_evals + 1e-6)) # More restarts early on
            local_x = self.local_search(self.gp, self.best_x, predicted_sigma[0], num_restarts=num_local_restarts)
            predicted_y_local, predicted_sigma_local = self.gp.predict(local_x.reshape(1, -1), return_std=True)


            # Probabilistic evaluation
            improvement = self.best_y - predicted_y_local[0]
            Z = improvement / predicted_sigma_local[0] if predicted_sigma_local[0] > 0 else np.inf
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

            # Decay exploration factor and kappa
            self.exploration_factor *= self.exploration_decay
            self.kappa = max(self.min_kappa, self.kappa * self.kappa_decay)
            self.ei_weight = max(self.min_ei_weight, self.ei_weight * self.ei_weight_decay)
            self.temperature *= self.temperature_decay
            self.ucb_ei_weight = max(self.min_ucb_ei_weight, self.ucb_ei_weight * self.ucb_ei_weight_decay)

            #Adaptive Kernel Update
            if len(self.y) > 1:
                improvement_ratio = abs(self.y[-1] - self.y[-2]) / abs(self.y[-2]) if abs(self.y[-2]) > 1e-6 else 0
                predicted_mu, predicted_sigma = self.gp.predict(self.X[-1].reshape(1, -1), return_std=True)
                uncertainty = predicted_sigma[0]

                # Calculate gradient norm
                def gp_objective(x):
                    mu, _ = self.gp.predict(x.reshape(1, -1), return_std=True)
                    return mu[0]

                gradient = np.zeros(self.dim)
                for i in range(self.dim):
                    x_plus = self.X[-1].copy()
                    x_minus = self.X[-1].copy()
                    delta = 1e-4
                    x_plus[i] += delta
                    x_minus[i] -= delta
                    gradient[i] = (gp_objective(x_plus) - gp_objective(x_minus)) / (2 * delta)
                gradient_norm = np.linalg.norm(gradient)

                if improvement_ratio > 0.1 or uncertainty > 0.5 or gradient_norm > self.gradient_norm_threshold:
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
                # Re-initialize GP model with samples clustered around the current best point
                num_clusters = 2
                kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit(self.X)
                closest_cluster_index = np.argmin([np.linalg.norm(self.best_x - kmeans.cluster_centers_[i]) for i in range(num_clusters)])
                cluster_samples = self.X[kmeans.labels_ == closest_cluster_index]
                if len(cluster_samples) > 0:
                    mean_sample = np.mean(cluster_samples, axis=0)
                    new_X = np.random.normal(mean_sample, 0.5, size=(self.n_init, self.dim))
                    new_X = np.clip(new_X, self.bounds[0], self.bounds[1])
                else:
                    new_X = self._sample_points(self.n_init)
                new_y = self._evaluate_points(func, new_X)
                self._update_eval_points(new_X, new_y)
                self._fit_model(self.X, self.y)
                self.local_optima_count = 0

        return self.best_y, self.best_x
```
## Feedback
 The algorithm EHBBO_BLS_PLE_AdaptiveKernel_v7 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1717 with standard deviation 0.1076.

took 528.71 seconds to run.