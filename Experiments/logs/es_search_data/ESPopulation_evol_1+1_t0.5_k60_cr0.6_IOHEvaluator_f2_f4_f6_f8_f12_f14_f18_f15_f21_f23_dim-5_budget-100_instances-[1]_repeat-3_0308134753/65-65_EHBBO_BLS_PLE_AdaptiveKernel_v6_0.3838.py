from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.linalg import solve

class EHBBO_BLS_PLE_AdaptiveKernel_v6:
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
        self.evaluated_points = set() # Keep track of evaluated points

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

        # Calculate gradient magnitude
        dmu_dx = self._gradient(X)
        gradient_magnitude = np.linalg.norm(dmu_dx, axis=1)

        # Weighted combination of UCB, EI, and gradient magnitude
        ei_w = max(self.ei_weight, self.min_ei_weight)
        return (ei_w * ei + (1 - ei_w) * ucb + 0.1 * gradient_magnitude) / self.temperature

    def _gradient(self, X):
        """Calculate the gradient of the GP mean function."""
        X = np.atleast_2d(X)
        n_samples = X.shape[0]
        dmu_dx = np.zeros((n_samples, self.dim))
        for i in range(n_samples):
            x = X[i]
            K = self.gp.kernel_(self.gp.X_train_, x.reshape(1, -1))
            alpha = self.gp.alpha_
            for d in range(self.dim):
                dK_dx = self._kernel_gradient(self.gp.X_train_, x, d)
                dmu_dx[i, d] = dK_dx.T @ alpha
        return dmu_dx

    def _kernel_gradient(self, X, x, dim):
        """Calculate the gradient of the RBF kernel with respect to a single dimension."""
        K = self.gp.kernel_.k2
        dx = X[:, dim] - x[dim]
        return K.length_scale ** -2 * dx[:, np.newaxis] * self.gp.kernel_(X, x.reshape(1, -1))

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

        # Calculate gradient of GP prediction
        def gp_objective(x):
            mu, _ = gp.predict(x.reshape(1, -1), return_std=True)
            return mu[0]

        # Adaptive search radius based on curvature
        H = self._hessian(x0)
        try:
            curvature = np.abs(np.linalg.eigvalsh(H)).max() # Maximum absolute eigenvalue as curvature
            search_radius = 0.1 + sigma / (1 + curvature) # Radius shrinks with higher curvature
        except:
            search_radius = 0.2 + sigma # Default radius if Hessian fails

        for _ in range(num_restarts):
            # Gradient-informed sampling
            gradient = self._gradient(x0.reshape(1, -1)).flatten()
            x_start = x0 + np.random.normal(0, 0.01, size=self.dim) + 0.01 * gradient # Add noise and gradient
            x_start = np.clip(x_start, self.bounds[0], self.bounds[1])


            bounds = [(max(self.bounds[0][i], x0[i] - search_radius), min(self.bounds[1][i], x0[i] + search_radius)) for i in range(self.dim)]  # Dynamic bounds
            res = minimize(gp_objective, x_start, method='L-BFGS-B', bounds=bounds)
            y_pred = gp_objective(res.x)

            if y_pred < best_y_pred:
                best_y_pred = y_pred
                best_x = res.x
        return best_x

    def _hessian(self, x):
        """Approximate the Hessian matrix using finite differences."""
        x = np.atleast_1d(x)
        n = x.shape[0]
        h = 1e-4
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x_forward_i = x.copy()
                x_forward_j = x.copy()
                x_forward_ij = x.copy()

                x_forward_i[i] += h
                x_forward_j[j] += h
                x_forward_ij[i] += h
                x_forward_ij[j] += h

                x_backward_i = x.copy()
                x_backward_j = x.copy()
                x_backward_ij = x.copy()

                x_backward_i[i] -= h
                x_backward_j[j] -= h
                x_backward_ij[i] -= h
                x_backward_ij[j] -= h

                if i == j:
                    H[i, i] = (self.gp.predict(x_forward_i.reshape(1, -1))[0] - 2 * self.gp.predict(x.reshape(1, -1))[0] + self.gp.predict(x_backward_i.reshape(1, -1))[0]) / (h * h)
                else:
                     H[i, j] = (self.gp.predict(x_forward_ij.reshape(1, -1))[0] - self.gp.predict(x_forward_i.reshape(1, -1))[0] - self.gp.predict(x_forward_j.reshape(1, -1))[0] + self.gp.predict(x.reshape(1, -1))[0]) / (h * h)
        return H

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
            next_X = self._select_next_points(1)

            # Check for duplicate points
            next_x_tuple = tuple(next_X.flatten().round(decimals=6))  # Round to avoid floating-point issues
            if next_x_tuple in self.evaluated_points:
                continue

            # Evaluate the new point(s)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self.evaluated_points.add(next_x_tuple)

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
            remaining_budget = self.budget - self.n_evals
            evaluation_threshold = 0.5 + 0.4 * (remaining_budget / self.budget)  # Higher threshold when budget is tight

            # Check for duplicate points before evaluation
            local_x_tuple = tuple(local_x.flatten().round(decimals=6))
            if prob_improvement > evaluation_threshold and self.n_evals < self.budget and local_x_tuple not in self.evaluated_points:
                # Evaluate the true function value only if the GP predicts a sufficiently high probability of improvement and we have budget
                local_y = func(local_x)
                self.n_evals += 1
                if local_y < self.best_y:
                    self.best_y = local_y
                    self.best_x = local_x
                self.evaluated_points.add(local_x_tuple)

            # Decay exploration factor and kappa
            self.exploration_factor *= self.exploration_decay
            self.kappa = max(self.min_kappa, self.kappa * self.kappa_decay)
            self.ei_weight = max(self.min_ei_weight, self.ei_weight * self.ei_weight_decay)
            self.temperature *= self.temperature_decay

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
            else:
                self.initial_sample_multiplier = max(1.0, self.initial_sample_multiplier * 0.95)

        return self.best_y, self.best_x
