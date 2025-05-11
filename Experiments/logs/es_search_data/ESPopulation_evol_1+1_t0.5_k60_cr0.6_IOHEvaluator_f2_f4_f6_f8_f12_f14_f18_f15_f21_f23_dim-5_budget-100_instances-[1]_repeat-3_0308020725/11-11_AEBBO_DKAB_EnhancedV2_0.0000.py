from collections.abc import Callable
from scipy.stats import qmc, norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize, Bounds, approx_fprime
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde

class AEBBO_DKAB_EnhancedV2:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim # number of initial samples

        # Do not add any other arguments without a default value
        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.acq_strategy = "EI+UCB"
        self.initial_exploration_weight = 0.2
        self.exploration_weight = self.initial_exploration_weight
        self.batch_size = 2
        self.kernel_length_scale = 1.0
        self.local_search_restarts = 3
        self.length_scale_weight = 0.5 # Weight for previous length scale in adaptation
        self.length_scale_history = []
        self.length_scale_history_length = 5
        self.stagnation_threshold = 10 * dim
        self.stagnation_counter = 0
        self.previous_best_y = float('inf')

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        if self.X is None or len(self.X) < self.dim + 1:
            sampler = qmc.LatinHypercube(d=self.dim)
            samples = sampler.random(n=n_points)
            return qmc.scale(samples, self.bounds[0], self.bounds[1])
        else:
            try:
                kde = gaussian_kde(self.X.T)
                samples = kde.resample(n_points)
                samples = np.clip(samples.T, self.bounds[0], self.bounds[1])
                return samples
            except np.linalg.LinAlgError:
                sampler = qmc.LatinHypercube(d=self.dim)
                samples = sampler.random(n=n_points)
                return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature

        # Dynamic kernel adaptation
        distances = cdist(X, X, metric='euclidean')
        distances = np.triu(distances, k=1)
        distances = distances[distances > 0]
        mean_distance = np.mean(distances) if len(distances) > 0 else 1.0

        # Smooth kernel length scale adaptation with moving average
        self.length_scale_history.append(mean_distance)
        if len(self.length_scale_history) > self.length_scale_history_length:
            self.length_scale_history.pop(0)
        mean_length_scale = np.mean(self.length_scale_history)

        self.kernel_length_scale = self.length_scale_weight * self.kernel_length_scale + (1 - self.length_scale_weight) * mean_length_scale

        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.kernel_length_scale, length_scale_bounds="fixed")
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.acq_strategy == "EI":
            return self._expected_improvement(X)
        elif self.acq_strategy == "UCB":
            return self._upper_confidence_bound(X)
        elif self.acq_strategy == "EI+UCB":
            ei = self._expected_improvement(X)
            ucb = self._upper_confidence_bound(X)
            return ei + self.exploration_weight * ucb
        else:
            raise ValueError("Invalid acquisition function strategy.")

    def _expected_improvement(self, X):
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-6)  # avoid division by zero
        gamma = (self.best_y - mu) / (sigma + 1e-9) # avoid division by zero
        ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return ei.reshape(-1, 1)

    def _upper_confidence_bound(self, X):
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu.reshape(-1, 1) + 2 * sigma.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        selected_points = []
        candidates = self._sample_points(100 * self.dim) # Generate a larger candidate set

        for _ in range(batch_size):
            acq_values = self._acquisition_function(candidates)
            best_index = np.argmax(acq_values)
            selected_point = candidates[best_index]

            # Local search to refine the selected point
            def obj(x):
                return -self._acquisition_function(x.reshape(1, -1))[0, 0]  # Negate for minimization

            def obj_grad(x):
                return -approx_fprime(x, lambda x: self._acquisition_function(x.reshape(1, -1))[0, 0], epsilon=1e-6)

            bounds = Bounds(self.bounds[0], self.bounds[1])
            best_x = None
            best_obj = float('inf')

            # Adaptive local search restarts
            adaptive_restarts = int(self.local_search_restarts * (1 - self.n_evals / self.budget)) + 1
            for _ in range(adaptive_restarts):
                res = minimize(obj, selected_point, method='L-BFGS-B', jac=obj_grad, bounds=bounds)
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_x = res.x
                selected_point = self._sample_points(1).flatten()

            if best_x is not None:
                selected_point = best_x

            selected_points.append(selected_point)

            # Remove the selected point from the candidates to avoid duplicates in the batch
            candidates = np.delete(candidates, best_index, axis=0)

        return np.array(selected_points)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        # Do not change the function signature
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

        # Update best seen value
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Optimization loop
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Dynamic batch size adjustment
            mean_sigma = np.mean(self.gp.predict(self.X, return_std=True)[1])
            self.batch_size = max(1, min(self.batch_size + (1 if mean_sigma > 0.5 else -1), 5))
            next_X = self._select_next_points(min(self.batch_size, self.budget - self.n_evals))

            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Adaptive acquisition balancing based on convergence rate
            if self.best_y < self.previous_best_y:
                self.stagnation_counter = 0
                convergence_rate = (self.previous_best_y - self.best_y) / self.previous_best_y if self.previous_best_y != 0 else 0
                self.exploration_weight = self.initial_exploration_weight * np.exp(-5 * convergence_rate)  # Reduce exploration if converging
            else:
                self.stagnation_counter += 1
                self.exploration_weight = min(1.0, self.exploration_weight * 1.1)  # Increase exploration if stagnating
            self.exploration_weight = np.clip(self.exploration_weight, 0.01, 1.0)
            self.previous_best_y = self.best_y

            # Stagnation restart mechanism
            if self.stagnation_counter > self.stagnation_threshold:
                print("Restarting due to stagnation")
                self.stagnation_counter = 0
                initial_X = self._sample_points(self.n_init)
                initial_y = self._evaluate_points(func, initial_X)
                self._update_eval_points(initial_X, initial_y)
                self.gp = None # Reset the GP

        return self.best_y, self.best_x
