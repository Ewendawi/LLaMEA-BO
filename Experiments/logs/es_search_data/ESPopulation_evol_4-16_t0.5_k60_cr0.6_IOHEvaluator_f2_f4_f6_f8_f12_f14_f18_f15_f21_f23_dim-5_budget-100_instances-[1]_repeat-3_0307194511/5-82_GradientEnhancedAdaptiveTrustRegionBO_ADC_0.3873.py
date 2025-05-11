from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize

class GradientEnhancedAdaptiveTrustRegionBO_ADC:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * self.dim # initial number of samples
        self.batch_size = min(5, self.dim)
        self.trust_region_size = 2.0
        self.trust_region_shrink = 0.5
        self.trust_region_expand = 2.0
        self.beta = 2.0  # Exploration parameter for UCB
        self.beta_decay = 0.99 # Decay rate for beta
        self.best_x = None
        self.best_y = np.inf
        self.delta = 1e-3 # step size for finite differences in gradient estimation
        self.trust_region_momentum = 0.5 # Momentum for trust region size update
        self.prev_trust_region_change = 0.0
        self.global_search_prob = 0.05 # Probability of performing a global search step
        self.ei_weight = 0.5 # Initial weight for EI
        self.ucb_weight = 0.5 # Initial weight for UCB
        self.success_threshold = 0.1 # Threshold for considering local search successful

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature

        # Efficient lengthscale estimation using nearest neighbors
        nn = NearestNeighbors(n_neighbors=min(len(X), 10)).fit(X)
        distances, _ = nn.kneighbors(X)
        median_distance = np.median(distances[:, 1])  # Exclude the point itself

        # Define the kernel with the estimated lengthscale
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=median_distance, length_scale_bounds=(1e-3, 1e3))
        self.lengthscale = median_distance # save lengthscale for gradient estimation

        # Gaussian Process Regressor
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
        model.fit(X, y)
        return model

    def _acquisition_function_ei(self, X):
        # Implement Expected Improvement acquisition function
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        model = self._fit_model(self.X, self.y)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        y_best = np.min(self.y)
        imp = y_best - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        return ei

    def _acquisition_function_ucb(self, X):
        # Implement Upper Confidence Bound acquisition function
        model = self._fit_model(self.X, self.y)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        ucb = mu - self.beta * sigma  # minimize

        return ucb

    def _select_next_points_ei(self, batch_size):
        # Select the next points to evaluate using EI
        candidate_points = self._sample_points(10 * batch_size)
        acquisition_values = self._acquisition_function_ei(candidate_points)

        indices = np.argsort(acquisition_values.flatten())[::-1][:batch_size]
        return candidate_points[indices]

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
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

        # Update best seen point
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]

    def _estimate_gradient(self, model, x):
        # Estimate the gradient of the function at point x using central finite differences
        gradient = np.zeros(self.dim)
        delta = min(min(self.lengthscale, self.delta), self.trust_region_size/10) # Adaptive delta

        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += delta
            x_minus[i] -= delta
            
            # Clip to ensure the points are within bounds
            x_plus[i] = np.clip(x_plus[i], self.bounds[0][i], self.bounds[1][i])
            x_minus[i] = np.clip(x_minus[i], self.bounds[0][i], self.bounds[1][i])

            # Use the GP model to predict function values
            y_plus, _ = model.predict(x_plus.reshape(1, -1), return_std=True)
            y_minus, _ = model.predict(x_minus.reshape(1, -1), return_std=True)

            gradient[i] = (y_plus - y_minus) / (2 * delta)
        return gradient

    def _mixed_acquisition(self, X):
        # Combine EI and UCB with dynamic weights
        ei = self._acquisition_function_ei(X)
        ucb = self._acquisition_function_ucb(X)
        return self.ei_weight * ei + self.ucb_weight * ucb

    def _local_search(self, model, center, n_points=50):
        # Perform local search within the trust region using the GP model and gradient information
        # Generate candidate points within the trust region
        candidate_points = center + self.trust_region_size * np.random.uniform(-1, 1, size=(n_points, self.dim))

        # Clip the candidate points to stay within the bounds
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])

        # Calculate the mixed acquisition function values
        acquisition_values = self._mixed_acquisition(candidate_points)

        # Select the point with the minimum acquisition value
        best_index = np.argmin(acquisition_values)
        best_point = candidate_points[best_index]

        return best_point

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_index = np.argmin(self.y)
        best_x = self.X[best_index]
        best_y = self.y[best_index][0]

        while self.n_evals < self.budget:
            # Fit the GP model
            model = self._fit_model(self.X, self.y)

            # Global search with probability global_search_prob
            if np.random.rand() < self.global_search_prob:
                next_x = self._select_next_points_ei(1)[0] # Select a point using EI for global search
            else:
                # Perform local search within the trust region
                next_x = self._local_search(model, best_x.copy())
                next_x = np.clip(next_x, self.bounds[0], self.bounds[1]) # Ensure it's within bounds

            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0] # Evaluate the actual function
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            # Check if the new point is better than the current best
            improvement = (best_y - next_y) / best_y if best_y != 0 else (best_y - next_y)
            relative_improvement = improvement / self.trust_region_size

            if next_y < best_y:
                best_x = next_x
                best_y = next_y
                # Adjust trust region size
                if relative_improvement > self.success_threshold:
                    self.trust_region_size *= self.trust_region_expand
                else:
                    self.trust_region_size *= (1 + relative_improvement) # Expand less aggressively

                # Increase UCB weight if local search is successful
                self.ucb_weight = min(1.0, self.ucb_weight + 0.1)
                self.ei_weight = max(0.0, self.ei_weight - 0.1)
            else:
                # Shrink trust region if no improvement
                self.trust_region_size *= self.trust_region_shrink

                # Increase EI weight if local search is unsuccessful
                self.ei_weight = min(1.0, self.ei_weight + 0.1)
                self.ucb_weight = max(0.0, self.ucb_weight - 0.1)

            # Apply momentum to trust region size update
            self.trust_region_size = self.trust_region_momentum * self.trust_region_size + (1 - self.trust_region_momentum) * np.clip(self.trust_region_size, 0.1, 5.0) # Keep trust region within reasonable bounds
            
            # Decay exploration parameter
            self.beta *= self.beta_decay

        return best_y, best_x
