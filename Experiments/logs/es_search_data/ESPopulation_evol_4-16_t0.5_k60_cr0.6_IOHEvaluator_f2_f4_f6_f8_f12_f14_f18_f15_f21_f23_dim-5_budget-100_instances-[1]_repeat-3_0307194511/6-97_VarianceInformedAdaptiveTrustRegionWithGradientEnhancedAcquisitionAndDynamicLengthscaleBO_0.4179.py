from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors

class VarianceInformedAdaptiveTrustRegionWithGradientEnhancedAcquisitionAndDynamicLengthscaleBO:
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
        self.global_search_prob = 0.05 # Probability of performing a global search step
        self.delta = 1e-3 # step size for finite differences in gradient estimation
        self.trust_region_momentum = 0.5 # Momentum for trust region size update
        self.ei_weight = 0.5 # Initial weight for EI
        self.ucb_weight = 0.5 # Initial weight for UCB
        self.success_threshold = 0.7 # Threshold for adjusting EI/UCB weights
        self.success_rate = 0.0 # Moving average of local search success
        self.success_momentum = 0.8 # Momentum for updating success rate
        self.gradient_points = 3 # Number of points to average for gradient estimation
        self.variance_threshold = 0.1 # Threshold for adjusting EI/UCB weights based on variance
        self.lengthscale = 1.0 # Initial lengthscale
        self.lengthscale_update_interval = 2 * self.dim
        self.min_trust_region_size = 0.1
        self.max_trust_region_size = 5.0

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

        # Adaptive lengthscale: re-estimate every few iterations
        if len(X) > self.n_init and self.n_evals % self.lengthscale_update_interval == 0:
            nn = NearestNeighbors(n_neighbors=min(len(X), 10)).fit(X)
            distances, _ = nn.kneighbors(X)
            self.lengthscale = np.median(distances[:, 1])  # Exclude the point itself
            self.lengthscale = np.clip(self.lengthscale, 1e-3, 1e3)

        # Define the kernel with the estimated lengthscale
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=self.lengthscale, length_scale_bounds=(1e-3, 1e3))

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

    def _acquisition_function(self, X):
        # Combine EI and UCB with dynamic weights
        ei = self._acquisition_function_ei(X)
        ucb = self._acquisition_function_ucb(X)
        return self.ei_weight * ei + self.ucb_weight * ucb

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        candidate_points = self._sample_points(10 * batch_size)
        acquisition_values = self._acquisition_function(candidate_points)

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

    def _estimate_gradient(self, model, x):
        # Estimate the gradient of the function at point x using finite differences
        gradient = np.zeros(self.dim)
        
        # Use predictive variance to modulate the gradient estimation step size
        _, pred_var = model.predict(x.reshape(1, -1), return_std=True)
        adaptive_delta = self.lengthscale / 10.0 * (1 + pred_var[0] / self.variance_threshold)
        adaptive_delta = np.clip(adaptive_delta, 1e-6, 0.1) # Ensure reasonable bounds

        # Average gradient over multiple points near x
        for _ in range(self.gradient_points):
            x_sample = x + np.random.normal(0, self.trust_region_size / 5, self.dim) # Add some noise
            x_sample = np.clip(x_sample, self.bounds[0], self.bounds[1]) # Clip to bounds
            
            gradient_sample = np.zeros(self.dim)
            for i in range(self.dim):
                x_plus = x_sample.copy()
                x_minus = x_sample.copy()
                x_plus[i] += adaptive_delta
                x_minus[i] -= adaptive_delta
                
                # Clip to ensure the points are within bounds
                x_plus[i] = np.clip(x_plus[i], self.bounds[0][i], self.bounds[1][i])
                x_minus[i] = np.clip(x_minus[i], self.bounds[0][i], self.bounds[1][i])

                # Use the GP model to predict function values
                y_plus, _ = model.predict(x_plus.reshape(1, -1), return_std=True)
                y_minus, _ = model.predict(x_minus.reshape(1, -1), return_std=True)

                gradient_sample[i] = (y_plus - y_minus) / (2 * adaptive_delta)
            
            gradient += gradient_sample
        
        return gradient / self.gradient_points

    def _local_search(self, model, center, gradient, n_points=50):
        # Perform local search within the trust region using the GP model and gradient information
        # Generate candidate points within the trust region
        candidate_points = center + self.trust_region_size * np.random.uniform(-1, 1, size=(n_points, self.dim))

        # Clip the candidate points to stay within the bounds
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])

        # Predict the mean values using the GP model
        mu, _ = model.predict(candidate_points, return_std=True)

        # Incorporate gradient information into the prediction
        mu = mu.reshape(-1) - 0.1 * np.sum(gradient * (candidate_points - center), axis=1)

        # Select the point with the minimum predicted mean value
        best_index = np.argmin(mu)
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
        
        num_success = 0

        while self.n_evals < self.budget:
            # Fit the GP model
            model = self._fit_model(self.X, self.y)

            # Global search with probability global_search_prob
            if np.random.rand() < self.global_search_prob:
                next_x = self._select_next_points(1)[0] # Select a point using EI for global search
            else:
                # Estimate gradient at the best point
                gradient = self._estimate_gradient(model, best_x)

                # Perform local search within the trust region
                next_x = self._local_search(model, best_x.copy(), gradient)
                next_x = np.clip(next_x, self.bounds[0], self.bounds[1]) # Ensure it's within bounds

            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0] # Evaluate the actual function
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            # Calculate average predictive variance within the trust region
            candidate_points = best_x + self.trust_region_size * np.random.uniform(-1, 1, size=(50, self.dim))
            candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])
            _, pred_var = model.predict(candidate_points, return_std=True)
            avg_pred_var = np.mean(pred_var)

            # Check if the new point is better than the current best
            improvement = best_y - next_y
            predicted_improvement, _ = model.predict(next_x.reshape(1, -1), return_std=True)
            predicted_improvement = best_y - predicted_improvement[0]

            if next_y < best_y:
                best_x = next_x
                best_y = next_y
                
                # Update success rate
                num_success += 1
                
                # Adjust trust region size
                if predicted_improvement > 0:
                    improvement_ratio = improvement / predicted_improvement
                    if improvement_ratio > 0.75:
                        self.trust_region_size *= self.trust_region_expand
                    else:
                        self.trust_region_size *= (1 + 0.5 * improvement_ratio) # Slightly expand if there's some improvement
                else:
                    self.trust_region_size *= self.trust_region_expand # Expand if predicted improvement is zero
            else:
                # Shrink trust region if no improvement
                self.trust_region_size *= self.trust_region_shrink

            # Adjust trust region size based on predictive variance
            if avg_pred_var > self.variance_threshold:
                self.trust_region_size *= 1.1 # Increase if variance is high
            else:
                self.trust_region_size *= 0.9 # Decrease if variance is low

            # Apply momentum to trust region size update
            self.trust_region_size = self.trust_region_momentum * self.trust_region_size + (1 - self.trust_region_momentum) * np.clip(self.trust_region_size, self.min_trust_region_size, self.max_trust_region_size) # Keep trust region within reasonable bounds
            
            # Decay exploration parameter
            self.beta *= self.beta_decay
            
            # Update success rate and adjust EI/UCB weights
            self.success_rate = self.success_momentum * self.success_rate + (1 - self.success_momentum) * (num_success > 0)
            num_success = 0 #reset
            
            # Adjust EI/UCB weights based on success rate and predictive variance
            if self.success_rate > self.success_threshold:
                # Increase UCB weight, decrease EI weight
                self.ucb_weight = min(1.0, self.ucb_weight + 0.05)
                self.ei_weight = max(0.0, self.ei_weight - 0.05)
            else:
                # Increase EI weight, decrease UCB weight
                self.ei_weight = min(1.0, self.ei_weight + 0.05)
                self.ucb_weight = max(0.0, self.ucb_weight - 0.05)

            if avg_pred_var > self.variance_threshold:
                self.ei_weight = min(1.0, self.ei_weight + 0.05)
                self.ucb_weight = max(0.0, self.ucb_weight - 0.05)
            else:
                self.ucb_weight = min(1.0, self.ucb_weight + 0.05)
                self.ei_weight = max(0.0, self.ei_weight - 0.05)

        return best_y, best_x
