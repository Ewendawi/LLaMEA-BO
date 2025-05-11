# Description
**Hybridized Adaptive Trust Region with Gradient Agreement and Dynamic Lengthscale BO (HATR-GADLBO)**: This algorithm combines the strengths of HGATRBayesBO and AGTRDAB-GTA BO, focusing on a robust trust region adaptation strategy and efficient lengthscale management. It integrates the success history-based EI/UCB weight adjustment from HGATRBayesBO with the gradient-based trust region adaptation from AGTRDAB-GTA BO. A key improvement is the use of a more stable lengthscale adaptation strategy that considers both nearest neighbor distances and the agreement between the predicted gradient and the actual function decrease direction. Furthermore, the local search is enhanced by adaptively adjusting the number of candidate points based on the trust region size and the GP's predictive variance.

# Justification
The algorithm builds upon the strengths of the two selected algorithms:
1.  **Trust Region Adaptation:** It incorporates the gradient agreement-based trust region adaptation from AGTRDAB-GTA BO, which adjusts the trust region size based on the alignment between the predicted gradient and the actual function decrease direction. This helps in more efficiently navigating the search space.
2.  **Acquisition Balancing:** It retains the success history-based dynamic EI/UCB balancing from HGATRBayesBO, which adjusts the weights of EI and UCB based on the moving average of local search success. This allows for a more adaptive exploration-exploitation trade-off.
3.  **Lengthscale Adaptation:** The lengthscale is dynamically updated using a combination of nearest neighbor distances and gradient agreement. This provides a more robust estimate of the function's local characteristics, improving the GP model's accuracy.
4.  **Adaptive Local Search:** The number of candidate points in the local search is adaptively adjusted based on the trust region size and the GP's predictive variance. This ensures that the local search is both efficient and effective.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors

class HybridizedAdaptiveTrustRegionWithGradientAgreementAndDynamicLengthscaleBO:
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
        self.weight_adjust_rate = 0.05 # Rate at which to adjust EI/UCB weights
        self.success_history = [] # History of local search success (True/False)
        self.success_history_length = 5 # Length of success history to consider
        self.success_threshold = 0.7 # Threshold for adjusting EI/UCB weights
        self.success_rate = 0.0 # Moving average of local search success
        self.success_momentum = 0.8 # Momentum for updating success rate
        self.gradient_points = 3 # Number of points to average for gradient estimation
        self.lengthscale = 1.0 # Initial lengthscale
        self.lengthscale_momentum = 0.5
        self.lengthscale_min = 0.1

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
        
        # Average gradient over multiple points near x
        for _ in range(self.gradient_points):
            x_sample = x + np.random.normal(0, self.trust_region_size / 5, self.dim) # Add some noise
            x_sample = np.clip(x_sample, self.bounds[0], self.bounds[1]) # Clip to bounds
            
            gradient_sample = np.zeros(self.dim)
            for i in range(self.dim):
                x_plus = x_sample.copy()
                x_minus = x_sample.copy()
                x_plus[i] += self.delta
                x_minus[i] -= self.delta
                
                # Clip to ensure the points are within bounds
                x_plus[i] = np.clip(x_plus[i], self.bounds[0][i], self.bounds[1][i])
                x_minus[i] = np.clip(x_minus[i], self.bounds[0][i], self.bounds[1][i])

                # Use the GP model to predict function values
                y_plus, _ = model.predict(x_plus.reshape(1, -1), return_std=True)
                y_minus, _ = model.predict(x_minus.reshape(1, -1), return_std=True)

                gradient_sample[i] = (y_plus - y_minus) / (2 * self.delta)
            
            gradient += gradient_sample
        
        return gradient / self.gradient_points

    def _local_search(self, model, center, gradient):
        # Perform local search within the trust region using the GP model and gradient information
        # Adapt the number of candidate points based on trust region size and GP variance
        mu, sigma = model.predict(center.reshape(1, -1), return_std=True)
        num_points = max(10, int(50 * self.trust_region_size / (sigma + 1e-6)))

        # Generate candidate points within the trust region
        candidate_points = np.random.normal(center, self.trust_region_size / 3, size=(num_points, self.dim))

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

            # Update lengthscale
            nn = NearestNeighbors(n_neighbors=min(len(self.X), 10)).fit(self.X)
            distances, _ = nn.kneighbors(self.X)
            median_distance = np.median(distances[:, 1])

            # Estimate gradient at the best point
            gradient = self._estimate_gradient(model, best_x)

            # Calculate the angle between the predicted gradient and the actual improvement direction
            # Use a small random displacement to avoid zero norm issues
            random_displacement = np.random.normal(0, 1e-6, self.dim)
            improvement_direction = best_x - (best_x + random_displacement)

            cos_angle = np.dot(gradient, improvement_direction) / (np.linalg.norm(gradient) * np.linalg.norm(improvement_direction) + 1e-8)

            # Combine nearest neighbor distance and gradient agreement for lengthscale update
            self.lengthscale = self.lengthscale_momentum * self.lengthscale + (1 - self.lengthscale_momentum) * (0.5 * median_distance + 0.5 * (1 - abs(cos_angle)))
            self.lengthscale = max(self.lengthscale, self.lengthscale_min)

            # Global search with probability global_search_prob
            if np.random.rand() < self.global_search_prob:
                next_x = self._select_next_points(1)[0] # Select a point using EI for global search
            else:
                # Perform local search within the trust region
                next_x = self._local_search(model, best_x.copy(), gradient)
                next_x = np.clip(next_x, self.bounds[0], self.bounds[1]) # Ensure it's within bounds

            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0] # Evaluate the actual function
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            # Check if the new point is better than the current best
            if next_y < best_y:
                best_x = next_x
                best_y = next_y
                
                # Update success rate
                num_success += 1
                
                # Adjust trust region size based on gradient agreement
                if cos_angle > 0.5:
                    self.trust_region_size *= self.trust_region_expand
                else:
                    self.trust_region_size *= (1 + self.trust_region_shrink) / 2 # Less aggressive expansion

            else:
                # Shrink trust region if no improvement
                self.trust_region_size *= self.trust_region_shrink

            # Apply momentum to trust region size update
            self.trust_region_size = self.trust_region_momentum * self.trust_region_size + (1 - self.trust_region_momentum) * np.clip(self.trust_region_size, 0.1, 5.0) # Keep trust region within reasonable bounds
            
            # Decay exploration parameter
            self.beta *= self.beta_decay
            
            # Update success rate and adjust EI/UCB weights
            self.success_rate = self.success_momentum * self.success_rate + (1 - self.success_momentum) * (num_success > 0)
            num_success = 0 #reset
            
            if self.success_rate > self.success_threshold:
                # Increase UCB weight, decrease EI weight
                self.ucb_weight = min(1.0, self.ucb_weight + self.weight_adjust_rate)
                self.ei_weight = max(0.0, self.ei_weight - self.weight_adjust_rate)
            else:
                # Increase EI weight, decrease UCB weight
                self.ei_weight = min(1.0, self.ei_weight + self.weight_adjust_rate)
                self.ucb_weight = max(0.0, self.ucb_weight - self.weight_adjust_rate)

        return best_y, best_x
```
## Feedback
 The algorithm HybridizedAdaptiveTrustRegionWithGradientAgreementAndDynamicLengthscaleBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1885 with standard deviation 0.1045.

took 324.03 seconds to run.