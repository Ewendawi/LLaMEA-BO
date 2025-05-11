# Description
**GradientEnhancedOptimisticTrustRegionBOV2 (GEOTRBOV2)**: This algorithm builds upon GEOTRBO by incorporating several refinements to improve its performance. Firstly, it introduces a more sophisticated trust region adaptation strategy that considers both the improvement in function value and the agreement between the GP model's predicted gradient and the finite difference gradient. Secondly, it employs a dynamic weighting of the GP model's prediction and the gradient information during local search. This weighting is adjusted based on the trust region's success. Finally, it adds a probabilistic global search step using EI to escape local optima, similar to AdaptiveTrustRegionHybridBO. The initial sampling is also improved by increasing the number of initial samples and using a Sobol sequence for better space-filling properties.

# Justification
The following changes are made to improve the performance of the original GEOTRBO:

1.  **Improved Trust Region Adaptation:** The original algorithm adjusts the trust region size based solely on whether the local search improves the function value. This is refined by also considering the agreement between the GP model's gradient and the finite difference gradient. If the gradients are in agreement, it suggests the GP model is accurate in the region, and the trust region can be expanded more aggressively. Disagreement leads to more conservative shrinking. This helps to manage the exploration-exploitation trade-off more effectively.

2.  **Dynamic Weighting of GP and Gradient in Local Search:** The original GEOTRBO uses a fixed weight for incorporating gradient information in the local search. This version adaptively adjusts this weight based on the success of the trust region. If the trust region search consistently yields improvements, the weight given to the gradient is reduced, allowing the GP model to guide the search more. Conversely, if the trust region is struggling, the gradient information is given more weight to potentially escape local optima.

3.  **Probabilistic Global Search with EI:** To prevent premature convergence, a probabilistic global search step is added. With a small probability, the algorithm will sample a point based on the Expected Improvement (EI) acquisition function and evaluate it. This provides a mechanism to escape local optima that the trust region may be stuck in.

4. **Improved Initial Sampling:** Sobol sequence is used instead of Latin Hypercube Sampling for better space-filling properties in the initial sampling. The number of initial samples is also increased to better represent the search space.

These changes aim to improve the algorithm's ability to balance exploration and exploitation, adapt to the local characteristics of the objective function, and escape local optima, leading to better overall performance.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import norm

class GradientEnhancedOptimisticTrustRegionBOV2:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 4 * self.dim # initial number of samples
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
        self.gradient_agreement_threshold = 0.5
        self.local_search_gradient_weight = 0.1
        self.global_search_probability = 0.05

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.Sobol(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        model = self._fit_model(self.X, self.y)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Upper Confidence Bound
        ucb = mu - self.beta * sigma # minimize
        return ucb

    def _expected_improvement(self, X):
        model = self._fit_model(self.X, self.y)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        
        best = np.min(self.y)
        imp = mu - best
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
        return -ei  # We want to maximize EI, but minimize the acquisition function

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        candidate_points = self._sample_points(10 * batch_size)
        acquisition_values = self._acquisition_function(candidate_points)
        indices = np.argsort(acquisition_values.flatten())[:batch_size] # minimize
        return candidate_points[indices]

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

        # Update best seen point
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]

    def _estimate_gradient(self, model, x):
        # Estimate the gradient of the function at point x using finite differences
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.delta
            x_minus[i] -= self.delta
            
            # Clip to ensure the points are within bounds
            x_plus[i] = np.clip(x_plus[i], self.bounds[0][i], self.bounds[1][i])
            x_minus[i] = np.clip(x_minus[i], self.bounds[0][i], self.bounds[1][i])

            # Use the GP model to predict function values
            y_plus, _ = model.predict(x_plus.reshape(1, -1), return_std=True)
            y_minus, _ = model.predict(x_minus.reshape(1, -1), return_std=True)

            gradient[i] = (y_plus - y_minus) / (2 * self.delta)
        return gradient

    def _local_search(self, model, center, gradient, n_points=50):
        # Perform local search within the trust region using the GP model and gradient information
        # Generate candidate points within the trust region
        candidate_points = center + self.trust_region_size * np.random.uniform(-1, 1, size=(n_points, self.dim))

        # Clip the candidate points to stay within the bounds
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])

        # Predict the mean values using the GP model
        mu, _ = model.predict(candidate_points, return_std=True)

        # Incorporate gradient information into the prediction
        mu = mu.reshape(-1) - self.local_search_gradient_weight * np.sum(gradient * (candidate_points - center), axis=1)

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

        while self.n_evals < self.budget:
            # Fit the GP model
            model = self._fit_model(self.X, self.y)

            # Estimate gradient at the best point using finite differences
            fd_gradient = self._estimate_gradient(model, best_x)

            # Get GP model's predicted gradient
            gp_gradient = self._estimate_gradient(model, best_x)

            # Perform local search within the trust region
            next_x = self._local_search(model, best_x.copy(), fd_gradient)
            next_x = np.clip(next_x, self.bounds[0], self.bounds[1]) # Ensure it's within bounds

            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0] # Evaluate the actual function
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            # Check if the new point is better than the current best
            improvement = next_y < best_y

            # Calculate the cosine similarity between the finite difference gradient and the GP gradient
            cos_similarity = np.dot(fd_gradient, gp_gradient) / (np.linalg.norm(fd_gradient) * np.linalg.norm(gp_gradient) + 1e-8)
            gradient_agreement = cos_similarity > self.gradient_agreement_threshold

            if improvement:
                best_x = next_x
                best_y = next_y
                # Adjust trust region size
                if gradient_agreement:
                    trust_region_change = self.trust_region_expand - 1.0
                    self.trust_region_size *= self.trust_region_expand
                else:
                    trust_region_change = 0.2
                    self.trust_region_size *= (1 + trust_region_change)
                self.local_search_gradient_weight = max(0.0, self.local_search_gradient_weight - 0.01)
            else:
                # Shrink trust region if no improvement
                trust_region_change = 1.0 - self.trust_region_shrink
                self.trust_region_size *= self.trust_region_shrink
                self.local_search_gradient_weight = min(0.2, self.local_search_gradient_weight + 0.01)

            # Apply momentum to trust region size update
            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0) # Keep trust region within reasonable bounds
            
            # Decay exploration parameter
            self.beta *= self.beta_decay

            # Probabilistic global search
            if np.random.rand() < self.global_search_probability:
                candidate_points = self._sample_points(10)
                ei_values = self._expected_improvement(candidate_points)
                best_index = np.argmin(ei_values)
                global_x = candidate_points[best_index]
                global_y = self._evaluate_points(func, global_x.reshape(1, -1))[0, 0]
                self._update_eval_points(global_x.reshape(1, -1), np.array([[global_y]]))
                if global_y < best_y:
                    best_x = global_x
                    best_y = global_y

        return best_y, best_x
```
## Feedback
 The algorithm GradientEnhancedOptimisticTrustRegionBOV2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1866 with standard deviation 0.1085.

took 131.25 seconds to run.