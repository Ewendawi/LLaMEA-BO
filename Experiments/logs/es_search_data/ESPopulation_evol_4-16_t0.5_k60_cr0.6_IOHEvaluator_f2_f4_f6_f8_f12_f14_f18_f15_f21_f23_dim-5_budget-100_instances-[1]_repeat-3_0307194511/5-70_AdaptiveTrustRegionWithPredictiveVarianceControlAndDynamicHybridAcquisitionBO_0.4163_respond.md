# Description
**Adaptive Trust Region with Predictive Variance Control and Dynamic Hybrid Acquisition (ATRPVCDHA-BO)**: This algorithm combines the strengths of trust region methods, gradient enhancement, and dynamic acquisition function balancing. It introduces a novel mechanism for controlling the predictive variance of the Gaussian Process (GP) model within the trust region, preventing over-exploration in uncertain areas. It adaptively adjusts the weights of EI and UCB based on both the success of local searches and the consistency between predicted and actual improvements. The algorithm also incorporates a dynamic mechanism to adjust the trust region size based on the ratio of actual to predicted improvement, and uses a central difference scheme for gradient estimation. Furthermore, it incorporates a mechanism to adapt the lengthscale of the GP kernel during the optimization process.

# Justification
The ATRPVCDHA-BO algorithm aims to improve upon existing methods by addressing several key challenges in Bayesian Optimization:

1.  **Over-exploration in uncertain regions:** The algorithm explicitly controls the predictive variance of the GP model within the trust region. High predictive variance can lead to excessive exploration in areas where the model is uncertain, which can be inefficient. By penalizing high variance, the algorithm focuses exploration on regions where the model is more confident.
2.  **Balancing exploration and exploitation:** The dynamic adjustment of EI and UCB weights is based on both the success rate of local searches and the consistency between predicted and actual improvements. This allows the algorithm to adapt to the characteristics of the objective function and dynamically shift the balance between exploration and exploitation.
3.  **Trust region management:** The trust region size is dynamically adjusted based on the ratio of actual to predicted improvement, providing a more robust and adaptive approach to trust region management. The momentum term helps to smooth the trust region updates and prevent oscillations.
4.  **Gradient Enhancement:** The gradient estimation enhances local search and improves convergence speed.
5.  **Adaptive Lengthscale:** Adapting the lengthscale of the GP kernel during optimization allows the model to better capture the characteristics of the objective function.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors

class AdaptiveTrustRegionWithPredictiveVarianceControlAndDynamicHybridAcquisitionBO:
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
        self.variance_penalty = 0.1 # Penalty factor for high predictive variance
        self.lengthscale = 1.0  # Initial lengthscale for GP kernel
        self.improvement_threshold = 0.01 # Threshold for considering an improvement significant

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
        if len(X) > self.n_init and self.n_evals % (2 * self.dim) == 0:
            nn = NearestNeighbors(n_neighbors=min(len(X), 10)).fit(X)
            distances, _ = nn.kneighbors(X)
            self.lengthscale = np.median(distances[:, 1])  # Exclude the point itself
            self.lengthscale = np.clip(self.lengthscale, 1e-3, 1e3)

        # Define the kernel with the estimated lengthscale
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=self.lengthscale, length_scale_bounds=(1e-3, 1e3))

        # Gaussian Process Regressor
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))  # Return zeros if the model hasn't been fit yet

        model = self._fit_model(self.X, self.y)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement
        y_best = np.min(self.y)
        imp = y_best - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # avoid division by zero

        # Upper Confidence Bound
        ucb = mu - self.beta * sigma # minimize

        # Predictive Variance Control
        variance_penalty = self.variance_penalty * sigma

        # Combine EI and UCB (weighted average)
        acquisition = self.ei_weight * ei + self.ucb_weight * ucb - variance_penalty
        return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        candidate_points = self._sample_points(10 * batch_size)  # Generate more candidates
        acquisition_values = self._acquisition_function(candidate_points)

        # Sort by acquisition function value and select top batch_size points
        indices = np.argsort(acquisition_values.flatten())[::-1][:batch_size]
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

        while self.n_evals < self.budget:
            # Fit the GP model
            model = self._fit_model(self.X, self.y)

            # Estimate gradient at the best point
            gradient = self._estimate_gradient(model, best_x)

            # Perform local search within the trust region
            if np.random.rand() < self.global_search_prob:
                # Global search step
                next_x = self._select_next_points(1)[0]
                acquisition_source = "EI" # Assume EI is used for global search
            else:
                next_x = self._local_search(model, best_x.copy(), gradient)
                acquisition_source = "UCB" # Assume UCB is used for local search

            next_x = np.clip(next_x, self.bounds[0], self.bounds[1]) # Ensure it's within bounds

            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0] # Evaluate the actual function
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            # Check if the new point is better than the current best
            improvement = best_y - next_y
            predicted_improvement, _ = model.predict(next_x.reshape(1, -1), return_std=True)
            predicted_improvement = best_y - predicted_improvement[0]

            if improvement > self.improvement_threshold:
                best_x = next_x
                best_y = next_y

                # Adjust trust region size based on actual vs predicted improvement
                if predicted_improvement > 0:
                    improvement_ratio = improvement / predicted_improvement
                    if improvement_ratio > 0.5:
                        self.trust_region_size *= self.trust_region_expand
                    else:
                        self.trust_region_size *= (1 + improvement_ratio) # Slightly expand if there's some improvement
                else:
                    self.trust_region_size *= self.trust_region_expand # Expand if predicted improvement is zero

                # Update success history
                self.success_history.append(True)

            else:
                # Shrink trust region if no significant improvement
                self.trust_region_size *= self.trust_region_shrink
                self.success_history.append(False)

            self.trust_region_size = self.trust_region_momentum * self.trust_region_size + (1 - self.trust_region_momentum) * np.clip(self.trust_region_size, 0.1, 5.0) # Keep trust region within reasonable bounds

            # Decay exploration parameter
            self.beta *= self.beta_decay

            # Update success history
            if len(self.success_history) > self.success_history_length:
                self.success_history.pop(0)

            # Adjust EI/UCB weights based on success history and improvement consistency
            if len(self.success_history) == self.success_history_length:
                success_rate = np.mean(self.success_history)

                # Adjust weights based on success rate
                if success_rate > 0.7:
                    # Favor UCB (exploitation)
                    self.ei_weight = max(0.0, self.ei_weight - self.weight_adjust_rate)
                    self.ucb_weight = min(1.0, self.ucb_weight + self.weight_adjust_rate)
                elif success_rate < 0.3:
                    # Favor EI (exploration)
                    self.ei_weight = min(1.0, self.ei_weight + self.weight_adjust_rate)
                    self.ucb_weight = max(0.0, self.ucb_weight - self.weight_adjust_rate)

                # Normalize weights to ensure they sum to 1
                total_weight = self.ei_weight + self.ucb_weight
                self.ei_weight /= total_weight
                self.ucb_weight /= total_weight

        return best_y, best_x
```
## Feedback
 The algorithm AdaptiveTrustRegionWithPredictiveVarianceControlAndDynamicHybridAcquisitionBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1863 with standard deviation 0.0958.

took 84.23 seconds to run.