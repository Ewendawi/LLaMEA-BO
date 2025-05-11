# Description
**Adaptive Trust Region with Dynamic Acquisition and Lengthscale Adaptation BO (ATRDALA-BO)**: This algorithm combines the strengths of adaptive trust region management, dynamic acquisition function balancing, and adaptive lengthscale control for improved Bayesian optimization. It uses gradient-enhanced local search within a trust region, dynamically adjusts the weights of EI and UCB based on their success, and adapts the lengthscale of the GP kernel during optimization. The trust region size is adapted based on the agreement between predicted and actual improvement. The gradient estimation uses an adaptive step size based on the current lengthscale. A key innovation is the introduction of a separate success rate for EI and UCB *within* the trust region to better guide the acquisition function balancing.

# Justification
This algorithm aims to improve performance by combining several successful strategies:

*   **Adaptive Trust Region:** The trust region approach helps to focus the search in promising areas, balancing exploration and exploitation. Adapting the trust region size based on the success of previous steps ensures efficient exploration.
*   **Dynamic Acquisition Balancing (EI/UCB):** Dynamically adjusting the weights of EI and UCB allows the algorithm to adapt to different stages of the optimization process. Initially, more exploration (EI) might be beneficial, while later stages may benefit from more exploitation (UCB). Tracking separate success rates *within the trust region* provides a more fine-grained control over this balance.
*   **Adaptive Lengthscale:** Adapting the lengthscale of the GP kernel during optimization allows the model to better capture the characteristics of the objective function. This is particularly important for non-stationary functions. The nearest neighbors approach provides a computationally efficient way to estimate the lengthscale.
*   **Gradient-Enhanced Local Search:** Using gradient information in the local search helps to accelerate convergence. The adaptive step size for gradient estimation improves the accuracy of the gradient estimate, especially when the lengthscale changes.
*   **Global Search Probability:** The global search step prevents the algorithm from getting stuck in local optima.

The combination of these strategies should result in a robust and efficient Bayesian optimization algorithm that can handle a wide range of black-box optimization problems. The separate success rate tracking for EI and UCB within the trust region is a novel aspect that should lead to more effective acquisition function balancing.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors

class AdaptiveTrustRegionDynamicAcquisitionLengthscaleBO:
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
        self.ei_weight = 0.5
        self.ucb_weight = 0.5
        self.ei_success = 0
        self.ucb_success = 0
        self.weight_update_rate = 0.1
        self.lengthscale = 1.0 # Initial lengthscale
        self.best_x = None
        self.best_y = np.inf

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

        ucb = mu - self.beta * sigma # minimize

        return ucb

    def _mixed_acquisition(self, X):
        # Combine EI and UCB with dynamic weights
        ei = self._acquisition_function_ei(X)
        ucb = self._acquisition_function_ucb(X)
        return self.ei_weight * ei + self.ucb_weight * ucb

    def _select_next_points(self, batch_size, acquisition_function):
        # Select the next points to evaluate
        candidate_points = self._sample_points(10 * batch_size)  # Generate more candidates
        acquisition_values = acquisition_function(candidate_points)

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

        # Update best seen point
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]

    def _estimate_gradient(self, model, x):
        # Estimate the gradient of the function at point x using central finite differences
        gradient = np.zeros(self.dim)
        delta = min(self.lengthscale, self.delta) # Adaptive delta

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

            # Global search with probability global_search_prob
            if np.random.rand() < self.global_search_prob:
                next_x = self._select_next_points(1, self._acquisition_function_ei)[0] # Select a point using EI for global search
                acquisition_source = "EI"
            else:
                # Estimate gradient at the best point
                gradient = self._estimate_gradient(model, best_x)

                # Perform local search within the trust region
                next_x = self._local_search(model, best_x.copy(), gradient)
                next_x = np.clip(next_x, self.bounds[0], self.bounds[1]) # Ensure it's within bounds
                acquisition_source = "UCB"

            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0] # Evaluate the actual function
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            # Check if the new point is better than the current best
            improvement = (best_y - next_y) / best_y if best_y != 0 else (best_y - next_y)
            if next_y < best_y:
                best_x = next_x
                best_y = next_y

                # Adjust trust region size
                self.trust_region_size *= self.trust_region_expand

                # Update EI/UCB success counters (within trust region)
                if acquisition_source == "EI":
                    self.ei_success += 1
                else:
                    self.ucb_success += 1

            else:
                # Shrink trust region if no improvement
                self.trust_region_size *= self.trust_region_shrink

            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0) # Keep trust region within reasonable bounds

            # Decay exploration parameter
            self.beta *= self.beta_decay

            # Update EI/UCB weights
            total_success = self.ei_success + self.ucb_success
            if total_success > 10: # start updating weights after some evaluations
                ei_ratio = self.ei_success / total_success
                ucb_ratio = self.ucb_success / total_success

                self.ei_weight = (1 - self.weight_update_rate) * self.ei_weight + self.weight_update_rate * ei_ratio
                self.ucb_weight = (1 - self.weight_update_rate) * self.ucb_weight + self.weight_update_rate * ucb_ratio

                self.ei_weight = np.clip(self.ei_weight, 0.1, 0.9)
                self.ucb_weight = np.clip(self.ucb_weight, 0.1, 0.9)

                # Reset success counters
                self.ei_success = 0
                self.ucb_success = 0

        return best_y, best_x
```
## Feedback
 The algorithm AdaptiveTrustRegionDynamicAcquisitionLengthscaleBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1863 with standard deviation 0.1125.

took 83.20 seconds to run.