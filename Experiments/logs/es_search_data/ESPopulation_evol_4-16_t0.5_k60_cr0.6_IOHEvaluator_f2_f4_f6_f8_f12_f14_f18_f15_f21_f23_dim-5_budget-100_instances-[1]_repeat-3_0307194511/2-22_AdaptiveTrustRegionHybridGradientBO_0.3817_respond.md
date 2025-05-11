# Description
**AdaptiveTrustRegionHybridGradientBO**: This algorithm builds upon `AdaptiveTrustRegionHybridBO` and `AdaptiveTrustRegionBO` by incorporating gradient information into the local search and trust region adaptation. It combines the efficient lengthscale estimation from `AdaptiveTrustRegionHybridBO` with the adaptive trust region sizing and global search probability from `AdaptiveTrustRegionBO`. Furthermore, it estimates the gradient at the current best point using a finite difference method *within the trust region* to refine the local search. The gradient information is incorporated into the local search by biasing the candidate points towards the direction of the negative gradient. To reduce the number of function evaluations for gradient estimation, a Gaussian Process model is used to approximate the function values required for the finite difference calculation. The trust region size is adapted based on both the success of the local search and the agreement between the predicted and actual improvement.

# Justification
The key components and changes are justified as follows:

1.  **Gradient-Enhanced Local Search:** Incorporating gradient information into the local search allows for a more informed exploration of the trust region. By biasing the candidate points towards the negative gradient direction, the algorithm can more efficiently identify promising regions within the trust region, leading to faster convergence.
2.  **Efficient Gradient Estimation:** To minimize the number of function evaluations required for gradient estimation, a Gaussian Process model is used to approximate the function values needed for the finite difference calculation. This approach allows the algorithm to estimate the gradient with fewer function evaluations, making it more computationally efficient. The gradient is estimated *within the trust region* to ensure that the gradient estimate is relevant to the local landscape.
3.  **Adaptive Trust Region Sizing:** The trust region size is adapted based on both the success of the local search and the agreement between the predicted and actual improvement. This adaptive approach allows the algorithm to dynamically adjust the trust region size based on the characteristics of the objective function. If the predicted and actual improvements agree well, the trust region size is expanded, allowing for more aggressive exploration. Conversely, if the predicted and actual improvements disagree, the trust region size is shrunk, promoting more conservative exploration.
4.  **Global Search Probability:** To prevent the algorithm from getting stuck in local optima, a small probability of performing a global search step is introduced. This allows the algorithm to occasionally escape the current trust region and explore other regions of the search space.
5. **Nearest Neighbors for Lengthscale:** This is kept from `AdaptiveTrustRegionHybridBO` as it is computationally efficient and provides a good estimate for the GP lengthscale.
6. **Combining Strengths:** By combining the strengths of `AdaptiveTrustRegionHybridBO` and `AdaptiveTrustRegionBO`, the algorithm aims to achieve a better balance between exploration and exploitation, leading to improved performance on a wide range of optimization problems.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors

class AdaptiveTrustRegionHybridGradientBO:
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
        self.local_search_exploitation = 0.8  # Weight for exploitation in local search
        self.global_search_prob = 0.05 # Probability of performing a global search step
        self.gradient_estimation_points = 5 # Number of points to use for gradient estimation

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

        return ei

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

    def _estimate_gradient(self, model, center):
        # Estimate the gradient at the center point using finite differences
        gradient = np.zeros(self.dim)
        delta = self.trust_region_size / 10.0  # Step size for finite differences

        for i in range(self.dim):
            # Create a point slightly offset in the i-th dimension
            offset = np.zeros(self.dim)
            offset[i] = delta

            # Ensure the points for gradient estimation are within the bounds
            x_plus = np.clip(center + offset, self.bounds[0], self.bounds[1])
            x_minus = np.clip(center - offset, self.bounds[0], self.bounds[1])

            # Predict function values using the GP model
            y_plus, _ = model.predict(x_plus.reshape(1, -1), return_std=True)
            y_minus, _ = model.predict(x_minus.reshape(1, -1), return_std=True)

            # Estimate the partial derivative using central difference
            gradient[i] = (y_plus[0] - y_minus[0]) / (2 * delta)

        return gradient

    def _local_search(self, model, center, best_y, gradient):
        # Perform local search within the trust region using the GP model, acquisition function, and gradient information
        n_points = 50
        candidate_points = center + self.trust_region_size * np.random.uniform(-1, 1, size=(n_points, self.dim))

        # Bias candidate points towards the negative gradient direction
        candidate_points += 0.1 * self.trust_region_size * gradient

        # Clip the candidate points to stay within the bounds
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])

        # Predict the mean values using the GP model
        mu, _ = model.predict(candidate_points, return_std=True)
        mu = mu.reshape(-1, 1)

        # Calculate acquisition function values
        ei = self._acquisition_function(candidate_points)

        # Combine predicted mean, acquisition function values and gradient information
        weighted_values = self.local_search_exploitation * mu + (1 - self.local_search_exploitation) * (-ei) # Minimize mu, maximize EI

        # Select the point with the minimum weighted value
        best_index = np.argmin(weighted_values)
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

            # Estimate gradient at the current best point
            gradient = self._estimate_gradient(model, best_x.copy())

            # Perform global search with a small probability
            if np.random.rand() < self.global_search_prob:
                next_x = self._select_next_points(1)[0] # Select a point using the acquisition function
            else:
                # Perform local search within the trust region
                next_x = self._local_search(model, best_x.copy(), best_y, gradient)

            next_x = np.clip(next_x, self.bounds[0], self.bounds[1]) # Ensure it's within bounds

            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0] # Evaluate the actual function
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            # Calculate the improvement
            improvement = best_y - next_y

            # Predict the improvement using the GP model
            predicted_y, _ = model.predict(next_x.reshape(1, -1), return_std=True)
            predicted_improvement = best_y - predicted_y[0]

            # Adjust trust region size based on the ratio of actual to predicted improvement
            if predicted_improvement != 0:
                ratio = improvement / predicted_improvement
                if ratio > 0.5:
                    self.trust_region_size *= self.trust_region_expand
                else:
                    self.trust_region_size *= self.trust_region_shrink
            else:
                # If predicted improvement is zero, shrink the trust region
                self.trust_region_size *= self.trust_region_shrink

            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0) # Keep trust region within reasonable bounds

            # Check if the new point is better than the current best
            if next_y < best_y:
                best_x = next_x
                best_y = next_y

        return best_y, best_x
```
## Feedback
 The algorithm AdaptiveTrustRegionHybridGradientBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1606 with standard deviation 0.1208.

took 129.91 seconds to run.