# Description
**AdaptiveTrustRegionOptimisticHybridBO**: This algorithm combines the strengths of AdaptiveTrustRegionBO and TrustRegionOptimisticBO, further enhanced with dynamic adjustment of local search points and trust region adaptation based on both improvement ratio and UCB value. It uses a Gaussian Process (GP) surrogate model with an adaptive Trust Region approach for local exploitation and an Upper Confidence Bound (UCB) acquisition function with a dynamic exploration parameter for global exploration. The number of local search points is dynamically adjusted based on the trust region size. The trust region size is adapted based on the success of the local search and the UCB value of the new point. A global search step is performed with a small probability to avoid getting stuck in local optima.

# Justification
This algorithm builds upon the strengths of AdaptiveTrustRegionBO and TrustRegionOptimisticBO.

1.  **Trust Region with Adaptive Size:** The trust region approach balances exploration and exploitation. The size of the trust region is dynamically adjusted based on the ratio of actual to predicted improvement, as in AdaptiveTrustRegionBO, but also incorporates the UCB value. If the UCB value is high, it suggests the region is promising, and the trust region is expanded even if the immediate improvement is not significant. This helps to escape local optima.

2.  **Optimistic Exploration with UCB:** The UCB acquisition function from TrustRegionOptimisticBO encourages exploration by considering both the predicted mean and the uncertainty (standard deviation) of the GP model. The exploration parameter `beta` is decayed over time to shift the focus from exploration to exploitation as the algorithm progresses.

3.  **Adaptive Local Search Points:** The number of points used in the local search is dynamically adjusted based on the trust region size. A larger trust region implies a greater need for exploration within that region, hence more points are sampled. This is computationally efficient, as it avoids unnecessary evaluations when the trust region is small.

4.  **Global Search Probability:** A small probability of performing a global search step, as in AdaptiveTrustRegionBO, helps to avoid getting stuck in local optima.

5. **Efficient Gradient Estimation**: The algorithm uses Expected Improvement (EI) instead of UCB for the acquisition function to select points for global search.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

class AdaptiveTrustRegionOptimisticHybridBO:
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
        self.global_search_prob = 0.05 # Probability of performing a global search step
        self.local_search_exploitation = 0.8

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
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
        model.fit(X, y)
        return model

    def _acquisition_function_ucb(self, X):
        # Implement UCB acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        model = self._fit_model(self.X, self.y)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Upper Confidence Bound
        ucb = mu - self.beta * sigma # minimize
        return ucb

    def _acquisition_function_ei(self, X):
        # Implement Expected Improvement acquisition function
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
        # Select the next points to evaluate using EI
        candidate_points = self._sample_points(10 * batch_size)
        acquisition_values = self._acquisition_function_ei(candidate_points)
        indices = np.argsort(acquisition_values.flatten())[::-1][:batch_size] # maximize EI
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
            self.y = np.vstack((self.y, new_X))

        # Update best seen point
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]

    def _local_search(self, model, center, n_points=50):
        # Perform local search within the trust region using the GP model and UCB
        # Generate candidate points within the trust region
        candidate_points = center + self.trust_region_size * np.random.uniform(-1, 1, size=(n_points, self.dim))

        # Clip the candidate points to stay within the bounds
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])

        # Predict the mean values using the GP model
        mu, sigma = model.predict(candidate_points, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Calculate UCB values
        ucb_values = self._acquisition_function_ucb(candidate_points)

        # Combine predicted mean and acquisition function values
        weighted_values = self.local_search_exploitation * mu + (1 - self.local_search_exploitation) * ucb_values # Minimize mu, minimize UCB

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

            # Dynamically adjust the number of local search points
            n_local_points = int(50 * (self.trust_region_size / 2.0))  # Scale points with trust region size
            n_local_points = np.clip(n_local_points, 10, 100)  # Ensure reasonable range

            # Perform global search with a small probability
            if np.random.rand() < self.global_search_prob:
                next_x = self._select_next_points(1)[0] # Select a point using EI
            else:
                # Perform local search within the trust region
                next_x = self._local_search(model, best_x.copy(), n_local_points)
            next_x = np.clip(next_x, self.bounds[0], self.bounds[1]) # Ensure it's within bounds

            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0] # Evaluate the actual function
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            # Calculate the improvement
            improvement = best_y - next_y

            # Predict the improvement using the GP model
            predicted_y, _ = model.predict(next_x.reshape(1, -1), return_std=True)
            predicted_improvement = best_y - predicted_y[0]
            ucb_value = self._acquisition_function_ucb(next_x.reshape(1, -1))[0,0]

            # Adjust trust region size based on the ratio of actual to predicted improvement and UCB value
            if predicted_improvement != 0:
                ratio = improvement / predicted_improvement
                if ratio > 0.5 or ucb_value < 0:  # Also expand if UCB is promising
                    self.trust_region_size *= self.trust_region_expand
                else:
                    self.trust_region_size *= self.trust_region_shrink
            else:
                # If predicted improvement is zero, shrink the trust region
                self.trust_region_size *= self.trust_region_shrink

            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0) # Keep trust region within reasonable bounds

            # Decay exploration parameter
            self.beta *= self.beta_decay

            # Check if the new point is better than the current best
            if next_y < best_y:
                best_x = next_x
                best_y = next_y

        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<AdaptiveTrustRegionOptimisticHybridBO>", line 165, in __call__
 165->             self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))
  File "<AdaptiveTrustRegionOptimisticHybridBO>", line 100, in _update_eval_points
  98 |         else:
  99 |             self.X = np.vstack((self.X, new_X))
 100->             self.y = np.vstack((self.y, new_X))
 101 | 
 102 |         # Update best seen point
  File "<__array_function__ internals>", line 200, in vstack
  File "<__array_function__ internals>", line 200, in concatenate
ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 1 and the array at index 1 has size 5
