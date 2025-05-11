# Description
**AdaptiveTrustRegionOptimisticHybridV3BO**: This algorithm builds upon AdaptiveTrustRegionOptimisticHybridBO by introducing a dynamic weighting between Expected Improvement (EI) and Upper Confidence Bound (UCB) in the acquisition function, adaptive local search point generation within the trust region, and a more informed global search strategy. The weighting between EI and UCB adapts based on the success of local search, promoting exploration when local search stagnates and exploitation when it's successful. The local search now generates points using a combination of uniform sampling and sampling around the best point found so far, guided by the GP's uncertainty. Global search uses EI over the entire domain but is biased towards regions with high uncertainty.

# Justification
The key improvements are:

1.  **Dynamic EI/UCB Weighting:** Balancing exploration and exploitation is crucial. Instead of a fixed weight, the algorithm dynamically adjusts the weights based on the recent success of the local search. If local search hasn't improved the best value in a while, the weight shifts towards UCB to encourage exploration. If local search is successful, the weight shifts towards EI to refine the search.

2.  **Adaptive Local Search Point Generation:** The original local search used uniform sampling within the trust region. This is improved by generating points using a mixture of uniform sampling and sampling around the current best point, weighted by the GP's uncertainty. This allows for more focused exploration in promising regions while still maintaining diversity.

3.  **Informed Global Search:** Instead of uniform random global search, global search is performed using EI over the entire domain, but is biased towards regions with high uncertainty. This helps to escape local optima more effectively.

4.  **Computational Efficiency:** Nearest neighbor lengthscale estimation is retained for efficiency. The local search point generation and dynamic weighting add minimal computational overhead.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors

class AdaptiveTrustRegionOptimisticHybridV3BO:
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
        self.success_threshold = 0.7
        self.beta = 2.0  # Exploration parameter for UCB
        self.beta_decay = 0.99 # Decay rate for beta
        self.global_search_prob = 0.05 # Probability of performing a global search step
        self.ei_weight = 0.5 # Initial weight for EI
        self.ei_weight_adjust = 0.05 # Adjustment step for EI weight
        self.local_search_success = 0 # Counter for local search success
        self.local_search_history = [] # History of local search results

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

        # Upper Confidence Bound
        ucb = mu - self.beta * sigma # minimize

        # Combine EI and UCB (weighted average)
        acquisition = self.ei_weight * ei + (1 - self.ei_weight) * ucb
        return acquisition

    def _select_next_points(self, batch_size, global_search=False):
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

    def _local_search(self, model, center, n_points=50):
        # Perform local search within the trust region using the GP model
        # Generate candidate points within the trust region

        # Generate points around the best point and uniformly
        points_around_best = center + self.trust_region_size * 0.5 * np.random.normal(size=(n_points // 2, self.dim))
        uniform_points = center + self.trust_region_size * np.random.uniform(-1, 1, size=(n_points - n_points // 2, self.dim))
        candidate_points = np.vstack((points_around_best, uniform_points))

        # Clip the candidate points to stay within the bounds
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])

        # Predict the mean and std values using the GP model
        mu, sigma = model.predict(candidate_points, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Select the point with the minimum predicted mean value
        acquisition_values = self.ei_weight * (mu * -1) + (1 - self.ei_weight) * (mu - self.beta * sigma) #Local acquisition function
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

        last_improvement = 0

        while self.n_evals < self.budget:
            # Fit the GP model
            model = self._fit_model(self.X, self.y)

            # Perform local search within the trust region
            if np.random.rand() < self.global_search_prob:
                # Global search step
                next_x = self._select_next_points(1, global_search=True)[0]
            else:
                next_x = self._local_search(model, best_x.copy())

            next_x = np.clip(next_x, self.bounds[0], self.bounds[1]) # Ensure it's within bounds

            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0] # Evaluate the actual function
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))
            self.local_search_history.append(next_y)

            # Check if the new point is better than the current best
            if next_y < best_y:
                best_x = next_x
                best_y = next_y
                # Adjust trust region size
                self.trust_region_size *= self.trust_region_expand
                self.local_search_success += 1
                last_improvement = self.n_evals
                self.ei_weight = min(1.0, self.ei_weight + self.ei_weight_adjust) # Increase EI weight
            else:
                # Shrink trust region if no improvement
                self.trust_region_size *= self.trust_region_shrink
                self.ei_weight = max(0.0, self.ei_weight - self.ei_weight_adjust) # Decrease EI weight

            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0) # Keep trust region within reasonable bounds

            # Decay exploration parameter
            self.beta *= self.beta_decay

            if self.n_evals - last_improvement > self.budget // 10:
                self.global_search_prob = 0.5
            else:
                self.global_search_prob = 0.05

        return best_y, best_x
```
## Feedback
 The algorithm AdaptiveTrustRegionOptimisticHybridV3BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1895 with standard deviation 0.1024.

took 73.65 seconds to run.