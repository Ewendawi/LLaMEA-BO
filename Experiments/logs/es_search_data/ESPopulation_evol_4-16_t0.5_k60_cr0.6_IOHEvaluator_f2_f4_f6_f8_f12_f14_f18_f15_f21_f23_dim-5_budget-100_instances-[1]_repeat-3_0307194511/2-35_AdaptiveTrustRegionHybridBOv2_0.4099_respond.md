# Description
**AdaptiveTrustRegionHybridBOv2**: This algorithm builds upon the AdaptiveTrustRegionHybridBO by incorporating a more robust trust region update mechanism and an improved local search strategy. The trust region size is now adjusted based on the ratio of predicted improvement to actual improvement, providing a more nuanced adaptation. The local search is enhanced by using the acquisition function to guide the search within the trust region, balancing exploitation and exploration more effectively. Additionally, a global search step is introduced with a small probability to escape local optima.

# Justification
The key improvements are:

1.  **Improved Trust Region Adaptation:** The trust region size adjustment is refined to be more responsive to the accuracy of the GP model's predictions. By comparing the predicted improvement with the actual improvement, the algorithm can more accurately determine whether to expand or shrink the trust region. This helps in avoiding premature convergence or inefficient exploration.
2.  **Acquisition Function Guided Local Search:** Instead of randomly sampling points within the trust region, the acquisition function is used to select candidate points. This focuses the local search on regions with high potential for improvement, leading to faster convergence.
3.  **Probabilistic Global Search:** To prevent the algorithm from getting stuck in local optima, a small probability is introduced for performing a global search step. This involves sampling a new point from the entire search space and evaluating it. This adds a degree of exploration to the algorithm, helping it to escape local optima.
4.  **Budget Aware Local Search:** The number of local search points is adjusted based on the remaining budget to ensure that the algorithm does not exceed the budget.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors

class AdaptiveTrustRegionHybridBOv2:
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
        self.global_search_prob = 0.05 # Probability of performing a global search

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

    def _local_search(self, model, center, n_points=50):
        # Perform local search within the trust region using the GP model and acquisition function
        # Generate candidate points within the trust region
        candidate_points = center + self.trust_region_size * np.random.uniform(-1, 1, size=(n_points, self.dim))

        # Clip the candidate points to stay within the bounds
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])

        # Calculate acquisition function values for candidate points
        acquisition_values = self._acquisition_function(candidate_points)

        # Select the point with the highest acquisition function value
        best_index = np.argmax(acquisition_values)
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

            # Perform local search within the trust region
            if np.random.rand() < self.global_search_prob:
                # Global search step
                next_x = self._sample_points(1)[0]
            else:
                # Local search step
                n_local_search_points = min(50, self.budget - self.n_evals) # Adjust based on remaining budget
                next_x = self._local_search(model, best_x.copy(), n_local_search_points)
            next_x = np.clip(next_x, self.bounds[0], self.bounds[1]) # Ensure it's within bounds

            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0] # Evaluate the actual function
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            # Check if the new point is better than the current best
            if next_y < best_y:
                # Calculate predicted improvement
                predicted_y, _ = model.predict(next_x.reshape(1, -1), return_std=True)
                predicted_improvement = best_y - predicted_y[0]

                # Calculate actual improvement
                actual_improvement = best_y - next_y

                # Adjust trust region size based on the ratio of predicted to actual improvement
                if actual_improvement > 0:
                    improvement_ratio = predicted_improvement / actual_improvement
                    if improvement_ratio < 0.25:
                        self.trust_region_size *= self.trust_region_expand
                    elif improvement_ratio > 4:
                        self.trust_region_size *= self.trust_region_shrink
                else:
                    self.trust_region_size *= self.trust_region_shrink # Shrink if no improvement

                best_x = next_x
                best_y = next_y

            else:
                # Shrink trust region if no improvement
                self.trust_region_size *= self.trust_region_shrink

            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0) # Keep trust region within reasonable bounds

        return best_y, best_x
```
## Feedback
 The algorithm AdaptiveTrustRegionHybridBOv2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1813 with standard deviation 0.1017.

took 125.69 seconds to run.