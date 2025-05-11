# Description
**Adaptive Hybrid Bayesian Optimization with Dynamic Exploration, Adaptive Batch Size, and Improved Local Search with Momentum (AHBBO_ABSLS_V3):** This algorithm builds upon AHBBO_ABSLS_V2 by incorporating momentum into the local search strategy and refining the exploration-exploitation balance. The local search uses a momentum term to accelerate convergence and escape local optima. The exploration weight decay schedule is further refined to consider not only the number of evaluations and GPR model uncertainty but also the improvement rate in the best-found solution. This allows for a more aggressive exploration in the early stages and a more focused exploitation as the optimization progresses. Furthermore, the local search radius is also adapted based on the improvement rate, shrinking faster if the improvement stagnates.

# Justification
The key improvements are:

1.  **Momentum in Local Search:** Adding momentum to the local search helps to accelerate convergence and escape local optima. The momentum term allows the search to continue in a promising direction, even if the current gradient is small or noisy. This is particularly useful in high-dimensional spaces where the landscape can be complex.
2.  **Refined Exploration Weight Decay:** The exploration weight decay is refined to consider the improvement rate in the best-found solution. This allows for a more adaptive exploration-exploitation balance. In the early stages, when the improvement rate is high, the algorithm can afford to explore more aggressively. As the optimization progresses and the improvement rate slows down, the algorithm can focus on exploitation.
3.  **Adaptive Local Search Radius based on Improvement Rate:** Adapting the local search radius based on the improvement rate allows for a more efficient local search. If the improvement stagnates, the local search radius is shrunk faster, allowing the algorithm to explore other regions of the search space.

These changes are designed to improve the convergence speed and the quality of the final solution, while maintaining computational efficiency.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class AHBBO_ABSLS_V3:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim # Initial number of points

        self.best_y = np.inf
        self.best_x = None
        self.previous_best_y = np.inf

        self.max_batch_size = min(10, dim) # Maximum batch size for selecting points
        self.min_batch_size = 1
        self.exploration_weight = 0.2 # Initial exploration weight
        self.exploration_weight_min = 0.01 # Minimum exploration weight
        self.uncertainty_threshold = 0.5  # Threshold for adjusting batch size
        self.initial_local_search_radius = 0.2
        self.local_search_radius = self.initial_local_search_radius
        self.local_search_radius_min = 0.01
        self.local_search_momentum = 0.1
        self.local_search_velocity = np.zeros(dim)

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
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Distance-based exploration term
        if self.X is not None:
            min_dist = np.min(np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2), axis=1, keepdims=True)
            exploration = min_dist / np.max(min_dist)
        else:
            exploration = np.ones(X.shape[0])[:,None]

        # Hybrid acquisition function
        acquisition = ei + self.exploration_weight * exploration
        return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Generate candidate points
        candidate_points = self._sample_points(50 * batch_size)  # Generate more candidates

        # Add points around the best solution (local search)
        if self.best_x is not None:
            # Local search with momentum
            self.local_search_velocity = self.local_search_momentum * self.local_search_velocity + (1 - self.local_search_momentum) * np.random.normal(loc=0, scale=self.local_search_radius, size=(self.dim))
            local_points = np.random.normal(loc=self.best_x + self.local_search_velocity, scale=self.local_search_radius, size=(50 * batch_size, self.dim))
            local_points = np.clip(local_points, self.bounds[0], self.bounds[1])
            candidate_points = np.vstack((candidate_points, local_points))

        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_points)

        # Select the top batch_size points with the highest acquisition values
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]

        return next_points

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

        # Update best seen value
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Optimization
            # select points by acquisition function

            # Adjust batch size based on uncertainty
            _, sigma = self.model.predict(self.X, return_std=True)
            avg_sigma = np.mean(sigma)

            if avg_sigma > self.uncertainty_threshold:
                batch_size = self.max_batch_size
            else:
                batch_size = self.min_batch_size

            remaining_evals = self.budget - self.n_evals
            batch_size = min(batch_size, remaining_evals) # Adjust batch size to budget

            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self.model = self._fit_model(self.X, self.y)

            # Calculate improvement rate
            improvement_rate = (self.previous_best_y - self.best_y) / self.previous_best_y if self.previous_best_y != 0 else 0
            self.previous_best_y = self.best_y

            # Update exploration weight
            self.exploration_weight = max(self.exploration_weight_min, self.exploration_weight * (1 - self.n_evals / self.budget) * (1 + avg_sigma) * (1 + improvement_rate))

            # Update local search radius
            self.local_search_radius = max(self.local_search_radius_min, self.initial_local_search_radius * (1 - self.n_evals / self.budget) * (1 + avg_sigma) * (1 - improvement_rate))

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AHBBO_ABSLS_V3 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1723 with standard deviation 0.0996.

took 311.64 seconds to run.