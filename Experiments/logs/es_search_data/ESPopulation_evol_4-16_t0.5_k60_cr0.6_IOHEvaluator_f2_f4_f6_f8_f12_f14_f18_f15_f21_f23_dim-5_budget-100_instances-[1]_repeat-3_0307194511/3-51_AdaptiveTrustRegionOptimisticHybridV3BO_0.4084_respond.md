# Description
**AdaptiveTrustRegionOptimisticHybridV3BO**: This algorithm builds upon `AdaptiveTrustRegionOptimisticHybridBO` by introducing several key enhancements. First, it incorporates a dynamic adjustment of the local search's sample size based on the success rate of the local search. If the local search consistently finds better points, the sample size is increased to further exploit the region. Conversely, if the local search stagnates, the sample size is decreased to reduce unnecessary evaluations. Second, instead of relying solely on UCB for local search, it blends UCB with the GP's predicted mean, with an adaptive weight that shifts from exploration to exploitation as the search progresses. Finally, a more sophisticated trust region adaptation strategy is implemented, using a momentum-based approach and considering the distance to the nearest evaluated point.

# Justification
The core idea is to make the local search more efficient and adaptive. The dynamic sample size adjustment ensures that we spend more evaluations in promising regions and fewer in stagnant ones. Blending UCB with the GP mean allows for a smoother transition from exploration to exploitation within the trust region. The momentum-based trust region adaptation helps to avoid oscillations in the trust region size and provides more stability. The distance to the nearest evaluated point is incorporated into the trust region update to prevent the algorithm from getting stuck in areas that have already been thoroughly explored.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

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
        self.local_search_n_points = 50
        self.local_search_n_points_min = 10
        self.local_search_n_points_max = 100
        self.local_search_success_rate = 0.0
        self.local_search_success_history = []
        self.mean_weight = 0.5 # Weight for GP mean in local search, dynamically adjusted

        self.trust_region_momentum = 0.5  # Momentum for trust region update
        self.prev_trust_region_size = self.trust_region_size

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

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using EI
        candidate_points = self._sample_points(10 * batch_size)
        acquisition_values = self._acquisition_function_ei(candidate_points)

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

    def _local_search(self, model, center):
        # Perform local search within the trust region using the GP model and UCB
        n_points = int(self.local_search_n_points)
        candidate_points = center + self.trust_region_size * np.random.uniform(-1, 1, size=(n_points, self.dim))
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])

        # Predict the mean and std using the GP model
        mu, sigma = model.predict(candidate_points, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Blend UCB with GP mean
        blended_values = self.mean_weight * mu - (1 - self.mean_weight) * self.beta * sigma # minimize

        # Select the point with the minimum blended value
        best_index = np.argmin(blended_values)
        best_point = candidate_points[best_index]

        return best_point

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
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
            next_x = self._local_search(model, best_x.copy())
            next_x = np.clip(next_x, self.bounds[0], self.bounds[1])

            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0]
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            # Check if the new point is better than the current best
            if next_y < best_y:
                best_x = next_x
                best_y = next_y
                # Adjust trust region size
                self.trust_region_size = self.trust_region_momentum * self.trust_region_size + (1 - self.trust_region_momentum) * (self.trust_region_size * self.trust_region_expand)
                self.local_search_success_history.append(1)
            else:
                # Shrink trust region if no improvement
                self.trust_region_size = self.trust_region_momentum * self.trust_region_size + (1 - self.trust_region_momentum) * (self.trust_region_size * self.trust_region_shrink)
                self.local_search_success_history.append(0)

            # Adjust local search sample size
            if len(self.local_search_success_history) > 5:
                self.local_search_success_rate = np.mean(self.local_search_success_history[-5:])
                if self.local_search_success_rate > self.success_threshold:
                    self.local_search_n_points = min(self.local_search_n_points * 1.2, self.local_search_n_points_max)
                else:
                    self.local_search_n_points = max(self.local_search_n_points * 0.8, self.local_search_n_points_min)

            # Decay exploration parameter
            self.beta *= self.beta_decay

            # Adjust mean weight
            self.mean_weight = min(self.mean_weight + 0.05, 1.0) # Shift towards exploitation

            # Distance to nearest evaluated point
            distances = cdist(next_x.reshape(1, -1), self.X)
            min_distance = np.min(distances)
            self.trust_region_size = min(self.trust_region_size, min_distance)


            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0)

        return best_y, best_x
```
## Feedback
 The algorithm AdaptiveTrustRegionOptimisticHybridV3BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1829 with standard deviation 0.1231.

took 73.11 seconds to run.