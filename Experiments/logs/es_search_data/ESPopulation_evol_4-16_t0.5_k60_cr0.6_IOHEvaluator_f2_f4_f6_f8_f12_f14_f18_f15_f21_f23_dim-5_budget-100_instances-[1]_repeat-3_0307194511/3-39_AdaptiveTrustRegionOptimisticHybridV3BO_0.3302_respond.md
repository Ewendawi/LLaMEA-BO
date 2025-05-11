# Description
**AdaptiveTrustRegionOptimisticHybridV3BO**: This algorithm builds upon `AdaptiveTrustRegionOptimisticHybridBO` and `AdaptiveTrustRegionHybridV2BO`, combining their strengths. It incorporates adaptive trust region management, efficient lengthscale estimation using nearest neighbors, and a hybrid acquisition function using both Expected Improvement (EI) and Upper Confidence Bound (UCB) with a dynamic exploration parameter. The local search within the trust region leverages a weighted combination of the GP model's mean prediction and the acquisition function, with dynamically adjusted weights to balance exploration and exploitation. A global search step, optimized using L-BFGS-B, is probabilistically introduced to escape local optima. The trust region size, UCB's exploration parameter, and the local search exploitation weight are adaptively adjusted based on the success of local search. This version also includes a mechanism to restart the GP model with fresh samples if stagnation is detected, promoting better exploration.

# Justification
The algorithm combines the best aspects of its predecessors:

*   **Trust Region Management:** Adaptive trust region size adjustment based on local search success, as in `AdaptiveTrustRegionOptimisticHybridBO` and `AdaptiveTrustRegionHybridV2BO`.
*   **Efficient Lengthscale Estimation:** Nearest neighbors-based lengthscale estimation for the GP kernel, as in previous versions.
*   **Hybrid Acquisition:** Combines EI for global exploration and UCB for local search, as in `AdaptiveTrustRegionOptimisticHybridBO`.
*   **Adaptive Local Search:** Dynamically adjusts the weight given to the GP model versus the acquisition function during local search, as in `AdaptiveTrustRegionHybridV2BO`. This allows the algorithm to adapt its local search strategy based on the characteristics of the function being optimized.
*   **Refined Global Search:** Uses L-BFGS-B to optimize the acquisition function for global search, as in `AdaptiveTrustRegionHybridV2BO`, which is more effective than random sampling.
*   **Stagnation Detection and Restart:** Detects stagnation by monitoring the improvement in the best function value over a certain number of iterations. If stagnation is detected, the GP model is reset with fresh samples, encouraging exploration of new regions of the search space. This addresses a common issue in BO where the algorithm can get stuck in local optima.
*   **Computational Efficiency:** The algorithm avoids computationally expensive operations where possible, such as using nearest neighbors for lengthscale estimation and limiting the number of iterations in the L-BFGS-B optimization.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize

class AdaptiveTrustRegionOptimisticHybridV3BO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim
        self.batch_size = min(5, self.dim)
        self.trust_region_size = 2.0
        self.trust_region_shrink = 0.5
        self.trust_region_expand = 2.0
        self.local_search_exploitation = 0.8  # Initial weight for exploitation in local search
        self.local_search_exploitation_adjust = 0.05
        self.global_search_prob = 0.05
        self.success_threshold = 0.7
        self.beta = 2.0  # Exploration parameter for UCB
        self.beta_decay = 0.99
        self.stagnation_threshold = 10 # Number of iterations without improvement before restart
        self.stagnation_counter = 0
        self.best_y_so_far = float('inf')

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        nn = NearestNeighbors(n_neighbors=min(len(X), 10)).fit(X)
        distances, _ = nn.kneighbors(X)
        median_distance = np.median(distances[:, 1])

        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(
            length_scale=median_distance, length_scale_bounds=(1e-3, 1e3)
        )

        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
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

        ucb = mu - self.beta * sigma  # minimize

        # Combine EI and UCB (weighted average)
        acquisition = 0.5 * ei + 0.5 * ucb
        return acquisition

    def _select_next_points(self, batch_size):
        candidate_points = self._sample_points(10 * batch_size)
        acquisition_values = self._acquisition_function(candidate_points)
        indices = np.argsort(acquisition_values.flatten())[::-1][:batch_size]
        return candidate_points[indices]

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def _local_search(self, model, center, n_points=50):
        candidate_points = center + self.trust_region_size * np.random.uniform(-1, 1, size=(n_points, self.dim))
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])

        mu, _ = model.predict(candidate_points, return_std=True)
        mu = mu.reshape(-1, 1)
        ei = self._acquisition_function(candidate_points)

        weighted_values = self.local_search_exploitation * mu + (1 - self.local_search_exploitation) * (-ei)
        best_index = np.argmin(weighted_values)
        best_point = candidate_points[best_index]

        return best_point

    def _global_search(self):
        def objective(x):
            return -self._acquisition_function(x.reshape(1, -1))[0, 0]

        x0 = self._sample_points(1)[0]
        bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5})
        return result.x

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_index = np.argmin(self.y)
        best_x = self.X[best_index]
        best_y = self.y[best_index][0]
        self.best_y_so_far = best_y

        successful_local_searches = 0
        while self.n_evals < self.budget:
            model = self._fit_model(self.X, self.y)

            if np.random.rand() < self.global_search_prob:
                next_x = self._global_search()
            else:
                next_x = self._local_search(model, best_x.copy())
            next_x = np.clip(next_x, self.bounds[0], self.bounds[1])

            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0]
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            if next_y < best_y:
                best_x = next_x
                best_y = next_y
                self.trust_region_size *= self.trust_region_expand
                successful_local_searches += 1
                self.local_search_exploitation = min(1.0, self.local_search_exploitation + self.local_search_exploitation_adjust)
                self.stagnation_counter = 0 # Reset stagnation counter
            else:
                self.trust_region_size *= self.trust_region_shrink
                successful_local_searches = 0
                self.local_search_exploitation = max(0.0, self.local_search_exploitation - self.local_search_exploitation_adjust)
                self.stagnation_counter += 1 # Increment stagnation counter

            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0)
            self.beta *= self.beta_decay

            # Stagnation detection and restart
            if self.stagnation_counter > self.stagnation_threshold:
                print("Stagnation detected, restarting GP model")
                self.X = None
                self.y = None
                initial_X = self._sample_points(self.n_init)
                initial_y = self._evaluate_points(func, initial_X)
                self._update_eval_points(initial_X, initial_y)

                best_index = np.argmin(self.y)
                best_x = self.X[best_index]
                best_y = self.y[best_index][0]
                self.stagnation_counter = 0 # Reset stagnation counter

            if best_y < self.best_y_so_far:
                self.best_y_so_far = best_y
                
        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<AdaptiveTrustRegionOptimisticHybridV3BO>", line 159, in __call__
 159->                 initial_y = self._evaluate_points(func, initial_X)
  File "<AdaptiveTrustRegionOptimisticHybridV3BO>", line 79, in _evaluate_points
  79->         y = np.array([func(x) for x in X])
  File "<AdaptiveTrustRegionOptimisticHybridV3BO>", line 79, in <listcomp>
  77 | 
  78 |     def _evaluate_points(self, func, X):
  79->         y = np.array([func(x) for x in X])
  80 |         self.n_evals += len(X)
  81 |         return y.reshape(-1, 1)
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
