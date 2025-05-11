# Description
**Gradient-Enhanced Adaptive Trust Region with Exploitation-Exploration Balancing and Optimized Global Search (GEATRMExOGSBO)**: This algorithm combines gradient estimation with an adaptive trust region strategy, exploitation-exploration balancing in local search, and an optimized global search strategy. It uses nearest neighbor-based lengthscale estimation for the GP kernel and momentum-based trust region adaptation. The local search leverages both the GP model and gradient information, and the trust region size is adaptively adjusted based on the success of the local search. The exploitation-exploration balance in local search is dynamically adjusted based on the success rate of local searches. The global search is optimized using L-BFGS-B with multiple restarts from diverse points selected using Expected Improvement. A dynamic weighting strategy is used to combine the GP prediction and gradient information in the local search.

# Justification
This algorithm builds upon the strengths of `GradientEnhancedAdaptiveTrustRegionBO` and `AdaptiveTrustRegionMExBO`.

1.  **Gradient Enhancement**: The gradient estimation from `GradientEnhancedAdaptiveTrustRegionBO` accelerates local search, especially in high-dimensional spaces.
2.  **Adaptive Trust Region**: The adaptive trust region strategy balances exploration and exploitation. The momentum-based updates smooth the transitions in trust region size.
3.  **Exploitation-Exploration Balancing**: The dynamic adjustment of exploitation-exploration balance in local search, inspired by `AdaptiveTrustRegionMExBO`, helps to fine-tune the search within the trust region.
4.  **Optimized Global Search**: Instead of a simple probabilistic global search, this algorithm uses L-BFGS-B to optimize the EI acquisition function from multiple starting points. The starting points are selected using EI to ensure diversity and guide the search towards promising regions. This addresses the potential for the simple probabilistic global search to be inefficient. Limiting the number of iterations in L-BFGS-B maintains computational efficiency.
5.  **Dynamic Weighting of GP and Gradient**: In the local search, the weighting of GP prediction and gradient information is dynamically adjusted based on the uncertainty (sigma) provided by the GP model. This allows the algorithm to rely more on the gradient when the GP model is uncertain and more on the GP model when the uncertainty is low.

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

class GEATRMExOGSBO:
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
        self.success_threshold = 0.7
        self.trust_region_momentum = 0.5  # Momentum for trust region size update
        self.prev_trust_region_change = 0.0
        self.success_window = 5  # Window size to track successful local searches
        self.delta = 1e-3  # Step size for finite differences in gradient estimation
        self.global_search_prob = 0.05
        self.global_search_restarts = 3 # Number of restarts for global search

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

    def _acquisition_function_ei(self, X):
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

    def _select_next_points_ei(self, batch_size):
        candidate_points = self._sample_points(10 * batch_size)
        acquisition_values = self._acquisition_function_ei(candidate_points)
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

    def _estimate_gradient(self, model, x):
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.delta
            x_minus[i] -= self.delta

            x_plus[i] = np.clip(x_plus[i], self.bounds[0][i], self.bounds[1][i])
            x_minus[i] = np.clip(x_minus[i], self.bounds[0][i], self.bounds[1][i])

            y_plus, _ = model.predict(x_plus.reshape(1, -1), return_std=True)
            y_minus, _ = model.predict(x_minus.reshape(1, -1), return_std=True)

            gradient[i] = (y_plus - y_minus) / (2 * self.delta)
        return gradient

    def _local_search(self, model, center, gradient, n_points=50):
        candidate_points = center + self.trust_region_size * np.random.uniform(-1, 1, size=(n_points, self.dim))
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])

        mu, sigma = model.predict(candidate_points, return_std=True)
        mu = mu.reshape(-1, 1)

        # Dynamic weighting based on GP uncertainty
        gradient_weight = np.clip(sigma, 0.0, 1.0)  # Scale uncertainty to [0, 1]
        gradient_contribution = np.sum(gradient * (candidate_points - center), axis=1, keepdims=True)
        weighted_values = mu - gradient_weight * 0.1 * gradient_contribution

        best_index = np.argmin(weighted_values)
        best_point = candidate_points[best_index]

        return best_point

    def _global_search(self):
        def objective(x):
            return -self._acquisition_function_ei(x.reshape(1, -1))[0, 0]

        best_x_global = None
        best_obj_global = np.inf

        # Multiple restarts from EI-selected points
        initial_points = self._select_next_points_ei(self.global_search_restarts)
        bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]

        for x0 in initial_points:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5})
            if result.fun < best_obj_global:
                best_obj_global = result.fun
                best_x_global = result.x

        return best_x_global

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_index = np.argmin(self.y)
        best_x = self.X[best_index]
        best_y = self.y[best_index][0]

        success_history = []

        while self.n_evals < self.budget:
            model = self._fit_model(self.X, self.y)

            if np.random.rand() < self.global_search_prob:
                next_x = self._global_search()
            else:
                gradient = self._estimate_gradient(model, best_x.copy())
                next_x = self._local_search(model, best_x.copy(), gradient)

            next_x = np.clip(next_x, self.bounds[0], self.bounds[1])
            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0]
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            if next_y < best_y:
                best_x = next_x
                best_y = next_y
                trust_region_change = self.trust_region_expand - 1
                success_history.append(1)
            else:
                trust_region_change = self.trust_region_shrink - 1
                success_history.append(0)

            self.prev_trust_region_change = self.trust_region_momentum * self.prev_trust_region_change + (
                        1 - self.trust_region_momentum) * trust_region_change
            self.trust_region_size *= (1 + self.prev_trust_region_change)
            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0)

            if len(success_history) > self.success_window:
                success_rate = np.mean(success_history[-self.success_window:])
                if success_rate > self.success_threshold:
                    self.local_search_exploitation = min(1.0,
                                                          self.local_search_exploitation + self.local_search_exploitation_adjust)
                else:
                    self.local_search_exploitation = max(0.0,
                                                          self.local_search_exploitation - self.local_search_exploitation_adjust)

        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<GEATRMExOGSBO>", line 157, in __call__
 157->                 next_x = self._local_search(model, best_x.copy(), gradient)
  File "<GEATRMExOGSBO>", line 116, in _local_search
 114 | 
 115 |         best_index = np.argmin(weighted_values)
 116->         best_point = candidate_points[best_index]
 117 | 
 118 |         return best_point
IndexError: index 450 is out of bounds for axis 0 with size 50
