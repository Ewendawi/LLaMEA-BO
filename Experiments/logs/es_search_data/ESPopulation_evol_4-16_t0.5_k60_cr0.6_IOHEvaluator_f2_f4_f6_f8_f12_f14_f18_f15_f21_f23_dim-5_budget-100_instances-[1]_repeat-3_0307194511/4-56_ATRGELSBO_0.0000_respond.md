# Description
**Adaptive Trust Region with Gradient-Enhanced Local Search and Dynamic Exploitation-Exploration (ATRGELSBO)**: This algorithm enhances the AdaptiveTrustRegionMExBO by incorporating gradient estimation into the local search, refining the trust region update mechanism, and dynamically adjusting the exploitation-exploration balance. Gradient estimation accelerates local search, while a more robust trust region update prevents premature convergence. We introduce a gradient-based acquisition function and a refined trust region update mechanism using a success ratio and dynamic adjustment of the exploitation-exploration balance.

# Justification
The key improvements are:

1.  **Gradient-Enhanced Local Search:** Incorporating gradient information into the local search helps to accelerate convergence and improve the accuracy of the search. This is done by estimating the gradient using finite differences and using it in conjunction with the GP model to guide the search.
2.  **Refined Trust Region Update:** The trust region size is updated based on the success ratio of local searches within the trust region. This helps to prevent premature convergence and ensures that the algorithm continues to explore the search space effectively.
3.  **Dynamic Exploitation-Exploration Balance:** The balance between exploitation and exploration in the local search is dynamically adjusted based on the success rate of local searches. This helps to ensure that the algorithm is able to effectively balance exploration and exploitation, leading to improved performance.
4. **Efficient Gradient Estimation:** The gradient estimation is done using a small number of evaluations to minimize the computational overhead.

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

class ATRGELSBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim
        self.trust_region_size = 2.0
        self.trust_region_shrink = 0.5
        self.trust_region_expand = 2.0
        self.local_search_exploitation = 0.8  # Initial weight for exploitation in local search
        self.local_search_exploitation_adjust = 0.05
        self.global_search_prob = 0.05
        self.success_threshold = 0.7
        self.trust_region_momentum = 0.5 # Momentum for trust region size update
        self.prev_trust_region_change = 0.0
        self.success_window = 5 # Window size to track successful local searches
        self.gradient_estimation_points = min(self.dim + 1, 10)


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

    def _estimate_gradient(self, func, x, n_points):
        # Estimate gradient using finite differences
        delta = np.random.normal(0, 0.01, size=(n_points, self.dim))
        points = np.clip(x + delta, self.bounds[0], self.bounds[1])
        values = np.array([func(p) for p in points])
        gradient = np.mean((values - func(x)) / delta, axis=0)
        return gradient

    def _acquisition_function(self, X, model, gradient=None):
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        y_best = np.min(self.y)
        imp = y_best - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        if gradient is not None:
            # Incorporate gradient information into the acquisition function
            gradient_component = np.dot(X - self.X[np.argmin(self.y)], gradient)
            ei += 0.1 * gradient_component.reshape(-1,1)

        return ei

    def _local_search(self, func, model, center, n_points=50):
        candidate_points = center + self.trust_region_size * np.random.uniform(-1, 1, size=(n_points, self.dim))
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])

        gradient = self._estimate_gradient(func, center, self.gradient_estimation_points)
        ei = self._acquisition_function(candidate_points, model, gradient)

        mu, _ = model.predict(candidate_points, return_std=True)
        mu = mu.reshape(-1, 1)

        weighted_values = self.local_search_exploitation * mu + (1 - self.local_search_exploitation) * (-ei)
        best_index = np.argmin(weighted_values)
        best_point = candidate_points[best_index]

        return best_point

    def _global_search(self):
        def objective(x):
            model = self._fit_model(self.X, self.y)
            return -self._acquisition_function(x.reshape(1, -1), model)[0, 0]

        x0 = self._sample_points(1)[0]
        bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5})
        return result.x

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
                next_x = self._local_search(func, model, best_x.copy())
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

            # Apply momentum to trust region update
            self.prev_trust_region_change = self.trust_region_momentum * self.prev_trust_region_change + (1 - self.trust_region_momentum) * trust_region_change
            self.trust_region_size *= (1 + self.prev_trust_region_change)
            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0)

            # Adjust exploitation-exploration balance based on success rate
            if len(success_history) > self.success_window:
                success_rate = np.mean(success_history[-self.success_window:])
                if success_rate > self.success_threshold:
                    self.local_search_exploitation = min(1.0, self.local_search_exploitation + self.local_search_exploitation_adjust)
                else:
                    self.local_search_exploitation = max(0.0, self.local_search_exploitation - self.local_search_exploitation_adjust)

        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<ATRGELSBO>", line 132, in __call__
 132->                 next_x = self._local_search(func, model, best_x.copy())
  File "<ATRGELSBO>", line 80, in _local_search
  80->         gradient = self._estimate_gradient(func, center, self.gradient_estimation_points)
  File "<ATRGELSBO>", line 55, in _estimate_gradient
  53 |         points = np.clip(x + delta, self.bounds[0], self.bounds[1])
  54 |         values = np.array([func(p) for p in points])
  55->         gradient = np.mean((values - func(x)) / delta, axis=0)
  56 |         return gradient
  57 | 
ValueError: operands could not be broadcast together with shapes (6,) (6,5) 
