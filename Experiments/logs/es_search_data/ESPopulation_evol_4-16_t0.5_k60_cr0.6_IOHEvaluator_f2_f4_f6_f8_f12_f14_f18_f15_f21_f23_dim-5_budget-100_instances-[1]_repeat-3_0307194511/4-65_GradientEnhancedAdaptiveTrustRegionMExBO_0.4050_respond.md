# Description
**GradientEnhancedAdaptiveTrustRegionMExBO (GEATRMExBO)**: This algorithm combines gradient-enhanced local search with adaptive trust region management, momentum-based trust region updates, and dynamic adjustment of exploitation-exploration balance in local search. It uses nearest neighbors for efficient lengthscale estimation in the Gaussian Process (GP) model. It employs a weighted combination of GP prediction and Expected Improvement (EI) within the trust region for local search. The trust region size is adaptively adjusted based on the success of local search, incorporating momentum to smooth the adjustments. The exploitation-exploration balance in local search is dynamically adjusted based on the success rate of local searches. It enhances gradient estimation by incorporating a regularization term to improve stability. A global search step, optimized using L-BFGS-B, is probabilistically introduced to escape local optima.

# Justification
This algorithm aims to improve upon AdaptiveTrustRegionMExBO and GradientEnhancedAdaptiveTrustRegionOptimisticHybridBO by combining their strengths and addressing their weaknesses.
1.  **Gradient Enhancement**: Gradient information is incorporated into the local search to accelerate convergence, similar to GradientEnhancedAdaptiveTrustRegionOptimisticHybridBO. This helps to guide the search towards promising regions more efficiently.
2.  **Adaptive Trust Region**: An adaptive trust region is used to balance exploration and exploitation. The trust region size is adjusted based on the success of local searches, and momentum is added to smooth these adjustments, as in AdaptiveTrustRegionMExBO.
3.  **Exploitation-Exploration Balancing**: The balance between exploitation and exploration in local search is dynamically adjusted based on the success rate of local searches, as in AdaptiveTrustRegionMExBO.
4.  **Efficient Lengthscale Estimation**: Nearest neighbors are used for efficient lengthscale estimation in the Gaussian Process (GP) model, as in both AdaptiveTrustRegionMExBO and GradientEnhancedAdaptiveTrustRegionOptimisticHybridBO.
5.  **Hybrid Acquisition**: The local search uses a weighted combination of GP prediction and Expected Improvement (EI), similar to AdaptiveTrustRegionMExBO.
6.  **Regularized Gradient Estimation**: A regularization term is added to the gradient estimation to improve stability, especially in high-dimensional spaces or when the GP model is uncertain.
7.  **Global Search**: A global search step, optimized using L-BFGS-B, is probabilistically introduced to escape local optima, as in AdaptiveTrustRegionMExBO.
8. **Computational Efficiency**: The algorithm is designed to be computationally efficient by using nearest neighbors for lengthscale estimation and L-BFGS-B with limited iterations for global search.

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

class GradientEnhancedAdaptiveTrustRegionMExBO:
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
        self.trust_region_momentum = 0.5 # Momentum for trust region size update
        self.prev_trust_region_change = 0.0
        self.successful_local_searches = 0
        self.success_window = 5 # Window size to track successful local searches
        self.delta = 1e-3 # step size for finite differences in gradient estimation
        self.gradient_regularization = 1e-3 # Regularization term for gradient estimation


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
        return ei

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

    def _estimate_gradient(self, model, x):
        # Estimate the gradient of the function at point x using finite differences
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.delta
            x_minus[i] -= self.delta
            
            # Clip to ensure the points are within bounds
            x_plus[i] = np.clip(x_plus[i], self.bounds[0][i], self.bounds[1][i])
            x_minus[i] = np.clip(x_minus[i], self.bounds[0][i], self.bounds[1][i])

            # Use the GP model to predict function values
            y_plus, _ = model.predict(x_plus.reshape(1, -1), return_std=True)
            y_minus, _ = model.predict(x_minus.reshape(1, -1), return_std=True)

            gradient[i] = (y_plus - y_minus) / (2 * self.delta)

        # Regularize the gradient to improve stability
        gradient = gradient / (np.linalg.norm(gradient) + self.gradient_regularization)
        return gradient

    def _local_search(self, model, center, gradient, n_points=50):
        candidate_points = center + self.trust_region_size * np.random.uniform(-1, 1, size=(n_points, self.dim))
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])

        mu, _ = model.predict(candidate_points, return_std=True)
        mu = mu.reshape(-1, 1)
        ei = self._acquisition_function(candidate_points)

        # Incorporate gradient information
        gradient_component = np.sum(gradient * (candidate_points - center), axis=1).reshape(-1, 1)
        weighted_values = self.local_search_exploitation * (mu - 0.1 * gradient_component) + (1 - self.local_search_exploitation) * (-ei)
        best_index = np.argmin(weighted_values)
        best_point = candidate_points[best_index]

        return best_point

    def _global_search(self):
        # Refined global search using optimization of the acquisition function
        def objective(x):
            return -self._acquisition_function(x.reshape(1, -1))[0, 0]

        x0 = self._sample_points(1)[0]  # Start from a random point
        bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5})  # Limited iterations
        return result.x

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

            # Estimate gradient at the best point
            gradient = self._estimate_gradient(model, best_x)

            if np.random.rand() < self.global_search_prob:
                next_x = self._global_search()  # Use refined global search
            else:
                next_x = self._local_search(model, best_x.copy(), gradient)
            next_x = np.clip(next_x, self.bounds[0], self.bounds[1])

            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0]
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            if next_y < best_y:
                best_x = next_x
                best_y = next_y
                trust_region_change = self.trust_region_expand - 1 # Change relative to current size
                success_history.append(1)
            else:
                trust_region_change = self.trust_region_shrink - 1 # Change relative to current size
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
## Feedback
 The algorithm GradientEnhancedAdaptiveTrustRegionMExBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1765 with standard deviation 0.1104.

took 218.91 seconds to run.