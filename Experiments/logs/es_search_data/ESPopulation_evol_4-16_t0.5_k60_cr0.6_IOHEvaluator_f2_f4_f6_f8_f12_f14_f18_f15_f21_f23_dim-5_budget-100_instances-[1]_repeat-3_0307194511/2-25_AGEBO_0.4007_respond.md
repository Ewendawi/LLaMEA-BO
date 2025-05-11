# Description
**Adaptive Gradient-Enhanced Trust Region Bayesian Optimization (AGEBO)**: This algorithm combines the strengths of AdaptiveTrustRegionBO and TREGEBO by incorporating adaptive trust region management, efficient gradient estimation, and a dynamic balance between local and global search. It adaptively adjusts the trust region size based on the agreement between predicted and actual improvements, uses gradient information to refine the local search, and dynamically adjusts the probability of global search based on the success of the local search. The gradient is estimated using the GP surrogate model, and the local search leverages both the GP predictions and the gradient information. To enhance exploration, a dynamically adjusted global search probability is introduced.

# Justification
This algorithm builds upon the strengths of TREGEBO and AdaptiveTrustRegionBO while addressing their limitations.

*   **Adaptive Trust Region:** The trust region size is adaptively adjusted based on the ratio of actual to predicted improvement, providing a more robust and efficient exploration-exploitation balance than fixed expansion/shrinkage factors.
*   **Efficient Gradient Estimation:** Gradient information is incorporated into the local search to guide the search direction and improve convergence. The gradient is estimated using the GP model to reduce the number of function evaluations.
*   **Dynamic Global Search Probability:** The probability of performing a global search step is dynamically adjusted based on the recent success of the local search. If the local search is consistently finding improvements, the global search probability is decreased to focus on exploitation. If the local search stagnates, the global search probability is increased to encourage exploration.
*   **Combination of GP Prediction and Gradient Information in Local Search**: The local search combines the GP model's predictions with gradient information to find promising points within the trust region. This provides a more informed search than using either method alone.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

class AGEBO:
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
        self.local_search_exploitation = 0.8
        self.global_search_prob = 0.05
        self.delta = 1e-3
        self.success_history = []  # Keep track of recent local search successes
        self.success_window = 5  # Number of recent iterations to consider for success rate

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
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

        mu, _ = model.predict(candidate_points, return_std=True)
        mu = mu.reshape(-1, 1)
        ei = self._acquisition_function(candidate_points)
        weighted_values = self.local_search_exploitation * mu + (1 - self.local_search_exploitation) * (-ei)

        # Incorporate gradient information
        gradient_component = -0.1 * np.sum(gradient * (candidate_points - center), axis=1).reshape(-1, 1)
        weighted_values += gradient_component

        best_index = np.argmin(weighted_values)
        best_point = candidate_points[best_index]
        return best_point

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_index = np.argmin(self.y)
        best_x = self.X[best_index]
        best_y = self.y[best_index][0]

        while self.n_evals < self.budget:
            model = self._fit_model(self.X, self.y)
            gradient = self._estimate_gradient(model, best_x)

            # Perform global search with dynamically adjusted probability
            if np.random.rand() < self.global_search_prob:
                next_x = self._select_next_points(1)[0]
            else:
                next_x = self._local_search(model, best_x.copy(), gradient)

            next_x = np.clip(next_x, self.bounds[0], self.bounds[1])
            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0]
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            improvement = best_y - next_y
            predicted_y, _ = model.predict(next_x.reshape(1, -1), return_std=True)
            predicted_improvement = best_y - predicted_y[0]

            if predicted_improvement != 0:
                ratio = improvement / predicted_improvement
                if ratio > 0.5:
                    self.trust_region_size *= self.trust_region_expand
                    self.success_history.append(1)  # Mark as success
                else:
                    self.trust_region_size *= self.trust_region_shrink
                    self.success_history.append(0)  # Mark as failure
            else:
                self.trust_region_size *= self.trust_region_shrink
                self.success_history.append(0)  # Mark as failure

            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0)

            # Adjust global search probability based on recent success rate
            if len(self.success_history) > self.success_window:
                self.success_history = self.success_history[-self.success_window:]
                success_rate = np.mean(self.success_history)
                # Dynamically adjust global search probability
                self.global_search_prob = np.clip(1 - success_rate, 0.05, 0.5)  # Increase exploration if local search fails

            if next_y < best_y:
                best_x = next_x
                best_y = next_y

        return best_y, best_x
```
## Feedback
 The algorithm AGEBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1704 with standard deviation 0.1061.

took 240.15 seconds to run.