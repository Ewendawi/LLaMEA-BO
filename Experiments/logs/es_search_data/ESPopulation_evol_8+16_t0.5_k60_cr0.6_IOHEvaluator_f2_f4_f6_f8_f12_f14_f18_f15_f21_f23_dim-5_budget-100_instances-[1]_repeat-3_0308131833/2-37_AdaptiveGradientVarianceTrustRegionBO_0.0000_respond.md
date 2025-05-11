# Description
**AdaptiveGradientVarianceTrustRegionBO (AGVTRBO):** This algorithm merges the strengths of AdaptiveTrustRegionBO and TrustRegionGradientBO, incorporating adaptive variance scaling of the EI, gradient-enhanced exploration, and dynamic adjustment of both the trust region and local search radii. A key innovation is the dynamic weighting of the gradient term in the acquisition function, adapting its influence based on the success of recent gradient-based steps. This aims to balance exploration and exploitation more effectively, leveraging gradient information when it proves beneficial and relying more on variance-scaled EI when gradients are less informative. The algorithm also includes a mechanism to prevent premature convergence by monitoring the diversity of sampled points within the trust region and adaptively increasing the trust region size if diversity falls below a threshold.

# Justification
The algorithm builds upon AdaptiveTrustRegionBO (ATrBO) by incorporating gradient information from TrustRegionGradientBO (TRGBO) in a more nuanced way.
1.  **Adaptive Gradient Weighting:** The gradient weight is not fixed but dynamically adjusted based on the correlation between the gradient direction and the actual improvement in function value. This allows the algorithm to learn when the gradient information is reliable and useful, and to reduce its influence when it is not. This addresses a potential weakness of TRGBO, where a fixed gradient weight might lead to suboptimal exploration if the gradient approximation is inaccurate.
2.  **Diversity Monitoring and Trust Region Adjustment:** To prevent premature convergence, the algorithm monitors the diversity of sampled points within the trust region. If the diversity falls below a certain threshold, the trust region size is increased to encourage more exploration. This is particularly useful in multimodal functions where the algorithm might get stuck in a local optimum. The diversity is measured by the average distance between the sampled points and the trust region center.
3.  **Adaptive Variance Scaling:** Similar to ATrBO, the EI is scaled by the predictive variance of the GP to promote exploration in uncertain regions.
4.  **Local Search with Adaptive Radius:** The local search radius is dynamically adjusted based on the success rate of previous local searches, allowing for finer exploitation in promising regions.
5. **Computational Efficiency:** The gradient is approximated using finite differences within the trust region, balancing accuracy and computational cost. The length_scale of the RBF kernel is optimized during GP fitting.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

class AdaptiveGradientVarianceTrustRegionBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.gp = None
        self.trust_region_radius = 2.0
        self.trust_region_shrink = 0.5
        self.trust_region_expand = 1.5
        self.success_threshold = 0.75
        self.failure_threshold = 0.25
        self.trust_region_center = np.zeros(dim)
        self.best_x = None
        self.best_y = np.inf
        self.local_search_radius = 0.1
        self.local_search_success_rate = 0.0
        self.local_search_success_memory = []
        self.local_search_success_window = 5
        self.gradient_weight = 0.1
        self.finite_difference_step = 0.1
        self.gradient_success_memory = []
        self.gradient_success_window = 5
        self.gradient_success_rate = 0.0
        self.diversity_threshold = 0.1 # Threshold for increasing trust region
        self.diversity_memory = []
        self.diversity_window = 5

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        scaled_sample = qmc.scale(sample, -self.trust_region_radius, self.trust_region_radius)
        return scaled_sample + self.trust_region_center

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)
        gp.fit(X, y)
        return gp

    def _approximate_gradients(self, func, x):
        gradients = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.finite_difference_step
            x_minus[i] -= self.finite_difference_step
            x_plus = np.clip(x_plus, self.bounds[0], self.bounds[1])
            x_minus = np.clip(x_minus, self.bounds[0], self.bounds[1])
            gradients[i] = (func(x_plus) - func(x_minus)) / (2 * self.finite_difference_step)
        return gradients

    def _acquisition_function(self, X):
        if self.gp is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.gp.predict(X, return_std=True)
        improvement = self.best_y - mu
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Adaptive scaling of EI based on variance
        ei = ei * sigma

        # Add gradient-based exploration
        gradient_exploration = np.zeros_like(ei)
        for i in range(len(X)):
            gradients = self._approximate_gradients(lambda x: self.gp.predict(x.reshape(1, -1))[0], X[i])
            gradient_exploration[i] = self.gradient_weight * np.linalg.norm(gradients)

        return (ei + gradient_exploration).reshape(-1, 1)

    def _select_next_points(self, batch_size):
        x_tries = self._sample_points(batch_size * 10)
        x_tries = np.clip(x_tries, self.bounds[0], self.bounds[1])
        acq_values = self._acquisition_function(x_tries)
        indices = np.argsort(acq_values.flatten())[::-1][:batch_size]
        return x_tries[indices]

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y
    
    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))
        
        best_index = np.argmin(self.y)
        if self.y[best_index, 0] < self.best_y:
            self.best_x = self.X[best_index]
            self.best_y = self.y[best_index, 0]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_X = np.clip(initial_X, self.bounds[0], self.bounds[1])
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        while self.n_evals < self.budget:
            self.gp = self._fit_model(self.X, self.y)

            batch_size = min(10, self.budget - self.n_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            predicted_improvement = self.best_y - self.gp.predict(self.best_x.reshape(1, -1))[0]
            actual_improvement = self.best_y - min(self.y)[0]

            if abs(predicted_improvement) < 1e-9:
                ratio = 0.0
            else:
                ratio = actual_improvement / predicted_improvement

            if ratio > self.success_threshold:
                self.trust_region_radius *= self.trust_region_expand
            elif ratio < self.failure_threshold:
                self.trust_region_radius *= self.trust_region_shrink
                self.trust_region_radius = max(self.trust_region_radius, 0.1)

            self.trust_region_center = self.best_x

            # Local search around the best solution
            if self.best_x is not None and self.n_evals < self.budget:
                local_X = self._sample_points(1) * self.local_search_radius + self.best_x
                local_X = np.clip(local_X, self.bounds[0], self.bounds[1])
                local_y = self._evaluate_points(func, local_X)
                self._update_eval_points(local_X, local_y)

                # Update local search success rate
                if local_y[0, 0] < self.best_y:
                    self.local_search_success_memory.append(1)
                else:
                    self.local_search_success_memory.append(0)

                if len(self.local_search_success_memory) > self.local_search_success_window:
                    self.local_search_success_memory.pop(0)

                self.local_search_success_rate = np.mean(self.local_search_success_memory)

                # Adjust local search radius
                if self.local_search_success_rate > 0.5:
                    self.local_search_radius *= 0.9  # Reduce radius if successful
                else:
                    self.local_search_radius *= 1.1  # Increase radius if unsuccessful
                self.local_search_radius = np.clip(self.local_search_radius, 0.01, 1.0)

            # Gradient success update
            gradients = self._approximate_gradients(func, self.best_x)
            predicted_direction = -gradients / np.linalg.norm(gradients) if np.linalg.norm(gradients) > 0 else np.zeros(self.dim)
            actual_direction = (self.best_x - self.trust_region_center) / np.linalg.norm(self.best_x - self.trust_region_center) if np.linalg.norm(self.best_x - self.trust_region_center) > 0 else np.zeros(self.dim)
            correlation = np.dot(predicted_direction, actual_direction)
            if correlation > 0:
                self.gradient_success_memory.append(1)
            else:
                self.gradient_success_memory.append(0)

            if len(self.gradient_success_memory) > self.gradient_success_window:
                self.gradient_success_memory.pop(0)

            self.gradient_success_rate = np.mean(self.gradient_success_memory)

            # Adjust gradient weight
            self.gradient_weight = 0.1 * self.gradient_success_rate

            # Diversity check
            if len(self.X) > self.dim + 1:
                distances = [np.linalg.norm(x - self.trust_region_center) for x in self.X]
                avg_distance = np.mean(distances)
                self.diversity_memory.append(avg_distance)

                if len(self.diversity_memory) > self.diversity_window:
                    self.diversity_memory.pop(0)

                avg_diversity = np.mean(self.diversity_memory)

                if avg_diversity < self.diversity_threshold:
                    self.trust_region_radius *= 1.1  # Increase TR radius if diversity is low
                    self.trust_region_radius = min(self.trust_region_radius, 5.0) # Limit TR radius

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<AdaptiveGradientVarianceTrustRegionBO>", line 121, in __call__
 121->             next_y = self._evaluate_points(func, next_X)
  File "<AdaptiveGradientVarianceTrustRegionBO>", line 93, in _evaluate_points
  93->         y = np.array([func(x) for x in X]).reshape(-1, 1)
  File "<AdaptiveGradientVarianceTrustRegionBO>", line 93, in <listcomp>
  91 | 
  92 |     def _evaluate_points(self, func, X):
  93->         y = np.array([func(x) for x in X]).reshape(-1, 1)
  94 |         self.n_evals += len(X)
  95 |         return y
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
