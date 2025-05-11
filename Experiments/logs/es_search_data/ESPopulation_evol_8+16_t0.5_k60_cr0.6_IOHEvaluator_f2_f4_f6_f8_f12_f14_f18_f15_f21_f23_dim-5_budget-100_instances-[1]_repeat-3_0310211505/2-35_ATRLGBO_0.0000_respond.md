# Description
**Adaptive Trust Region Landscape-Aware Bayesian Optimization with Gradient Estimation (ATRLGBO)**: This algorithm combines the adaptive trust region approach of ATRBO, the landscape analysis of ALTRBO, and gradient estimation to enhance the search process. It adaptively adjusts the trust region based on the success of previous iterations, uses landscape analysis to adjust the exploration-exploitation trade-off by modifying the temperature parameter in the Expected Improvement (EI) acquisition function, and incorporates gradient information to guide the search within the trust region. Gradient information is estimated using finite differences and is incorporated into the acquisition function to improve the efficiency and robustness of the search.

# Justification
This algorithm builds upon ATRBO and ALTRBO by adding gradient estimation.
1.  **Adaptive Trust Region**: The adaptive trust region management from ATRBO ensures efficient exploration and exploitation by dynamically adjusting the search space based on the success of previous iterations.
2.  **Landscape Analysis**: The landscape analysis from ALTRBO helps to adjust the exploration-exploitation trade-off by modifying the temperature parameter in the EI acquisition function based on the smoothness of the landscape.
3.  **Gradient Estimation**: The gradient estimation enhances the search process by providing information about the local landscape. This information is incorporated into the acquisition function to guide the search towards promising regions. Finite differences are used for gradient estimation to avoid the need for analytical gradients.
4. **Computational Efficiency**: The use of finite differences for gradient estimation is computationally efficient and avoids the need for analytical gradients. The adaptive trust region and landscape analysis also help to focus the search on promising regions, reducing the number of function evaluations required.
5. **Sobol Sampling**: Sobol sequence is used for initial sampling to ensure better space-filling properties, especially for low-dimensional problems.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances

class ATRLGBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(10 * dim, self.budget // 5)
        self.trust_region_width = 2.0  # Initial trust region width
        self.success_threshold = 0.1  # Threshold for increasing trust region
        self.best_y = np.inf  # Initialize best_y with a large value
        self.best_x = None
        self.temperature = 1.0  # Initial temperature for exploration
        self.landscape_correlation = 0.0  # Initial landscape correlation
        self.smoothness_threshold = 0.5  # Threshold for considering the landscape smooth
        self.gradient_weight = 0.1  # Weight for gradient in acquisition function
        self.delta = 0.01  # Step size for finite differences

    def _sample_points(self, n_points, center=None, width=None, use_sobol=False):
        if center is None:
            if use_sobol:
                sampler = qmc.Sobol(d=self.dim, scramble=True)
            else:
                sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            # Sample within the trust region
            lower_bound = np.maximum(self.bounds[0], center - width / 2)
            upper_bound = np.minimum(self.bounds[1], center + width / 2)
            if use_sobol:
                sampler = qmc.Sobol(d=self.dim, scramble=True)
            else:
                sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            scaled_sample = qmc.scale(sample, lower_bound, upper_bound)
            return scaled_sample

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        model.fit(X, y)
        return model

    def _analyze_landscape(self):
        if self.X is None or self.y is None or len(self.X) < 2:
            return 0.0

        distances = pairwise_distances(self.X)
        value_differences = np.abs(self.y - self.y.T)

        # Flatten the matrices and remove the diagonal elements
        distances = distances.flatten()
        value_differences = value_differences.flatten()
        indices = np.arange(len(distances))
        distances = distances[indices % (len(self.X) + 1) != 0]
        value_differences = value_differences[indices % (len(self.X) + 1) != 0]

        # Calculate Pearson correlation coefficient
        correlation, _ = pearsonr(distances, value_differences)
        return correlation if not np.isnan(correlation) else 0.0

    def _estimate_gradient(self, func, x):
        # Estimate gradient using finite differences
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.delta
            x_minus[i] -= self.delta
            gradient[i] = (func(x_plus) - func(x_minus)) / (2 * self.delta)
        return gradient

    def _acquisition_function(self, X, func):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement with temperature
        imp = self.best_y - mu
        Z = imp / (self.temperature * sigma + 1e-9)  # Adding a small constant to avoid division by zero
        ei = imp * norm.cdf(Z) + self.temperature * sigma * norm.pdf(Z)
        ei = np.clip(ei, 0, 1e10)  # Clip EI to avoid potential NaN issues
        ei[sigma <= 1e-6] = 0.0  # avoid division by zero

        # Incorporate gradient information
        gradient_values = np.array([self._estimate_gradient(func, x) for x in X])
        # Normalize gradients
        gradient_norm = np.linalg.norm(gradient_values, axis=1, keepdims=True)
        gradient_norm[gradient_norm == 0] = 1  # Avoid division by zero
        normalized_gradients = gradient_values / gradient_norm

        # Calculate cosine similarity between gradients and vector to best_x
        to_best_x = self.best_x - X
        to_best_x_norm = np.linalg.norm(to_best_x, axis=1, keepdims=True)
        to_best_x_norm[to_best_x_norm == 0] = 1  # Avoid division by zero
        normalized_to_best_x = to_best_x / to_best_x_norm

        cosine_similarity = np.sum(normalized_gradients * normalized_to_best_x, axis=1, keepdims=True)

        # Combine EI and gradient information
        acquisition_values = ei + self.gradient_weight * cosine_similarity

        return acquisition_values

    def _select_next_points(self, batch_size, func):
        n_candidates = max(1000, batch_size * 100)
        X_cand = self._sample_points(n_candidates, center=self.best_x, width=self.trust_region_width)

        acq_values = self._acquisition_function(X_cand, func)

        top_indices = np.argsort(acq_values.flatten())[::-1][:batch_size]
        return X_cand[top_indices]

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.concatenate((self.X, new_X), axis=0)
            self.y = np.concatenate((self.y, new_y), axis=0)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial sampling
        X_init = self._sample_points(self.n_init, use_sobol=True)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_x = X_init[np.argmin(y_init)]
        self.best_y = np.min(y_init)

        # Optimization loop
        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            # Fit the model
            self.model = self._fit_model(self.X, self.y)

            # Analyze the landscape
            self.landscape_correlation = self._analyze_landscape()

            # Adjust temperature based on landscape correlation
            if self.landscape_correlation > self.smoothness_threshold:
                self.temperature = max(0.1, self.temperature * 0.9)  # Reduce temperature for smoother landscapes
            else:
                self.temperature = min(2.0, self.temperature * 1.1)  # Increase temperature for rugged landscapes

            # Select next points
            X_next = self._select_next_points(batch_size, func)

            # Evaluate points
            y_next = self._evaluate_points(func, X_next)

            # Update evaluated points
            self._update_eval_points(X_next, y_next)

            # Update best solution
            new_best_y = np.min(self.y)
            new_best_x = self.X[np.argmin(self.y)]

            # Adjust trust region width
            if (self.best_y - new_best_y) / self.best_y > self.success_threshold:
                self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)  # Increase if successful
            else:
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)  # Decrease if not successful

            self.best_y = new_best_y
            self.best_x = new_best_x

        # Return best solution
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<ATRLGBO>", line 165, in __call__
 165->             X_next = self._select_next_points(batch_size, func)
  File "<ATRLGBO>", line 123, in _select_next_points
 123->         acq_values = self._acquisition_function(X_cand, func)
  File "<ATRLGBO>", line 100, in _acquisition_function
 100->         gradient_values = np.array([self._estimate_gradient(func, x) for x in X])
  File "<ATRLGBO>", line 100, in <listcomp>
 100->         gradient_values = np.array([self._estimate_gradient(func, x) for x in X])
  File "<ATRLGBO>", line 81, in _estimate_gradient
  79 |             x_plus[i] += self.delta
  80 |             x_minus[i] -= self.delta
  81->             gradient[i] = (func(x_plus) - func(x_minus)) / (2 * self.delta)
  82 |         return gradient
  83 | 
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
