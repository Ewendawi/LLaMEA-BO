# Description
**AGE-DTRBO**: Adaptive Gradient-Enhanced Diversity Trust Region Bayesian Optimization. This algorithm synergistically integrates gradient information, adaptive trust region management, and diversity promotion to enhance black-box optimization performance. It builds upon AGETRBO by incorporating a diversity term into the acquisition function, similar to ADTRBO. The trust region is adapted based on success, and gradient information is used to improve the accuracy of the surrogate model. The next points are selected by L-BFGS-B optimization within the trust region.

# Justification
The combination of AGETRBO and ADTRBO aims to leverage the strengths of both algorithms. AGETRBO uses gradient information to improve the surrogate model's accuracy, leading to better exploitation of promising regions. ADTRBO promotes diversity, which helps to avoid premature convergence and explore the search space more effectively. By combining these two approaches, AGE-DTRBO seeks to achieve a better balance between exploration and exploitation, leading to improved performance on a wider range of black-box optimization problems. The use of L-BFGS-B for acquisition function optimization within the trust region ensures efficient and accurate selection of the next evaluation points.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
from sklearn.metrics import pairwise_distances

class AGEDTRBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(10 * dim, self.budget // 5)
        self.trust_region_width = 2.0
        self.success_threshold = 0.1
        self.diversity_weight = 0.1
        self.best_y = np.inf
        self.best_x = None
        self.delta = 1e-3  # Step size for finite difference gradient estimation

    def _sample_points(self, n_points, center=None, width=None):
        if center is None:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            lower_bound = np.maximum(self.bounds[0], center - width / 2)
            upper_bound = np.minimum(self.bounds[1], center + width / 2)
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            scaled_sample = qmc.scale(sample, lower_bound, upper_bound)
            return scaled_sample

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        model.fit(X, y)
        return model

    def _estimate_gradient(self, func, x):
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.delta
            x_minus[i] -= self.delta
            x_plus = np.clip(x_plus, self.bounds[0][i], self.bounds[1][i])
            x_minus = np.clip(x_minus, self.bounds[0][i], self.bounds[1][i])
            gradient[i] = (func(x_plus) - func(x_minus)) / (2 * self.delta)
        return gradient

    def _acquisition_function(self, X):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement
        best = np.min(self.y)
        imp = best - mu
        Z = imp / (sigma + 1e-9)  # Adding a small constant to avoid division by zero
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei = np.clip(ei, 0, 1e10)  # Clip EI to avoid potential NaN issues

        # Diversity term
        if self.X is not None:
            distances = pairwise_distances(X, self.X)
            min_distances = np.min(distances, axis=1).reshape(-1, 1)
            diversity = min_distances
        else:
            diversity = np.zeros_like(ei)

        # Combined acquisition function
        acq = ei + self.diversity_weight * diversity
        return acq

    def _select_next_points(self, func, batch_size):
        X_next = []
        for _ in range(batch_size):
            # Define the objective function for optimization
            def objective(x):
                x = x.reshape(1, -1)
                return -self._acquisition_function(x)[0, 0]

            # Optimization within bounds (trust region)
            lower_bound = np.maximum(self.bounds[0], self.best_x - self.trust_region_width / 2)
            upper_bound = np.minimum(self.bounds[1], self.best_x + self.trust_region_width / 2)
            bounds = [(lower_bound[i], upper_bound[i]) for i in range(self.dim)]

            # Initial guess (randomly sampled within trust region)
            x0 = self._sample_points(1, center=self.best_x, width=self.trust_region_width).flatten()

            # Perform optimization
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

            X_next.append(result.x)

        return np.array(X_next)

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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_x = X_init[np.argmin(y_init)]
        self.best_y = np.min(y_init)

        # Optimization loop
        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            # Fit the model
            self.model = self._fit_model(self.X, self.y)

            # Select next points
            X_next = self._select_next_points(func, batch_size)

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
## Feedback
 The algorithm AGEDTRBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1796 with standard deviation 0.1097.

took 687.50 seconds to run.