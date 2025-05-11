# Description
Adaptive Gradient-Enhanced Temperature-Adjusted Trust Region Bayesian Optimization with Model Averaging (AGETRMATBO) is a novel algorithm that combines the strengths of AGETRBO and ATSMABO. It incorporates gradient information, estimated using finite differences, into the Gaussian Process Regression (GPR) model, adaptively adjusts the trust region based on success, employs an ensemble of GPR models with different kernels (RBF and Matern) as surrogate models, and uses an adaptive temperature schedule to balance exploration and exploitation. The acquisition function is Expected Improvement (EI) calculated using the averaged predictions from the GPR ensemble. The next points are selected by L-BFGS-B optimization within the trust region.

# Justification
This algorithm aims to leverage the benefits of both gradient-enhanced modeling and surrogate model averaging within an adaptive trust region framework.
- **Gradient Enhancement:** Incorporating gradient information can improve the accuracy of the GPR model, especially in high-dimensional spaces, leading to more efficient exploration.
- **Surrogate Model Averaging:** Using an ensemble of GPR models with different kernels (RBF and Matern) can improve the robustness of the algorithm by capturing different aspects of the objective function's landscape. Averaging the predictions of these models reduces the risk of overfitting to a single kernel.
- **Adaptive Temperature:** Dynamically adjusting the temperature parameter in the EI acquisition function allows the algorithm to adapt its exploration-exploitation trade-off based on the optimization progress. This helps to avoid premature convergence and efficiently explore the search space.
- **Adaptive Trust Region:** By dynamically adjusting the trust region, the algorithm can focus the search on promising regions while maintaining exploration capabilities.
- **L-BFGS-B Optimization:** Using L-BFGS-B for acquisition function optimization within the trust region allows for efficient exploitation of the local landscape.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize


class AGETRMATBO:
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
        self.delta = 1e-3  # Step size for finite difference gradient estimation
        self.temperature = 1.0  # Initial temperature for exploration
        self.temperature_decay = 0.95  # Initial decay rate
        self.temperature_decay_min = 0.8  # Minimum decay rate
        self.temperature_decay_max = 0.99  # Maximum decay rate

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
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        kernel_rbf = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        kernel_matern = ConstantKernel(1.0) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)

        gp_rbf = GaussianProcessRegressor(kernel=kernel_rbf, n_restarts_optimizer=5)
        gp_matern = GaussianProcessRegressor(kernel=kernel_matern, n_restarts_optimizer=5)

        gp_rbf.fit(X_train, y_train)
        gp_matern.fit(X_train, y_train)

        y_pred_rbf, _ = gp_rbf.predict(X_val, return_std=True)
        y_pred_matern, _ = gp_matern.predict(X_val, return_std=True)

        error_rbf = np.mean((y_val - y_pred_rbf.reshape(-1, 1)) ** 2)
        error_matern = np.mean((y_val - y_pred_matern.reshape(-1, 1)) ** 2)

        total_error = error_rbf + error_matern
        weight_rbf = 1.0 - (error_rbf / total_error) if total_error > 0 else 0.5
        weight_matern = 1.0 - (error_matern / total_error) if total_error > 0 else 0.5

        return gp_rbf, gp_matern, weight_rbf, weight_matern

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

        mu_rbf, sigma_rbf = self.model_rbf.predict(X, return_std=True)
        mu_matern, sigma_matern = self.model_matern.predict(X, return_std=True)

        mu_rbf = mu_rbf.reshape(-1, 1)
        sigma_rbf = sigma_rbf.reshape(-1, 1)
        mu_matern = mu_matern.reshape(-1, 1)
        sigma_matern = sigma_matern.reshape(-1, 1)

        mu = self.weight_rbf * mu_rbf + self.weight_matern * mu_matern
        sigma = self.weight_rbf * sigma_rbf + self.weight_matern * sigma_matern

        best = np.min(self.y)
        imp = best - mu
        Z = imp / (self.temperature * sigma + 1e-9)
        ei = imp * norm.cdf(Z) + self.temperature * sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        return ei

    def _select_next_points(self, func, batch_size):
        X_next = []
        for _ in range(batch_size):
            def objective(x):
                x = x.reshape(1, -1)
                return -self._acquisition_function(x)[0, 0]

            lower_bound = np.maximum(self.bounds[0], self.best_x - self.trust_region_width / 2)
            upper_bound = np.minimum(self.bounds[1], self.best_x + self.trust_region_width / 2)
            bounds = [(lower_bound[i], upper_bound[i]) for i in range(self.dim)]

            x0 = self._sample_points(1, center=self.best_x, width=self.trust_region_width).flatten()

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
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_x = X_init[np.argmin(y_init)]
        self.best_y = np.min(y_init)

        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            self.model_rbf, self.model_matern, self.weight_rbf, self.weight_matern = self._fit_model(self.X, self.y)

            X_next = self._select_next_points(func, batch_size)

            y_next = self._evaluate_points(func, X_next)

            self._update_eval_points(X_next, y_next)

            new_best_y = np.min(self.y)
            new_best_x = self.X[np.argmin(self.y)]

            if (self.best_y - new_best_y) / self.best_y > self.success_threshold:
                self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)
            else:
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)

            if new_best_y < self.best_y:
                self.temperature_decay = min(self.temperature_decay + 0.02, self.temperature_decay_max)
            else:
                self.temperature_decay = max(self.temperature_decay - 0.02, self.temperature_decay_min)

            self.temperature *= self.temperature_decay
            self.best_y = new_best_y
            self.best_x = new_best_x

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Feedback
 The algorithm AGETRMATBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1802 with standard deviation 0.1124.

took 1088.01 seconds to run.