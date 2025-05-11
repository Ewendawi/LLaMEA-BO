# Description
Adaptive Trust Region with Dynamic Temperature and Model Averaging Bayesian Optimization (ATRDTMABO) is a novel algorithm that combines the adaptive trust region management of ATRBO with the dynamic temperature adjustment and surrogate model averaging of ATSMABO. It leverages the strengths of both algorithms to balance exploration and exploitation more effectively. The algorithm uses an ensemble of Gaussian Process Regression (GPR) models with RBF and Matern kernels, adaptively adjusts the trust region based on success, and dynamically adjusts the temperature in the Expected Improvement (EI) acquisition function based on landscape characteristics. This combination allows the algorithm to efficiently explore rugged landscapes and exploit smooth regions, while also mitigating the risk of premature convergence.

# Justification
The algorithm combines ATRBO's trust region approach with ATSMABO's model averaging and adaptive temperature. The trust region helps to focus the search, while model averaging improves the robustness of the surrogate model. The adaptive temperature schedule allows the algorithm to dynamically adjust the exploration-exploitation trade-off based on the optimization progress.

*   **Adaptive Trust Region:** ATRBO's adaptive trust region is included to focus sampling in promising regions and adjust the region size based on the success of previous iterations.
*   **Surrogate Model Averaging:** ATSMABO's surrogate model averaging with RBF and Matern kernels, weighted by validation error, is used to improve the accuracy and robustness of the surrogate model.
*   **Dynamic Temperature:** ATSMABO's adaptive temperature schedule is incorporated to dynamically adjust the exploration-exploitation trade-off based on the landscape characteristics.
*   **Batch Size:** The batch size is dynamically adjusted based on the dimension of the problem to balance exploration and exploitation.
*   **Sobol Sequence:** The initial sampling is improved by using Sobol sequence, which is known to have better space-filling properties than Latin Hypercube Sampling, especially for low-dimensional problems.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from sklearn.model_selection import train_test_split


class ATRDTMABO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(10 * dim, self.budget // 5)
        self.temperature = 1.0  # Initial temperature for exploration
        self.temperature_decay = 0.95  # Initial decay rate
        self.temperature_decay_min = 0.8  # Minimum decay rate
        self.temperature_decay_max = 0.99  # Maximum decay rate
        self.best_y = np.inf
        self.last_best_y = np.inf
        self.trust_region_width = 2.0
        self.success_threshold = 0.1

    def _sample_points(self, n_points, center=None, width=None):
        if center is None:
            sampler = qmc.Sobol(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            lower_bound = np.maximum(self.bounds[0], center - width / 2)
            upper_bound = np.minimum(self.bounds[1], center + width / 2)
            sampler = qmc.Sobol(d=self.dim)
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

    def _select_next_points(self, batch_size):
        n_candidates = max(1000, batch_size * 100)
        X_cand = self._sample_points(n_candidates, center=self.best_x, width=self.trust_region_width)
        acq_values = self._acquisition_function(X_cand)
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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_y = np.min(y_init)
        self.last_best_y = self.best_y
        self.best_x = X_init[np.argmin(y_init)]

        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            self.model_rbf, self.model_matern, self.weight_rbf, self.weight_matern = self._fit_model(self.X, self.y)

            X_next = self._select_next_points(batch_size)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

            current_best_y = np.min(self.y)
            new_best_x = self.X[np.argmin(self.y)]

            if current_best_y < self.best_y:
                self.temperature_decay = min(self.temperature_decay + 0.02, self.temperature_decay_max)
                if (self.best_y - current_best_y) / self.best_y > self.success_threshold:
                    self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)
                else:
                    self.trust_region_width = self.trust_region_width
            else:
                self.temperature_decay = max(self.temperature_decay - 0.02, self.temperature_decay_min)
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)

            self.temperature *= self.temperature_decay
            self.best_y = current_best_y
            self.best_x = new_best_x

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Feedback
 The algorithm ATRDTMABO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1801 with standard deviation 0.1086.

took 1305.88 seconds to run.