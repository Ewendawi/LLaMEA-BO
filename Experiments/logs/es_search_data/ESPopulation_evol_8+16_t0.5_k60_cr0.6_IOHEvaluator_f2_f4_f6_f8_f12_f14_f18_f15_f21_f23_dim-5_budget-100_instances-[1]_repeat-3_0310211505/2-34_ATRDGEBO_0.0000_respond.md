# Description
**Adaptive Trust Region with Dynamic Ensemble and Gradient-Enhanced Bayesian Optimization (ATRDGEBO)**: This algorithm combines the strengths of ATRBO, AGETRBO and ATRSMABO. It uses an adaptive trust region to focus the search, employs an ensemble of Gaussian Process Regression (GPR) models with different kernels (RBF and Matern) for improved surrogate modeling, and incorporates gradient information to enhance the accuracy of the GPR predictions. The trust region width is adjusted based on the success rate and the uncertainty of the GPR model. The acquisition function is Expected Improvement (EI). The algorithm adaptively weights the models in the ensemble based on their performance on a validation set of past evaluations. Gradient information is estimated using finite differences and incorporated into the GPR model.

# Justification
This algorithm builds upon ATRBO and ATRSMABO by adding gradient information, similar to AGETRBO, and dynamically weighting the surrogate models based on their performance.
1.  **Adaptive Trust Region:** The adaptive trust region strategy from ATRBO is used to focus the search in promising regions of the search space.
2.  **Surrogate Model Averaging:** An ensemble of GPR models with RBF and Matern kernels, similar to ATRSMABO, is used to improve the accuracy and robustness of the surrogate model. The models are dynamically weighted based on their performance on a validation set.
3.  **Gradient Enhancement:** Gradient information, estimated using finite differences, is incorporated into the GPR model, similar to AGETRBO, to improve the accuracy of the predictions.
4.  **Dynamic Ensemble Weighting:** The weights of the GPR models in the ensemble are dynamically adjusted based on their performance on a validation set of past evaluations. This allows the algorithm to adapt to the characteristics of the optimization problem and prioritize the models that are performing best.
5.  **Computational Efficiency:** The algorithm uses finite differences to estimate gradients, which is computationally efficient. The ensemble size is limited to two models to reduce the computational cost of surrogate model fitting and prediction.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from sklearn.model_selection import train_test_split

class ATRDGEBO:
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
        self.best_y = np.inf
        self.gradient_estimation_radius = 0.1
        self.model_weights = [0.5, 0.5]  # Initial weights for RBF and Matern kernels

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

    def _estimate_gradient(self, func, x, radius):
        # Estimate gradient using finite differences
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += radius
            x_minus[i] -= radius
            x_plus = np.clip(x_plus, self.bounds[0][i], self.bounds[1][i])
            x_minus = np.clip(x_minus, self.bounds[0][i], self.bounds[1][i])
            gradient[i] = (func(x_plus) - func(x_minus)) / (2 * radius)
        return gradient

    def _fit_model(self, X, y, func):
        # Gradient-enhanced GPR model fitting with dynamic ensemble weighting
        kernel_rbf = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        kernel_matern = ConstantKernel(1.0) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)

        gp_rbf = GaussianProcessRegressor(kernel=kernel_rbf, n_restarts_optimizer=5)
        gp_matern = GaussianProcessRegressor(kernel=kernel_matern, n_restarts_optimizer=5)

        # Use gradient information
        X_grad = []
        y_grad = []
        for i in range(len(X)):
            gradient = self._estimate_gradient(func, X[i], self.gradient_estimation_radius)
            X_grad.append(X[i])
            y_grad.append(gradient)
        X_grad = np.array(X_grad)
        y_grad = np.array(y_grad)

        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        gp_rbf.fit(X_train, y_train)
        gp_matern.fit(X_train, y_train)

        # Dynamic ensemble weighting based on validation performance
        rbf_val_error = np.mean((gp_rbf.predict(X_val) - y_val)**2)
        matern_val_error = np.mean((gp_matern.predict(X_val) - y_val)**2)

        total_error = rbf_val_error + matern_val_error
        if total_error > 0:
            self.model_weights[0] = 1 - (rbf_val_error / total_error)
            self.model_weights[1] = 1 - (matern_val_error / total_error)
        else:
            self.model_weights = [0.5, 0.5]  # Revert to equal weights if errors are zero

        return gp_rbf, gp_matern

    def _acquisition_function(self, X):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu_rbf, sigma_rbf = self.model_rbf.predict(X, return_std=True)
        mu_matern, sigma_matern = self.model_matern.predict(X, return_std=True)

        mu_rbf = mu_rbf.reshape(-1, 1)
        sigma_rbf = sigma_rbf.reshape(-1, 1)
        mu_matern = mu_matern.reshape(-1, 1)
        sigma_matern = sigma_matern.reshape(-1, 1)

        # Weighted averaging of predictions
        mu = self.model_weights[0] * mu_rbf + self.model_weights[1] * mu_matern
        sigma = self.model_weights[0] * sigma_rbf + self.model_weights[1] * sigma_matern

        imp = self.best_y - mu
        Z = imp / (sigma + 1e-9)
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
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
        self.best_x = X_init[np.argmin(y_init)][0]
        self.best_y = np.min(y_init)

        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            self.model_rbf, self.model_matern = self._fit_model(self.X, self.y, func)
            X_next = self._select_next_points(batch_size)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

            new_best_y = np.min(self.y)
            new_best_x = self.X[np.argmin(self.y)]

            if (self.best_y - new_best_y) / self.best_y > self.success_threshold:
                self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)
            else:
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)

            self.best_y = new_best_y
            self.best_x = new_best_x

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<ATRDGEBO>", line 139, in __call__
 139->             self.model_rbf, self.model_matern = self._fit_model(self.X, self.y, func)
  File "<ATRDGEBO>", line 62, in _fit_model
  62->             gradient = self._estimate_gradient(func, X[i], self.gradient_estimation_radius)
  File "<ATRDGEBO>", line 47, in _estimate_gradient
  45 |             x_plus = np.clip(x_plus, self.bounds[0][i], self.bounds[1][i])
  46 |             x_minus = np.clip(x_minus, self.bounds[0][i], self.bounds[1][i])
  47->             gradient[i] = (func(x_plus) - func(x_minus)) / (2 * radius)
  48 |         return gradient
  49 | 
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
