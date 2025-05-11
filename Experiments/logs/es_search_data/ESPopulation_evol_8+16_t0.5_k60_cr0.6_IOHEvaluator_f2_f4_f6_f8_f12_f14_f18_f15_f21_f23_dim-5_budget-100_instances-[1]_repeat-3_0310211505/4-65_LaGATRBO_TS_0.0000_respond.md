# Description
**LaGATRBO_TS: Landscape-Aware Gradient-Enhanced Adaptive Trust Region Bayesian Optimization with Thompson Sampling for Kernel Selection.** This algorithm combines landscape analysis, gradient estimation, adaptive trust region, and Thompson Sampling for kernel selection. It builds upon ALGTRBOUE and ATRSMABO_TS, incorporating landscape analysis to adjust exploration, gradient estimation to improve model accuracy, adaptive trust region for efficient search space exploration, and Thompson Sampling to dynamically select between RBF and Matern kernels. The landscape analysis adjusts a temperature parameter in the Expected Improvement (EI) acquisition function. Gradient estimation is performed using finite differences and incorporated into the GPR model. The trust region is adapted based on the success rate and model uncertainty. Thompson Sampling is used to sample from the posterior predictive distribution of each kernel, and the acquisition function is calculated based on these samples, allowing the algorithm to adaptively favor the kernel that is performing better.

# Justification
This algorithm synergistically combines the strengths of ALGTRBOUE and ATRSMABO_TS while addressing their limitations.

*   **Landscape Analysis:** The landscape analysis from ALGTRBOUE helps to adapt the exploration-exploitation trade-off by adjusting the temperature parameter in the EI acquisition function. This allows the algorithm to explore more in rugged landscapes and exploit more in smoother landscapes.
*   **Gradient Estimation:** Gradient estimation, inspired by AGETRBO, is incorporated to improve the accuracy of the Gaussian Process Regression (GPR) model, especially in regions where the landscape is relatively smooth. Finite differences are used to estimate gradients.
*   **Adaptive Trust Region:** The adaptive trust region mechanism, similar to ATRBOUE, dynamically adjusts the search space based on the success rate and model uncertainty. This ensures efficient exploration of the search space.
*   **Thompson Sampling for Kernel Selection:** Thompson Sampling, adopted from ATRSMABO_TS, is used to dynamically select between RBF and Matern kernels. This allows the algorithm to adaptively favor the kernel that is performing better, leading to more efficient exploration and exploitation.

The combination of these techniques aims to create a robust and efficient Bayesian Optimization algorithm that can handle a wide range of black-box optimization problems. By incorporating landscape analysis, gradient estimation, adaptive trust region, and Thompson Sampling for kernel selection, the algorithm can effectively balance exploration and exploitation, adapt to the characteristics of the landscape, and improve the accuracy of the surrogate model.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances

class LaGATRBO_TS:
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
        self.exploration_weight = 0.1  # Weight for exploration term in acquisition function
        self.thompson_temperature = 1.0
        self.thompson_decay = 0.95
        self.gradient_estimation_points = min(dim + 1, 5) # Number of points to use for gradient estimation

    def _sample_points(self, n_points, center=None, width=None):
        if center is None:
            sampler = qmc.Sobol(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            # Sample within the trust region
            lower_bound = np.maximum(self.bounds[0], center - width / 2)
            upper_bound = np.minimum(self.bounds[1], center + width / 2)
            sampler = qmc.Sobol(d=self.dim)
            sample = sampler.random(n=n_points)
            scaled_sample = qmc.scale(sample, lower_bound, upper_bound)
            return scaled_sample

    def _estimate_gradient(self, func, x, delta=1e-3):
        # Estimate gradient using finite differences
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += delta
            x_minus[i] -= delta
            gradient[i] = (func(x_plus) - func(x_minus)) / (2 * delta)
        return gradient

    def _fit_model(self, X, y, func):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature

        # Estimate gradients at existing points
        gradients = np.array([self._estimate_gradient(func, x) for x in X])

        # Augment the training data with gradient information
        X_augmented = np.concatenate((X, X), axis=0)
        y_augmented = np.concatenate((y, y), axis=0) # Duplicate y values
        gradients_flattened = gradients.flatten().reshape(-1, 1)

        # Define kernels
        kernel_rbf = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        kernel_matern = ConstantKernel(1.0) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)

        # Initialize models
        gp_rbf = GaussianProcessRegressor(kernel=kernel_rbf, n_restarts_optimizer=5)
        gp_matern = GaussianProcessRegressor(kernel=kernel_matern, n_restarts_optimizer=5)

        # Fit models
        gp_rbf.fit(X_augmented, y_augmented)
        gp_matern.fit(X_augmented, y_augmented)

        return gp_rbf, gp_matern

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

    def _acquisition_function(self, X, model):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement with temperature
        imp = self.best_y - mu
        Z = imp / (self.temperature * sigma + 1e-9)  # Adding a small constant to avoid division by zero
        ei = imp * norm.cdf(Z) + self.temperature * sigma * norm.pdf(Z)
        ei = np.clip(ei, 0, 1e10)  # Clip EI to avoid potential NaN issues
        ei[sigma <= 1e-6] = 0.0  # avoid division by zero

        # Uncertainty-aware exploration
        exploration_term = self.exploration_weight * sigma
        acq_values = ei + exploration_term

        return acq_values

    def _select_next_points(self, batch_size):
        n_candidates = max(1000, batch_size * 100)
        X_cand = self._sample_points(n_candidates, center=self.best_x, width=self.trust_region_width)

        # Thompson Sampling for Acquisition Function Selection
        # Sample a function value from each GPR model
        mu_rbf, sigma_rbf = self.model_rbf.predict(self.X, return_std=True)
        mu_matern, sigma_matern = self.model_matern.predict(self.X, return_std=True)

        # Sample from the posterior predictive distribution
        try:
            sampled_rbf = np.random.normal(mu_rbf, sigma_rbf * self.thompson_temperature)
            sampled_matern = np.random.normal(mu_matern, sigma_matern * self.thompson_temperature)
        except Exception as e:
            sampled_rbf = mu_rbf
            sampled_matern = mu_matern

        # Calculate acquisition function values for each model
        ei_rbf = self._acquisition_function(X_cand, self.model_rbf)
        ei_matern = self._acquisition_function(X_cand, self.model_matern)

        # Select the model with the highest sampled value
        if np.mean(sampled_rbf) > np.mean(sampled_matern):
            acq_values = ei_rbf
        else:
            acq_values = ei_matern

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
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_x = X_init[np.argmin(y_init)]
        self.best_y = np.min(y_init)

        # Optimization loop
        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            # Fit the model
            self.model_rbf, self.model_matern = self._fit_model(self.X, self.y, func)

            # Analyze the landscape
            self.landscape_correlation = self._analyze_landscape()

            # Adjust temperature based on landscape correlation
            if self.landscape_correlation > self.smoothness_threshold:
                self.temperature = max(0.1, self.temperature * 0.9)  # Reduce temperature for smoother landscapes
            else:
                self.temperature = min(2.0, self.temperature * 1.1)  # Increase temperature for rugged landscapes

            # Select next points
            X_next = self._select_next_points(batch_size)

            # Evaluate points
            y_next = self._evaluate_points(func, X_next)

            # Update evaluated points
            self._update_eval_points(X_next, y_next)

            # Update best solution
            new_best_y = np.min(self.y)
            new_best_x = self.X[np.argmin(self.y)]

            # Adjust trust region width
            success_ratio = (self.best_y - new_best_y) / self.best_y
            mean_sigma = np.mean(self.model_rbf.predict(self.X, return_std=True)[1]) # Using RBF model for uncertainty

            if success_ratio > self.success_threshold and mean_sigma < 0.5:
                self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)  # Increase if successful and low uncertainty
            else:
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)  # Decrease if not successful or high uncertainty

            self.best_y = new_best_y
            self.best_x = new_best_x

            self.thompson_temperature *= self.thompson_decay

        # Return best solution
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<LaGATRBO_TS>", line 177, in __call__
 177->             self.model_rbf, self.model_matern = self._fit_model(self.X, self.y, func)
  File "<LaGATRBO_TS>", line 62, in _fit_model
  62->         gradients = np.array([self._estimate_gradient(func, x) for x in X])
  File "<LaGATRBO_TS>", line 62, in <listcomp>
  62->         gradients = np.array([self._estimate_gradient(func, x) for x in X])
  File "<LaGATRBO_TS>", line 53, in _estimate_gradient
  51 |             x_plus[i] += delta
  52 |             x_minus[i] -= delta
  53->             gradient[i] = (func(x_plus) - func(x_minus)) / (2 * delta)
  54 |         return gradient
  55 | 
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
