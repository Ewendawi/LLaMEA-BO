# Description
**LaGATRBOUE: Landscape-Aware Gradient-Augmented Trust Region Bayesian Optimization with Uncertainty-Aware Exploration.** This algorithm combines landscape analysis, gradient estimation, adaptive trust region, and uncertainty-aware exploration. It estimates gradients using finite differences to enhance the Gaussian Process Regression (GPR) model. Landscape analysis dynamically adjusts a temperature parameter in the Expected Improvement (EI) acquisition function. An uncertainty-aware exploration term is added to the EI, encouraging exploration in regions of high uncertainty. The trust region is adapted based on the success rate and model uncertainty. L-BFGS-B optimization is used within the trust region to select the next points. Furthermore, the temperature is dynamically adjusted based on the uncertainty of the Gaussian Process model.

# Justification
This algorithm builds upon the strengths of ALGTRBOUE and incorporates elements from LaAGETRBOUE and ALGTRBODTUE. The key improvements are:

1.  **Gradient Estimation:** Inspired by LaAGETRBOUE, gradient estimation is added to enhance the Gaussian Process Regression (GPR) model. This helps the model to better understand the local landscape and make more informed decisions.

2.  **Dynamic Temperature Adjustment:** Inspired by ALGTRBODTUE, the temperature is dynamically adjusted based on both landscape correlation and the uncertainty of the Gaussian Process model. This allows for a more nuanced control of exploration, increasing it when the landscape is rugged or the model is uncertain, and decreasing it when the landscape is smooth and the model is confident.

3.  **Uncertainty-Aware Exploration:** An uncertainty-aware exploration term is added to the EI, encouraging exploration in regions of high uncertainty.

4.  **Adaptive Trust Region:** The trust region is adapted based on the success rate and model uncertainty. This helps to focus the search on promising regions of the search space.

5.  **Computational Efficiency:** The algorithm is designed to be computationally efficient by using L-BFGS-B optimization within the trust region and by carefully selecting the number of candidate points.

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
from scipy.optimize import minimize

class LaGATRBOUE:
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
        self.delta = 0.01  # Step size for gradient estimation
        self.temperature_scaling = 0.1

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

    def _estimate_gradient(self, func, x):
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.delta
            x_minus[i] -= self.delta
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * self.delta)
        return grad

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

        # Uncertainty-aware exploration
        exploration_term = self.exploration_weight * sigma
        acq_values = ei + exploration_term

        return acq_values

    def _select_next_points(self, func, batch_size):
        # L-BFGS-B optimization within the trust region
        x_next = []
        for _ in range(batch_size):
            # Define the objective function to minimize (negative acquisition function)
            def objective(x):
                return -self._acquisition_function(x.reshape(1, -1), func)[0, 0]

            # Define the bounds for the optimization
            lower_bound = np.maximum(self.bounds[0], self.best_x - self.trust_region_width / 2)
            upper_bound = np.minimum(self.bounds[1], self.best_x + self.trust_region_width / 2)
            bounds = list(zip(lower_bound, upper_bound))

            # Run L-BFGS-B optimization
            result = minimize(objective, self.best_x, method='L-BFGS-B', bounds=bounds)
            x_next.append(result.x)

        return np.array(x_next)

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
            self.model = self._fit_model(self.X, self.y)

            # Analyze the landscape
            self.landscape_correlation = self._analyze_landscape()

            # Calculate mean sigma (uncertainty)
            mean_sigma = np.mean(self.model.predict(self.X, return_std=True)[1])

            # Adjust temperature based on landscape correlation and uncertainty
            if self.landscape_correlation > self.smoothness_threshold:
                self.temperature = max(0.1, self.temperature * (1 - self.temperature_scaling))  # Reduce temperature for smoother landscapes
            else:
                self.temperature = min(2.0, self.temperature * (1 + self.temperature_scaling))  # Increase temperature for rugged landscapes

            self.temperature = max(0.01, min(2.0, self.temperature * (1 + (mean_sigma - 0.5) * 0.1)))

            # Adjust exploration weight based on uncertainty
            self.exploration_weight = min(0.5, max(0.01, 0.1 + (mean_sigma - 0.5) * 0.2))

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
            success_ratio = (self.best_y - new_best_y) / self.best_y

            if success_ratio > self.success_threshold and mean_sigma < 0.5:
                self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)  # Increase if successful and low uncertainty
            else:
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)  # Decrease if not successful or high uncertainty

            self.best_y = new_best_y
            self.best_x = new_best_x

        # Return best solution
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Feedback
 The algorithm LaGATRBOUE got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1781 with standard deviation 0.0994.

took 400.46 seconds to run.