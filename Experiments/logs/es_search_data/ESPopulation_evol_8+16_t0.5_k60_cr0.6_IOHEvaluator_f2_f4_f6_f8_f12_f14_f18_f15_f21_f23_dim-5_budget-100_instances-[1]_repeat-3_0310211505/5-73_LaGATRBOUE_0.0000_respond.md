# Description
**LaGATRBOUE**: Landscape-Aware Gradient-enhanced Adaptive Trust Region Bayesian Optimization with Uncertainty-Aware Exploration. This algorithm enhances the exploration-exploitation balance by integrating landscape analysis, gradient estimation, adaptive trust region management, and uncertainty-aware exploration. It estimates gradients using finite differences to refine the Gaussian Process Regression (GPR) model. Landscape analysis dynamically adjusts the temperature in the Expected Improvement (EI) acquisition function. Uncertainty-aware exploration is incorporated by adding a term proportional to the predicted standard deviation to the EI, encouraging exploration in regions of high uncertainty. The trust region is adapted based on the success rate and model uncertainty. A key improvement is the dynamic adjustment of the delta parameter used in gradient estimation, adapting it to the trust region size and GPR model uncertainty. L-BFGS-B optimization is used within the trust region to select the next points.

# Justification
This algorithm builds upon the strengths of ALGTRBOUE and LaAGETRBOUE. It retains the landscape analysis and uncertainty-aware exploration from ALGTRBOUE, and the gradient estimation from LaAGETRBOUE. The crucial addition is the dynamic adjustment of the delta parameter used in gradient estimation, which is inspired by ATRBODGEAD.

*   **Landscape Analysis:** The landscape correlation provides valuable information about the structure of the objective function. A high correlation suggests a smooth landscape, allowing for more exploitation, while a low correlation indicates a rugged landscape, requiring more exploration.

*   **Gradient Estimation:** Incorporating gradient information into the GPR model can significantly improve its accuracy, especially in high-dimensional spaces. By estimating gradients using finite differences, the algorithm can better exploit the local landscape.

*   **Uncertainty-Aware Exploration:** Adding an uncertainty-aware exploration term to the EI acquisition function encourages the algorithm to explore regions of high uncertainty, which can lead to the discovery of new optima.

*   **Adaptive Trust Region:** The adaptive trust region mechanism helps to balance exploration and exploitation by adjusting the size of the trust region based on the success rate and model uncertainty.

*   **Dynamic Delta for Gradient Estimation:** The delta parameter used in the finite difference method is crucial for accurate gradient estimation. A smaller delta provides more accurate estimates in smooth regions, while a larger delta is needed to explore rugged regions. By dynamically adjusting the delta based on the trust region size and model uncertainty, the algorithm can improve the efficiency and accuracy of gradient estimation.

*   **L-BFGS-B Optimization:** Using L-BFGS-B to optimize the acquisition function within the trust region allows for efficient exploitation of the local landscape.

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
        self.delta = 0.01  # Initial delta for gradient estimation
        self.delta_scaling = 0.1  # Scaling factor for delta adjustment

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
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus_delta = x.copy()
            x_minus_delta = x.copy()
            x_plus_delta[i] += self.delta
            x_minus_delta[i] -= self.delta

            # Clip values to stay within bounds
            x_plus_delta[i] = np.clip(x_plus_delta[i], self.bounds[0][i], self.bounds[1][i])
            x_minus_delta[i] = np.clip(x_minus_delta[i], self.bounds[0][i], self.bounds[1][i])

            gradient[i] = (func(x_plus_delta) - func(x_minus_delta)) / (2 * self.delta)
        return gradient

    def _fit_model(self, X, y, func):
        # Estimate gradients at each point
        gradients = np.array([self._estimate_gradient(func, x) for x in X])
        
        # Augment the training data with gradient information
        X_augmented = np.concatenate((X, X), axis=0)
        y_augmented = np.concatenate((y, y), axis=0)
        gradients_flattened = gradients.flatten().reshape(-1, 1)
        X_gradient_features = np.tile(np.eye(self.dim), (len(X), 1))
        X_augmented = np.concatenate((X_augmented, X_gradient_features), axis=0)
        y_augmented = np.concatenate((y_augmented, gradients_flattened), axis=0)

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

    def _acquisition_function(self, X):
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
        # Define the acquisition function to be minimized
        def objective(x):
            return -self._acquisition_function(x.reshape(1, -1)).flatten()

        # Initial guess: best_x
        x0 = self.best_x

        # Define bounds for L-BFGS-B
        bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]

        # Run L-BFGS-B optimization
        results = []
        for _ in range(batch_size):  # Generate multiple points
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            results.append(res.x)
            x0 = self._sample_points(1, center=self.best_x, width=self.trust_region_width).flatten() # perturb the initial guess

        return np.array(results)

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
            self.model = self._fit_model(self.X, self.y, func)

            # Analyze the landscape
            self.landscape_correlation = self._analyze_landscape()

            # Adjust temperature based on landscape correlation
            if self.landscape_correlation > self.smoothness_threshold:
                self.temperature = max(0.1, self.temperature * 0.9)  # Reduce temperature for smoother landscapes
            else:
                self.temperature = min(2.0, self.temperature * 1.1)  # Increase temperature for rugged landscapes

            # Adjust delta based on trust region width and model uncertainty
            mean_sigma = np.mean(self.model.predict(self.X, return_std=True)[1])
            self.delta = min(self.trust_region_width / 10, max(0.001, self.delta * (1 + self.delta_scaling * (mean_sigma - 0.5))))


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
            mean_sigma = np.mean(self.model.predict(self.X, return_std=True)[1])

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
## Error
 Traceback (most recent call last):
  File "<LaGATRBOUE>", line 161, in __call__
 161->             self.model = self._fit_model(self.X, self.y, func)
  File "<LaGATRBOUE>", line 62, in _fit_model
  62->         gradients = np.array([self._estimate_gradient(func, x) for x in X])
  File "<LaGATRBOUE>", line 62, in <listcomp>
  62->         gradients = np.array([self._estimate_gradient(func, x) for x in X])
  File "<LaGATRBOUE>", line 57, in _estimate_gradient
  55 |             x_minus_delta[i] = np.clip(x_minus_delta[i], self.bounds[0][i], self.bounds[1][i])
  56 | 
  57->             gradient[i] = (func(x_plus_delta) - func(x_minus_delta)) / (2 * self.delta)
  58 |         return gradient
  59 | 
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
