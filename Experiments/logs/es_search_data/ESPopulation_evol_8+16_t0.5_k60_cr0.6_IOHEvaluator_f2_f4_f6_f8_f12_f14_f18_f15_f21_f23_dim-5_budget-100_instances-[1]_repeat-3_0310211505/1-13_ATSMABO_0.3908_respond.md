# Description
Adaptive Temperature Surrogate Model Averaging Bayesian Optimization (ATSMABO) is a Bayesian optimization algorithm that builds upon SMABO by introducing an adaptive temperature schedule and a more refined surrogate model averaging strategy. Instead of a fixed temperature decay, ATSMABO adjusts the temperature based on the landscape characteristics encountered during the optimization process. Specifically, it monitors the change in the best function value and adjusts the temperature decay rate accordingly. Furthermore, ATSMABO introduces a weighted averaging of the surrogate model predictions, where the weights are determined by the models' performance on a validation set.

# Justification
The key improvements in ATSMABO are justified as follows:

1.  **Adaptive Temperature:** The original SMABO uses a fixed temperature decay, which may not be optimal for all landscapes. By adapting the temperature decay based on the observed improvement in the best function value, ATSMABO can dynamically adjust the exploration-exploitation trade-off. If the optimization is progressing well (i.e., the best function value is improving rapidly), the temperature decay is increased to focus on exploitation. Conversely, if the optimization is stagnating, the temperature decay is decreased to encourage exploration. This adaptive strategy allows the algorithm to better adapt to the characteristics of the objective function.

2.  **Weighted Surrogate Model Averaging:** The original SMABO averages the predictions of the RBF and Matern kernels with equal weights. However, one kernel might be better suited for a particular problem than the other. ATSMABO uses a validation set to estimate the performance of each kernel and assigns weights accordingly. This allows the algorithm to leverage the strengths of each kernel and improve the accuracy of the surrogate model.

3. **Validation Set:** The validation set is created by splitting the current data into training and validation. This allows the algorithm to estimate the performance of each kernel on unseen data and assign weights accordingly.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from sklearn.model_selection import train_test_split


class ATSMABO:
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
        self.best_y = np.inf  # Initialize best_y with a large value
        self.last_best_y = np.inf  # Store the last best y for adaptive temperature

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define kernels
        kernel_rbf = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        kernel_matern = ConstantKernel(1.0) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)

        # Initialize models
        gp_rbf = GaussianProcessRegressor(kernel=kernel_rbf, n_restarts_optimizer=5)
        gp_matern = GaussianProcessRegressor(kernel=kernel_matern, n_restarts_optimizer=5)

        # Fit models
        gp_rbf.fit(X_train, y_train)
        gp_matern.fit(X_train, y_train)

        # Calculate validation error
        y_pred_rbf, _ = gp_rbf.predict(X_val, return_std=True)
        y_pred_matern, _ = gp_matern.predict(X_val, return_std=True)

        error_rbf = np.mean((y_val - y_pred_rbf.reshape(-1, 1)) ** 2)
        error_matern = np.mean((y_val - y_pred_matern.reshape(-1, 1)) ** 2)

        # Calculate weights based on validation error
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

        # Weighted average of predictions
        mu = self.weight_rbf * mu_rbf + self.weight_matern * mu_matern
        sigma = self.weight_rbf * sigma_rbf + self.weight_matern * sigma_matern

        # Expected Improvement with temperature
        best = np.min(self.y)
        imp = best - mu
        Z = imp / (self.temperature * sigma + 1e-9)
        ei = imp * norm.cdf(Z) + self.temperature * sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        return ei

    def _select_next_points(self, batch_size):
        n_candidates = max(2000, batch_size * 200)
        X_cand = self._sample_points(n_candidates)
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
        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_y = np.min(y_init)
        self.last_best_y = self.best_y

        # Optimization loop
        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            # Fit the model
            self.model_rbf, self.model_matern, self.weight_rbf, self.weight_matern = self._fit_model(self.X, self.y)

            # Select next points
            X_next = self._select_next_points(batch_size)

            # Evaluate points
            y_next = self._evaluate_points(func, X_next)

            # Update evaluated points
            self._update_eval_points(X_next, y_next)

            # Update best y
            current_best_y = np.min(self.y)

            # Adaptive temperature decay
            if current_best_y < self.best_y:
                # Significant improvement, reduce exploration
                self.temperature_decay = min(self.temperature_decay + 0.02, self.temperature_decay_max)
            else:
                # Stagnation, increase exploration
                self.temperature_decay = max(self.temperature_decay - 0.02, self.temperature_decay_min)

            self.temperature *= self.temperature_decay
            self.best_y = current_best_y

        # Return best solution
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Feedback
 The algorithm ATSMABO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1638 with standard deviation 0.1028.

took 298.09 seconds to run.