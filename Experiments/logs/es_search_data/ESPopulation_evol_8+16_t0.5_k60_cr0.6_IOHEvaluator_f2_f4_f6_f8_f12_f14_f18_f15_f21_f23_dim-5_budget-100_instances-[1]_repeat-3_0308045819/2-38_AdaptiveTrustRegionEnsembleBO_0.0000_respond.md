# Description
**Adaptive Trust Region Ensemble Bayesian Optimization with Dynamic Acquisition Blending (ATREBO-DAB):** This algorithm builds upon the TrustRegionEnsembleBO (TREBO) by introducing dynamic blending of acquisition functions and adaptive kernel selection within the Gaussian Process (GP) ensemble. It combines Thompson Sampling (TS) and Lower Confidence Bound (LCB) acquisition functions, dynamically adjusting their weights based on the optimization progress and the trust region's success. Furthermore, instead of fixing the kernel parameters, the algorithm optimizes the length scale of each GP kernel in the ensemble using a gradient-based method. The trust region radius is also adjusted more aggressively based on the success ratio.

# Justification
1.  **Dynamic Acquisition Blending:** Combining Thompson Sampling (exploration) and Lower Confidence Bound (exploitation) allows for a more robust exploration-exploitation balance. Dynamically adjusting the weights based on the optimization progress (e.g., using LCB more in later stages) improves convergence. The blending is controlled by a `lcb_weight` parameter that is increased over time.
2.  **Adaptive Kernel Tuning:** Optimizing the kernel parameters (length scale) of each GP in the ensemble allows the GPs to better adapt to the local characteristics of the objective function. This is done using L-BFGS-B optimization.
3.  **Aggressive Trust Region Adaptation:** The trust region radius is adjusted more aggressively based on the success ratio, allowing for faster convergence in promising regions.
4.  **Batch Size Adaptation:** The batch size is dynamically adjusted based on the remaining budget to allow for more evaluations in the later stages of the optimization.
5.  **Validation Error Estimation:** Using validation error to estimate the ensemble weights improves the robustness of the algorithm.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc, norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

class AdaptiveTrustRegionEnsembleBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(5 * dim, self.budget // 10)
        self.gp_ensemble = []
        self.ensemble_weights = []
        self.best_x = None
        self.best_y = float('inf')
        self.n_ensemble = 3
        self.trust_region_radius = 2.0
        self.radius_decay = 0.85  # More aggressive decay
        self.radius_increase = 1.2  # More aggressive increase
        self.min_radius = 0.1
        self.success_ratio = 0.0
        self.random_restart_prob = 0.05
        self.lcb_weight = 0.1  # Initial LCB weight
        self.lcb_weight_increase = 0.05
        self.kappa = 1.96 # Kappa parameter for LCB

    def _sample_points(self, n_points):
        if self.best_x is None or np.random.rand() < self.random_restart_prob:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            points = []
            while len(points) < n_points:
                sample = np.random.uniform(-1, 1, size=(self.dim,))
                sample = self.best_x + self.trust_region_radius * sample
                sample = np.clip(sample, self.bounds[0], self.bounds[1])
                points.append(sample)
            return np.array(points)

    def _fit_model(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        if not self.gp_ensemble:
            kernels = [
                ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
                ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=0.5),
                ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
            ]
            for i in range(self.n_ensemble):
                gp = GaussianProcessRegressor(kernel=kernels[i % len(kernels)], n_restarts_optimizer=3, alpha=1e-6)
                gp.fit(X_train, y_train)
                self.gp_ensemble.append(gp)
                self.ensemble_weights.append(1.0 / self.n_ensemble)
        else:
            for gp in self.gp_ensemble:
                # Optimize kernel parameters
                def obj(x):
                    gp.kernel_.k2.length_scale = x
                    return -gp.log_marginal_likelihood(gp.X_train_, gp.y_train_)

                initial_length_scale = gp.kernel_.k2.length_scale
                bounds = gp.kernel_.k2.length_scale_bounds
                result = minimize(obj, initial_length_scale, method='L-BFGS-B', bounds=[bounds])
                gp.kernel_.k2.length_scale = result.x
                gp.fit(X_train, y_train)

        val_errors = []
        for gp in self.gp_ensemble:
            y_pred, _ = gp.predict(X_val, return_std=True)
            error = np.mean((y_pred - y_val.flatten())**2)
            val_errors.append(error)

        val_errors = np.array(val_errors)
        weights = np.exp(-val_errors) / np.sum(np.exp(-val_errors))
        self.ensemble_weights = weights

    def _acquisition_function(self, X):
        # Dynamic Acquisition Blending: Thompson Sampling + LCB
        if not self.gp_ensemble:
            return np.random.normal(size=(len(X), 1))
        else:
            ts_values = np.zeros((len(X), self.n_ensemble))
            lcb_values = np.zeros((len(X), self.n_ensemble))
            for i, gp in enumerate(self.gp_ensemble):
                mu, sigma = gp.predict(X, return_std=True)
                ts_values[:, i] = np.random.normal(mu, sigma)
                lcb_values[:, i] = mu - self.kappa * sigma

            # Average across the ensemble
            ts_values = np.mean(ts_values, axis=1)
            lcb_values = np.mean(lcb_values, axis=1)

            # Blend TS and LCB
            acquisition_values = (1 - self.lcb_weight) * ts_values + self.lcb_weight * lcb_values
            return acquisition_values.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(100 * batch_size)
        acquisition_values = self._acquisition_function(candidates)
        best_indices = np.argsort(acquisition_values.flatten())[:batch_size]
        return candidates[best_indices]

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
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]
            self.success_ratio = 1.0
        else:
            self.success_ratio *= 0.75

    def _adjust_trust_region(self):
        if self.success_ratio > 0.5:
            self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)

        self.lcb_weight = min(self.lcb_weight + self.lcb_weight_increase, 1.0)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        batch_size = min(2, self.dim)
        while self.n_evals < self.budget:
            remaining_evals = self.budget - self.n_evals
            batch_size = min(batch_size, remaining_evals) # Adjust batch size

            self._fit_model(self.X, self.y)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self._adjust_trust_region()

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<AdaptiveTrustRegionEnsembleBO>", line 149, in __call__
 149->             self._fit_model(self.X, self.y)
  File "<AdaptiveTrustRegionEnsembleBO>", line 70, in _fit_model
  70->                 result = minimize(obj, initial_length_scale, method='L-BFGS-B', bounds=[bounds])
  File "<AdaptiveTrustRegionEnsembleBO>", line 66, in obj
  64 |                 def obj(x):
  65 |                     gp.kernel_.k2.length_scale = x
  66->                     return -gp.log_marginal_likelihood(gp.X_train_, gp.y_train_)
  67 | 
  68 |                 initial_length_scale = gp.kernel_.k2.length_scale
ValueError: theta has not the correct number of entries. Should be 1; given are 8
