# Description
**Variance-Aware Ensemble Trust Region Bayesian Optimization (VAETRBO)**: This algorithm combines the strengths of AdaptiveTrustRegionBO_DREV and AdaptiveEnsembleTrustRegionBO_EIR. It uses an ensemble of Gaussian Process Regression (GPR) models with different kernels to improve robustness and model uncertainty. It incorporates the variance of the EI values into the trust region radius adjustment strategy, balancing exploration and exploitation. It also introduces a dynamic weighting scheme for the ensemble members based on their individual performance, further enhancing the robustness and adaptability of the algorithm. Finally, a more sophisticated point selection strategy is implemented that considers both EI and GP uncertainty.

# Justification
This algorithm builds upon the strengths of the two selected algorithms. The ensemble approach from AdaptiveEnsembleTrustRegionBO_EIR improves the robustness of the model by using multiple kernels. The EI variance-based radius adjustment from AdaptiveTrustRegionBO_DREV helps to balance exploration and exploitation. The dynamic weighting scheme for the ensemble members allows the algorithm to adapt to the specific characteristics of the optimization problem. The point selection strategy based on both EI and GP uncertainty helps to avoid premature convergence and promotes exploration of promising regions of the search space.

The key changes and their justifications are:

1.  **Ensemble with Dynamic Weights**: Instead of a simple average of the ensemble predictions, a dynamic weighting scheme is introduced. The weights are updated based on the ensemble member's performance on the observed data. This allows the algorithm to prioritize models that are better suited for the specific optimization problem. The weight update is based on the negative root mean squared error (RMSE) of the GP predictions on the observed data.

2.  **EI Variance in Ensemble**: The EI variance is now calculated based on the ensemble predictions. This provides a more robust estimate of the uncertainty in the EI values.

3.  **Point Selection with GP Uncertainty**: The point selection strategy is modified to consider both EI and GP uncertainty. This helps to avoid premature convergence and promotes exploration of promising regions of the search space. The next points are selected based on a combination of EI and GP uncertainty, using an upper confidence bound (UCB) approach.

4.  **Periodic Kernel Re-optimization**: To adapt to changing function characteristics, the kernel hyperparameters of each GP in the ensemble are periodically re-optimized using L-BFGS-B. This ensures that the models remain well-calibrated throughout the optimization process.

These changes are designed to improve the robustness, adaptability, and convergence speed of the algorithm.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

class VarianceAwareEnsembleTrustRegionBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * (dim + 1)
        self.ensemble_size = 3
        self.kernels = [
            C(1.0, constant_value_bounds=(1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
            C(1.0, constant_value_bounds=(1e-2, 1e2)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5),
            C(1.0, constant_value_bounds=(1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
        ]
        self.gps = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-5) for kernel in self.kernels]
        self.weights = np.ones(self.ensemble_size) / self.ensemble_size  # Initialize weights equally
        self.update_interval = 5
        self.last_gp_update = 0
        self.trust_region_center = np.zeros(dim)
        self.trust_region_radius = 2.5
        self.min_radius = 0.1
        self.radius_decay_base = 0.95
        self.radius_grow_base = 1.1
        self.ei_scaling = 0.1
        self.ei_variance_scaling = 0.05
        self.recentering_threshold = 0.5
        self.ucb_kappa = 2.0
        self.kernel_reopt_interval = 10

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        sample = sampler.random(n=n_points)
        scaled_sample = qmc.scale(sample, -1.0, 1.0)
        points = self.trust_region_center + scaled_sample * self.trust_region_radius
        points = np.clip(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        for i, gp in enumerate(self.gps):
            gp.fit(X, y)
            # Update weights based on RMSE
            y_pred, _ = gp.predict(X, return_std=True)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            self.weights[i] = -rmse  # Use negative RMSE as weight

        # Normalize weights
        self.weights = np.maximum(0, self.weights)  # Ensure weights are non-negative
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            self.weights = np.ones(self.ensemble_size) / self.ensemble_size


        return self.gps

    def _acquisition_function(self, X):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu_list = []
        sigma_list = []
        for gp, weight in zip(self.gps, self.weights):
            mu, sigma = gp.predict(X, return_std=True)
            mu_list.append(mu * weight)
            sigma_list.append(sigma * weight)

        mu_ensemble = np.sum(mu_list, axis=0)
        # Ensemble variance calculation
        variance_ensemble = np.sum(np.square(sigma_list) + np.square(mu_list), axis=0) - np.square(mu_ensemble)
        sigma_ensemble = np.sqrt(np.maximum(variance_ensemble, 1e-9))  # Ensure non-negative variance
        sigma_ensemble = np.clip(sigma_ensemble, 1e-9, np.inf)

        y_best = np.min(self.y)
        gamma = (y_best - mu_ensemble) / sigma_ensemble
        ei = sigma_ensemble * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return ei.reshape(-1, 1), mu_ensemble.reshape(-1, 1), sigma_ensemble.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Sample candidate points
        candidates = self._sample_points(100 * self.dim)

        # Calculate acquisition function values
        ei, mu, sigma = self._acquisition_function(candidates)

        # UCB acquisition
        ucb = mu + self.ucb_kappa * sigma

        # Select top batch_size candidates based on UCB
        selected_indices = np.argsort(ucb.flatten())[-batch_size:]
        selected_points = candidates[selected_indices]

        return selected_points

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        while self.n_evals < self.budget:
            if self.n_evals - self.last_gp_update >= self.update_interval:
                self._fit_model(self.X, self.y)
                self.last_gp_update = self.n_evals

                if self.n_evals % self.kernel_reopt_interval == 0:
                    for gp in self.gps:
                        gp.kernel_ = gp.kernel_.clone_with_theta(gp.kernel_.theta)  # Reset optimizer state
                        gp.fit(self.X, self.y)

            batch_size = min(self.budget - self.n_evals, 5)
            next_X = self._select_next_points(batch_size)

            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            best_idx = np.argmin(self.y)
            current_best_y = self.y[best_idx][0]
            current_best_x = self.X[best_idx]

            # Calculate EI values and statistics
            ei_values, _, _ = self._acquisition_function(next_X)
            avg_ei = np.mean(ei_values)
            ei_variance = np.var(ei_values)

            if current_best_y < best_y:
                # Improvement: shrink radius, considering EI and variance
                decay_rate = self.radius_decay_base + self.ei_scaling * avg_ei - self.ei_variance_scaling * ei_variance
                decay_rate = max(decay_rate, 0.5) # Ensure a minimum decay
                self.trust_region_radius *= decay_rate
                self.trust_region_center = current_best_x
                best_y = current_best_y
                best_x = current_best_x
            else:
                # No improvement: expand radius, but slower if EI is high
                self.trust_region_radius *= (self.radius_grow_base - self.ei_scaling * avg_ei + self.ei_variance_scaling * ei_variance)
                self.trust_region_radius = min(self.trust_region_radius, 2.5)

            self.trust_region_radius = max(self.trust_region_radius, self.min_radius)

            # Re-center trust region if current center is far from best point
            distance = np.linalg.norm(self.trust_region_center - best_x)
            if distance > self.recentering_threshold * self.trust_region_radius:
                self.trust_region_center = best_x

        return best_y, best_x
```
## Feedback
 The algorithm VarianceAwareEnsembleTrustRegionBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1368 with standard deviation 0.1010.

took 105.54 seconds to run.