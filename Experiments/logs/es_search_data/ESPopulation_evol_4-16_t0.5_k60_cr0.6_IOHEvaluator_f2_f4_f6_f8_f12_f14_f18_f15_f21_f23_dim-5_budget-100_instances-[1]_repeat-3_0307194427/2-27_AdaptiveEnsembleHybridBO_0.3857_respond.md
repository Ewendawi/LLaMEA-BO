# Description
**Adaptive Ensemble Hybrid Bayesian Optimization (AEHBO)**: This algorithm synergistically combines the strengths of AdaptiveEnsembleTrustRegionBO and EnhancedEfficientHybridBO. It leverages an ensemble of Gaussian Process Regression (GPR) models, similar to AdaptiveEnsembleTrustRegionBO, to improve the robustness and exploration of the search space. It also incorporates the adaptive kernel lengthscale optimization and dynamic batch size strategy of EnhancedEfficientHybridBO to efficiently capture the function's characteristics and balance exploration and exploitation. Instead of a fixed trust region, AEHBO uses a dynamic constraint based on the GPR uncertainty to encourage exploration in uncertain regions. The next points are selected by optimizing the EI acquisition function.

# Justification
The combination of AdaptiveEnsembleTrustRegionBO and EnhancedEfficientHybridBO allows the algorithm to achieve a better balance between exploration and exploitation. The ensemble of GPR models improves the robustness of the algorithm by reducing the risk of overfitting to a single model. The adaptive kernel lengthscale optimization and dynamic batch size strategy improve the efficiency of the algorithm by allowing it to focus on promising regions of the search space. The dynamic constraint based on the GPR uncertainty encourages exploration in uncertain regions, which can help the algorithm to escape local optima.

*   **Ensemble of GPR Models**: Using an ensemble of GPR models with different kernels enhances the robustness and exploration capabilities of the algorithm. Each GPR model captures different aspects of the function's characteristics, and their combined predictions provide a more reliable estimate of the function's behavior.
*   **Adaptive Kernel Lengthscale Optimization**: Optimizing the kernel lengthscale periodically allows the algorithm to adapt to the changing characteristics of the function. This is particularly important for non-stationary functions, where the function's behavior changes over time.
*   **Dynamic Batch Size Strategy**: Adjusting the batch size dynamically based on the uncertainty of the GPR model helps to balance exploration and exploitation. When the uncertainty is high, the algorithm increases the batch size to explore more of the search space. When the uncertainty is low, the algorithm decreases the batch size to focus on exploiting promising regions.
*   **Dynamic Uncertainty Constraint**: Instead of a fixed trust region, AEHBO uses a dynamic constraint based on the GPR uncertainty to encourage exploration in uncertain regions. This allows the algorithm to adapt to the changing characteristics of the function and to escape local optima.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

class AdaptiveEnsembleHybridBO:
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
        self.update_interval = 5
        self.last_gp_update = 0
        self.length_scale = 1.0
        self.kernel_optim_interval = 5

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        for gp in self.gps:
            gp.fit(X, y)
        return self.gps

    def _optimize_kernel(self):
        def obj(length_scale):
            kernels = [
                C(1.0, constant_value_bounds=(1e-2, 1e2)) * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2)),
                C(1.0, constant_value_bounds=(1e-2, 1e2)) * Matern(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2), nu=1.5),
                C(1.0, constant_value_bounds=(1e-2, 1e2)) * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
            ]
            gps = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-5) for kernel in kernels]
            log_likelihood = 0
            for gp in gps:
                gp.fit(self.X, self.y)
                log_likelihood += gp.log_marginal_likelihood()
            return -log_likelihood

        res = minimize(obj, x0=self.length_scale, method='L-BFGS-B', bounds=[(1e-2, 10)])
        self.length_scale = res.x[0]
        for i in range(self.ensemble_size):
            if isinstance(self.gps[i].kernel_, C):
                if isinstance(self.gps[i].kernel_.k2, RBF):
                    self.gps[i].kernel_.k2.length_scale = self.length_scale
                elif isinstance(self.gps[i].kernel_.k2, Matern):
                    self.gps[i].kernel_.k2.length_scale = self.length_scale
            elif isinstance(self.gps[i].kernel_, RBF):
                self.gps[i].kernel_.length_scale = self.length_scale


    def _acquisition_function(self, X):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu_list = []
        sigma_list = []
        for gp in self.gps:
            mu, sigma = gp.predict(X, return_std=True)
            mu_list.append(mu)
            sigma_list.append(sigma)

        mu_ensemble = np.mean(mu_list, axis=0)
        sigma_ensemble = np.sqrt(np.mean(np.square(sigma_list) + np.square(mu_list), axis=0) - np.square(mu_ensemble))
        sigma_ensemble = np.clip(sigma_ensemble, 1e-9, np.inf)
        y_best = np.min(self.y)
        gamma = (y_best - mu_ensemble) / sigma_ensemble
        ei = sigma_ensemble * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(100 * self.dim)
        ei = self._acquisition_function(candidates).flatten()

        # Apply uncertainty constraint
        _, sigma_list = self._predict(candidates, return_std=True)
        uncertainty = np.mean(sigma_list, axis=0)
        uncertainty_threshold = np.percentile(uncertainty, 25)  # Adjust percentile as needed
        eligible_candidates = candidates[uncertainty > uncertainty_threshold]

        if len(eligible_candidates) == 0:
            selected_indices = np.argsort(ei)[-batch_size:]
            selected_points = candidates[selected_indices]
        else:
            eligible_ei = self._acquisition_function(eligible_candidates).flatten()
            selected_indices = np.argsort(eligible_ei)[-batch_size:]
            selected_points = eligible_candidates[selected_indices]

        # Ensure diversity by penalizing points that are too close to existing points
        if self.X is not None:
            distances = cdist(selected_points, self.X)
            min_distances = np.min(distances, axis=1)
            # Only select points that are sufficiently far away from existing points
            selected_points = selected_points[min_distances > 0.1]
            if len(selected_points) < batch_size:
              remaining_needed = batch_size - len(selected_points)
              additional_indices = np.argsort(ei)[:-batch_size-1:-1][:remaining_needed]
              additional_points = candidates[additional_indices]
              selected_points = np.concatenate([selected_points, additional_points], axis=0)
        return selected_points[:batch_size]

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

    def _predict(self, X, return_std=True):
        mu_list = []
        sigma_list = []
        for gp in self.gps:
            mu, sigma = gp.predict(X, return_std=return_std)
            mu_list.append(mu)
            sigma_list.append(sigma if return_std else np.zeros_like(mu))

        mu_ensemble = np.mean(mu_list, axis=0)
        if return_std:
            sigma_ensemble = np.sqrt(np.mean(np.square(sigma_list) + np.square(mu_list), axis=0) - np.square(mu_ensemble))
            return mu_ensemble, sigma_ensemble
        else:
            return mu_ensemble, None

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        iteration = 0
        while self.n_evals < self.budget:
            if self.n_evals - self.last_gp_update >= self.update_interval:
                self._fit_model(self.X, self.y)
                self.last_gp_update = self.n_evals

            # Optimize kernel lengthscale periodically
            if iteration % self.kernel_optim_interval == 0:
                self._optimize_kernel()
                self._fit_model(self.X, self.y)

            # Dynamic batch size
            _, sigma = self._predict(self.X, return_std=True)
            avg_sigma = np.mean(sigma)
            batch_size = min(self.budget - self.n_evals, int(np.ceil(5 * (avg_sigma / np.std(self.bounds))))) if np.std(self.bounds) > 0 else min(self.budget - self.n_evals, 5)
            batch_size = max(1, batch_size)

            next_X = self._select_next_points(batch_size)

            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            best_idx = np.argmin(self.y)
            best_y = self.y[best_idx][0]
            best_x = self.X[best_idx]

            iteration += 1

        return best_y, best_x
```
## Feedback
 The algorithm AdaptiveEnsembleHybridBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1625 with standard deviation 0.1064.

took 701.50 seconds to run.