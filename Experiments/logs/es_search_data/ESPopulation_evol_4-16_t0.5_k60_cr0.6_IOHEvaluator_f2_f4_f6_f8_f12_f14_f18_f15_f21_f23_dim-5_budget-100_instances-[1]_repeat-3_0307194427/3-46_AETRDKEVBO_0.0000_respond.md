# Description
**Adaptive Ensemble Trust Region with Dynamic Kernel and EI-Variance Improvement (AETRDKEVBO)**: This algorithm combines the strengths of AdaptiveEnsembleTrustRegionBO_EIR and ATRKTEIBO, incorporating the dynamic kernel tuning from ATRKTEIBO and the ensemble modeling from AdaptiveEnsembleTrustRegionBO_EIR. It further enhances the trust region radius adaptation by considering the variance of the EI values, similar to AdaptiveTrustRegionBO_DREV. The algorithm uses an ensemble of Gaussian Process Regression (GPR) models with different kernels to improve robustness, dynamically tunes the kernel lengthscale, and adjusts the trust region radius based on both the average and variance of the Expected Improvement (EI) to balance exploration and exploitation.

# Justification
The algorithm builds upon the strengths of the ensemble approach (AETRBO-EIR) and dynamic kernel tuning (ATRKTEIBO).
1.  **Ensemble Modeling**: Using an ensemble of GPs with different kernels provides robustness and handles different function characteristics better than a single GP.
2.  **Dynamic Kernel Tuning**: Periodically optimizing the kernel lengthscale allows the GP models to adapt to the local function landscape, improving the accuracy of predictions and the EI values.
3.  **EI-Variance based Radius Adjustment**: Adjusting the trust region radius based on both the average and variance of EI provides a more nuanced approach to balancing exploration and exploitation. High average EI with low variance suggests exploitation, while high average EI with high variance indicates potential for further exploration.
4.  **Computational Efficiency**: The kernel optimization is performed periodically, rather than at every iteration, to reduce computational cost. The batch size is dynamically adjusted based on the GP uncertainty to improve sample efficiency.
5.  **Diversity Promotion**: The diversity threshold from ATRKTEIBO is incorporated to prevent premature convergence by encouraging exploration of less-visited regions.

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

class AETRDKEVBO:
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
        self.trust_region_center = np.zeros(dim)
        self.trust_region_radius = 2.5
        self.min_radius = 0.1
        self.radius_decay_base = 0.95
        self.radius_grow_base = 1.1
        self.ei_scaling = 0.1
        self.recentering_threshold = 0.5
        self.length_scale = 1.0
        self.kernel_optim_interval = 5
        self.diversity_threshold = 0.1

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        sample = sampler.random(n=n_points)
        scaled_sample = qmc.scale(sample, -1.0, 1.0)
        points = self.trust_region_center + scaled_sample * self.trust_region_radius
        points = np.clip(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        for gp in self.gps:
            gp.fit(X, y)
        return self.gps

    def _optimize_kernel(self):
        def obj(length_scale):
            kernel = C(1.0, constant_value_bounds="fixed") * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 10))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-5)
            gp.fit(self.X, self.y)
            return -gp.log_marginal_likelihood()

        res = minimize(obj, x0=self.length_scale, method='L-BFGS-B', bounds=[(1e-2, 10)])
        self.length_scale = res.x[0]
        # Update length_scale for all RBF kernels in the ensemble
        for gp in self.gps:
            if isinstance(gp.kernel_.k2, RBF):
                gp.kernel_ = C(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.length_scale, length_scale_bounds=(1e-2, 10)) + gp.kernel_.k1
            elif isinstance(gp.kernel_, RBF):
                gp.kernel_ = C(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.length_scale, length_scale_bounds=(1e-2, 10))

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
        # Sample candidate points
        candidates = self._sample_points(100 * self.dim)
        
        # Calculate acquisition function values
        ei = self._acquisition_function(candidates)
        
        # Select top batch_size candidates based on EI
        selected_indices = np.argsort(ei.flatten())[-batch_size:]
        selected_points = candidates[selected_indices]

        if self.X is not None:
            distances = cdist(selected_points, self.X)
            min_distances = np.min(distances, axis=1)
            selected_points = selected_points[min_distances > self.diversity_threshold]
            if len(selected_points) < batch_size:
                remaining_needed = batch_size - len(selected_points)
                additional_indices = np.argsort(ei.flatten())[:-batch_size-1:-1][:remaining_needed]
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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        iteration = 0
        while self.n_evals < self.budget:
            # Kernel Optimization
            if iteration % self.kernel_optim_interval == 0:
                self._optimize_kernel()

            if self.n_evals - self.last_gp_update >= self.update_interval:
                self._fit_model(self.X, self.y)
                self.last_gp_update = self.n_evals

            # Dynamic batch size
            mu_list = []
            sigma_list = []
            for gp in self.gps:
                mu, sigma = gp.predict(self.X, return_std=True)
                mu_list.append(mu)
                sigma_list.append(sigma)

            sigma_ensemble = np.sqrt(np.mean(np.square(sigma_list) + np.square(mu_list), axis=0) - np.square(np.mean(mu_list, axis=0)))
            avg_sigma = np.mean(sigma_ensemble)

            batch_size = min(self.budget - self.n_evals, int(np.ceil(5 * (avg_sigma / np.std(self.bounds))))) if np.std(self.bounds) > 0 else min(self.budget - self.n_evals, 5)
            batch_size = max(1, batch_size)

            next_X = self._select_next_points(batch_size)

            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            best_idx = np.argmin(self.y)
            current_best_y = self.y[best_idx][0]
            current_best_x = self.X[best_idx]
            
            # Calculate average EI and variance of EI of evaluated points
            ei_values = self._acquisition_function(next_X)
            avg_ei = np.mean(ei_values)
            ei_variance = np.var(ei_values)

            if current_best_y < best_y:
                # Improvement: shrink radius based on EI, consider variance
                self.trust_region_radius *= (self.radius_decay_base + self.ei_scaling * (avg_ei - ei_variance))
                self.trust_region_center = current_best_x
                best_y = current_best_y
                best_x = current_best_x
            else:
                # No improvement: expand radius, but slower if EI is high, consider variance
                self.trust_region_radius *= (self.radius_grow_base - self.ei_scaling * (avg_ei + ei_variance))
                self.trust_region_radius = min(self.trust_region_radius, 2.5)

            self.trust_region_radius = max(self.trust_region_radius, self.min_radius)

            # Re-center trust region if current center is far from best point
            distance = np.linalg.norm(self.trust_region_center - best_x)
            if distance > self.recentering_threshold * self.trust_region_radius:
                self.trust_region_center = best_x
            
            iteration += 1

        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<AETRDKEVBO>", line 136, in __call__
 136->                 self._optimize_kernel()
  File "<AETRDKEVBO>", line 63, in _optimize_kernel
  61 |         # Update length_scale for all RBF kernels in the ensemble
  62 |         for gp in self.gps:
  63->             if isinstance(gp.kernel_.k2, RBF):
  64 |                 gp.kernel_ = C(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.length_scale, length_scale_bounds=(1e-2, 10)) + gp.kernel_.k1
  65 |             elif isinstance(gp.kernel_, RBF):
AttributeError: 'GaussianProcessRegressor' object has no attribute 'kernel_'. Did you mean: 'kernel'?
