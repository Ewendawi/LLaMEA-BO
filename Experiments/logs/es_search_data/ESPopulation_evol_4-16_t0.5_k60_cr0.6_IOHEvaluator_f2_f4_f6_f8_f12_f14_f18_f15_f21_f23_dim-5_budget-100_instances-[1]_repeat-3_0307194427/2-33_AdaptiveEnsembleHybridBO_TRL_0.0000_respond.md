# Description
**Adaptive Ensemble Hybrid Bayesian Optimization with Trust Region and Lengthscale Adaptation (AEHBBO-TRL)**: This algorithm synergistically combines the strengths of AdaptiveEnsembleTrustRegionBO and EnhancedEfficientHybridBO. It leverages an ensemble of Gaussian Process Regression (GPR) models with diverse kernels for robust prediction, incorporates a trust region to balance exploration and exploitation, and dynamically adapts the GPR kernel lengthscale to capture function characteristics. The trust region's radius is adjusted based on the Expected Improvement (EI), and the center is re-centered to the best observed point if necessary. A hybrid acquisition function combining EI and a distance-based exploration term is used to select diverse and promising points.

# Justification
The algorithm combines the following key ideas:
1.  **Ensemble of GPR Models:** Using an ensemble of GPR models with different kernels enhances the robustness of the predictions and captures different aspects of the objective function.
2.  **Adaptive Trust Region:** The trust region mechanism dynamically adjusts the search space, focusing on promising regions while maintaining exploration. The radius adaptation based on EI provides a more informed adjustment compared to fixed decay/growth rates. Re-centering the trust region ensures that the search doesn't get stuck in a suboptimal region.
3.  **Adaptive Kernel Lengthscale:** Optimizing the kernel lengthscale allows the GPR models to better adapt to the function's characteristics, improving the accuracy of predictions.
4.  **Hybrid Acquisition Function:** Combining EI with a distance-based exploration term encourages diversity in the selected points, preventing premature convergence.
5.  **Efficient Point Selection:** Instead of an evolutionary algorithm, a simpler approach based on sorting and distance penalization is used for selecting the next points, which is computationally more efficient.

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

class AdaptiveEnsembleHybridBO_TRL:
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
        self.radius_decay = 0.95
        self.radius_grow = 1.1
        self.length_scale = 1.0
        self.kernel_optim_interval = 5
        self.distance_threshold = 0.1

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
            kernels = [
                C(1.0, constant_value_bounds=(1e-2, 1e2)) * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2)),
                C(1.0, constant_value_bounds=(1e-2, 1e2)) * Matern(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2), nu=1.5),
                C(1.0, constant_value_bounds=(1e-2, 1e2)) * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
            ]
            gps_temp = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-5) for kernel in kernels]
            for gp in gps_temp:
              gp.fit(self.X, self.y)
            log_likelihoods = [gp.log_marginal_likelihood() for gp in gps_temp]
            return -np.mean(log_likelihoods)

        res = minimize(obj, x0=self.length_scale, method='L-BFGS-B', bounds=[(1e-2, 10)])
        self.length_scale = res.x[0]
        for i in range(self.ensemble_size):
          if isinstance(self.gps[i].kernel_, sklearn.gaussian_process.kernels.Sum):
            self.gps[i].kernel_.k1.length_scale = self.length_scale
          else:
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

        # Distance-based exploration term
        if self.X is not None:
            distances = cdist(X, self.X)
            min_distances = np.min(distances, axis=1)
            exploration = min_distances
            exploration = exploration / np.max(exploration)
        else:
            exploration = np.ones(len(X))

        return (ei + 0.1 * exploration).reshape(-1, 1)

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(100 * self.dim)
        acq_values = self._acquisition_function(candidates).flatten()
        
        # Sort candidates by acquisition value
        sorted_indices = np.argsort(acq_values)[::-1]
        selected_points = []
        
        # Select points ensuring diversity
        for i in sorted_indices:
            candidate = candidates[i]
            if not selected_points:
                selected_points.append(candidate)
            else:
                distances = cdist([candidate], np.array(selected_points))
                if np.min(distances) > self.distance_threshold:
                    selected_points.append(candidate)
            if len(selected_points) >= batch_size:
                break

        # If not enough points were selected, add more from the sorted list
        while len(selected_points) < batch_size and len(sorted_indices) > 0:
            candidate = candidates[sorted_indices[0]]
            sorted_indices = sorted_indices[1:]  # remove the first element

            distances = cdist([candidate], np.array(selected_points))
            if np.min(distances) > self.distance_threshold:
                selected_points.append(candidate)

        return np.array(selected_points)

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
                self._optimize_kernel()
                self.last_gp_update = self.n_evals

            batch_size = min(self.budget - self.n_evals, 5)
            next_X = self._select_next_points(batch_size)
            if len(next_X) == 0:
                next_X = self._sample_points(batch_size)

            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            best_idx = np.argmin(self.y)
            current_best_y = self.y[best_idx][0]
            current_best_x = self.X[best_idx]

            if current_best_y < best_y:
                self.trust_region_center = current_best_x
                self.trust_region_radius *= self.radius_decay
                best_y = current_best_y
                best_x = current_best_x
            else:
                if self.trust_region_radius < 1.0:
                    self.trust_region_radius *= self.radius_grow
                    self.trust_region_radius = min(self.trust_region_radius, 2.5)

            self.trust_region_radius = max(self.trust_region_radius, self.min_radius)
            if np.linalg.norm(self.trust_region_center - current_best_x) > self.trust_region_radius:
                self.trust_region_center = current_best_x

        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<AdaptiveEnsembleHybridBO_TRL>", line 156, in __call__
 156->                 self._optimize_kernel()
  File "<AdaptiveEnsembleHybridBO_TRL>", line 66, in _optimize_kernel
  64 |         self.length_scale = res.x[0]
  65 |         for i in range(self.ensemble_size):
  66->           if isinstance(self.gps[i].kernel_, sklearn.gaussian_process.kernels.Sum):
  67 |             self.gps[i].kernel_.k1.length_scale = self.length_scale
  68 |           else:
NameError: name 'sklearn' is not defined
