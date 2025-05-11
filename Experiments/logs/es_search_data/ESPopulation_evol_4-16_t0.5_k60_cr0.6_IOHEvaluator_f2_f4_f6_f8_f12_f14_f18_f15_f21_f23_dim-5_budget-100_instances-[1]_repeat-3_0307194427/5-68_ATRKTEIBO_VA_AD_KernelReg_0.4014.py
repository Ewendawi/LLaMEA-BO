from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.optimize import minimize, Bounds
from scipy.spatial.distance import cdist, pdist, squareform

class ATRKTEIBO_VA_AD_KernelReg:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * (dim + 1)
        self.trust_region_center = np.zeros(dim)
        self.trust_region_radius = 2.5
        self.min_radius = 0.1
        self.radius_decay_base = 0.95
        self.radius_grow_base = 1.1
        self.gp = None
        self.ei_scaling_init = 0.2  # Initial EI scaling
        self.ei_scaling_final = 0.01 # Final EI scaling
        self.recentering_threshold = 0.5
        self.length_scale = 1.0
        self.kernel_optim_interval = 5
        self.diversity_threshold_base = 0.05 # Base diversity threshold
        self.initial_diversity_threshold = 0.1
        self.diversity_threshold = 0.1
        self.diversity_adaptation_rate = 0.05
        self.kernel_regularization_strength = 0.01 # Initial kernel regularization strength
        self.kernel_regularization_decay = 0.995  # Decay rate for kernel regularization

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        sample = sampler.random(n=n_points)
        scaled_sample = qmc.scale(sample, -1.0, 1.0)
        points = self.trust_region_center + scaled_sample * self.trust_region_radius
        points = np.clip(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * \
                 RBF(length_scale=self.length_scale, length_scale_bounds=(1e-3, 1e3)) + \
                 WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-7, 1e-3))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=self.kernel_regularization_strength)
        self.gp.fit(X, y)
        return self.gp

    def _optimize_kernel(self):
        # Initialize length_scale with median distance
        distances = pdist(self.X)
        if len(distances) > 0:
            self.length_scale = np.median(distances)
        else:
            self.length_scale = 1.0

        def obj(length_scale):
            kernel = ConstantKernel(constant_value=1.0, constant_value_bounds="fixed") * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 10))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=self.kernel_regularization_strength)
            gp.fit(self.X, self.y)
            return -gp.log_marginal_likelihood()

        bounds = Bounds(1e-2, 10)
        res = minimize(obj, x0=self.length_scale, method='SLSQP', bounds=bounds)
        self.length_scale = res.x[0]

    def _acquisition_function(self, X):
        if self.gp is None or self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)
        y_best = np.min(self.y)
        gamma = (y_best - mu) / sigma
        ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(100 * self.dim)
        ei = self._acquisition_function(candidates)

        # Enhanced diversity mechanism using EI
        if self.X is not None:
            distances = cdist(candidates, self.X)
            min_distances = np.min(distances, axis=1)
            # Filter candidates based on both distance and EI
            diversity_threshold = self.diversity_threshold
            diversity_mask = min_distances > diversity_threshold
            ei_threshold = np.quantile(ei, 0.25)  # Consider only top 75% EI values
            ei_mask = ei.flatten() >= ei_threshold
            selected_candidates = candidates[diversity_mask & ei_mask]
            selected_ei = ei[diversity_mask & ei_mask]

            if len(selected_candidates) < batch_size:
                # If not enough diverse candidates, add some more based on EI only
                remaining_needed = batch_size - len(selected_candidates)
                additional_indices = np.argsort(ei.flatten())[::-1][:remaining_needed]
                additional_candidates = candidates[additional_indices]
                selected_candidates = np.concatenate([selected_candidates, additional_candidates], axis=0)
                selected_ei = np.concatenate([selected_ei, ei[additional_indices]], axis=0)
            candidates = selected_candidates
            ei = selected_ei

        selected_indices = np.argsort(ei.flatten())[-batch_size:]
        selected_points = candidates[selected_indices]
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

            # Fit GP model
            gp = self._fit_model(self.X, self.y)

            # Dynamic batch size
            mu, sigma = gp.predict(self.X, return_std=True)
            ei_values = self._acquisition_function(self.X)
            ei_variance = np.var(ei_values)
            avg_sigma = np.mean(sigma)
            batch_size = min(self.budget - self.n_evals, int(np.ceil(5 * (avg_sigma / np.std(self.bounds))))) if np.std(self.bounds) > 0 else min(self.budget - self.n_evals, 5)

            # Adaptive EI scaling
            ei_scaling = self.ei_scaling_init + (self.ei_scaling_final - self.ei_scaling_init) * (iteration / self.budget)
            batch_size = int(batch_size * (1 + ei_scaling * ei_variance))
            batch_size = max(1, min(batch_size, self.budget - self.n_evals))

            # Select next points
            next_X = self._select_next_points(batch_size)

            # Evaluate points
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Update best solution
            best_idx = np.argmin(self.y)
            current_best_y = self.y[best_idx][0]
            current_best_x = self.X[best_idx]

            # Calculate average EI and variance of evaluated points
            ei_values = self._acquisition_function(next_X)
            avg_ei = np.mean(ei_values)
            ei_variance = np.var(ei_values)

            # Adjust trust region radius based on EI and EI variance
            # Weighted average of EI and EI variance
            ei_weight = 0.7
            variance_weight = 0.3
            if current_best_y < best_y:
                # Improvement: shrink radius based on EI, slower if EI variance is high
                self.trust_region_radius *= (self.radius_decay_base + ei_scaling * avg_ei * ei_weight) * (1 - ei_scaling * ei_variance * variance_weight)
                self.trust_region_center = current_best_x
                best_y = current_best_y
                best_x = current_best_x
            else:
                # No improvement: expand radius, but slower if EI is high, faster if EI variance is high
                self.trust_region_radius *= (self.radius_grow_base - ei_scaling * avg_ei * ei_weight) * (1 + ei_scaling * ei_variance * variance_weight)
                self.trust_region_radius = min(self.trust_region_radius, 2.5)  # Limit to initial radius

            self.trust_region_radius = max(self.trust_region_radius, self.min_radius)

            # Re-center trust region if current center is far from best point
            distance = np.linalg.norm(self.trust_region_center - best_x)
            if distance > self.recentering_threshold * self.trust_region_radius:
                self.trust_region_center = best_x

            # Adapt diversity threshold
            self.diversity_threshold = self.initial_diversity_threshold * (self.trust_region_radius / 2.5) * (1 - iteration / self.budget)
            self.diversity_threshold = np.clip(self.diversity_threshold, 0.01, 0.5)

            # Adapt kernel regularization strength
            self.kernel_regularization_strength *= self.kernel_regularization_decay * (1 + ei_scaling * ei_variance)
            self.kernel_regularization_strength = np.clip(self.kernel_regularization_strength, 1e-5, 0.1)

            iteration += 1

        return best_y, best_x
