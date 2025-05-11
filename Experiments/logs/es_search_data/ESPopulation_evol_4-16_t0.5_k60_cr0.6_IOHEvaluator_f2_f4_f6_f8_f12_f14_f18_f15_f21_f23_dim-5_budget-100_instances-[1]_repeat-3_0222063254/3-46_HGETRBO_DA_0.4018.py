from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import approx_fprime
from scipy.special import expit  # sigmoid function


class HGETRBO_DA:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(10 * dim, self.budget // 5)
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.5
        self.rho = 0.95
        self.kappa = 2.0 + np.log(dim)
        self.success_history = []
        self.success_window = 5
        self.rng = np.random.RandomState(42)
        self.global_sampling_prob = 0.05
        self.prev_best_y = float('inf')
        self.noise_level = 1e-6
        self.success_count = 0
        self.failure_count = 0
        self.min_evals_for_adjust = 5 * dim
        self.kappa_success_failure_weight = 0.5
        self.kappa_weight_scaling = 5.0  # Controls the steepness of the sigmoid
        self.kappa_weight_offset = 0.5   # Shifts the sigmoid along the x-axis

    def _sample_points(self, n_points, center=None, radius=None, global_sample=False):
        if global_sample:
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            points = sampler.random(n=n_points)
            points = qmc.scale(points, self.bounds[0], self.bounds[1])
            return points
        else:
            if center is None:
                center = (self.bounds[1] + self.bounds[0]) / 2
            if radius is None:
                radius = np.max(self.bounds[1] - self.bounds[0]) / 2

            sampler = qmc.Sobol(d=self.dim, scramble=True)
            points = sampler.random(n=n_points)
            points = qmc.scale(points, -1, 1)

            lengths = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / lengths * self.rng.uniform(0, 1, size=lengths.shape) ** (1 / self.dim)

            points = points * radius + center
            points = np.clip(points, self.bounds[0], self.bounds[1])
            return points

    def _fit_model(self, X, y):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=self.noise_level, random_state=42)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp):
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        return mu - self.kappa * sigma

    def _select_next_point(self, gp, func):
        if self.rng.rand() < self.global_sampling_prob:
            # Global sampling
            n_samples = 100 * self.dim
            samples = self._sample_points(n_samples, global_sample=True)
            acq_values = self._acquisition_function(samples, gp)
            best_index = np.argmin(acq_values)
            return samples[best_index]
        else:
            # Trust region sampling
            n_samples = 100 * self.dim
            samples = self._sample_points(n_samples, center=self.best_x, radius=self.trust_region_radius)
            acq_values = self._acquisition_function(samples, gp)
            best_index = np.argmin(acq_values)
            return samples[best_index]

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

        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx].copy()

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        self.best_x = self.X[np.argmin(self.y)].copy()
        self.prev_best_y = self.best_y

        while self.n_evals < self.budget:
            # Optimization
            gp = self._fit_model(self.X, self.y)
            next_x = self._select_next_point(gp, func)
            next_y = self._evaluate_points(func, next_x.reshape(1, -1))
            self._update_eval_points(next_x.reshape(1, -1), next_y)

            # Update success history
            success = next_y < self.best_y
            self.success_history.append(success[0])
            if len(self.success_history) > self.success_window:
                self.success_history = self.success_history[-self.success_window:]

            if self.n_evals > self.n_init + self.min_evals_for_adjust:
                # Radius Adjustment (DKRA component)
                if next_y < self.best_y:
                    self.success_count += 1
                    self.failure_count = 0
                    success_ratio = self.success_count / (self.success_count + self.failure_count + 1e-9)
                    self.trust_region_radius /= (0.9 + 0.09 * success_ratio)  # Expand faster with higher success
                else:
                    self.failure_count += 1
                    self.success_count = 0
                    failure_ratio = self.failure_count / (self.success_count + self.failure_count + 1e-9)
                    self.trust_region_radius *= (0.9 + 0.09 * failure_ratio)  # Shrink faster with higher failure
                self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)

                # Adapt kappa based on GP variance, gradient, and success/failure
                mu, sigma = gp.predict(self._sample_points(100, center=self.best_x, radius=self.trust_region_radius), return_std=True)
                avg_sigma = np.mean(sigma)
                gradient = approx_fprime(self.best_x, lambda x: gp.predict(x.reshape(1,-1))[0], epsilon=1e-6)
                gradient_norm = np.linalg.norm(gradient)

                kappa_gp_component = 1.0 + np.log(1 + avg_sigma + gradient_norm)

                success_ratio = self.success_count / (self.success_count + self.failure_count + 1e-9)
                kappa_success_failure_component = (0.9 + 0.09 * success_ratio)

                # Dynamic weighting of kappa components using a sigmoid
                weight = expit(self.kappa_weight_scaling * (success_ratio - self.kappa_weight_offset))
                self.kappa = (weight * kappa_success_failure_component +
                               (1 - weight) * kappa_gp_component)
                self.kappa = np.clip(self.kappa, 0.1, 10.0)

            # Adapt rho based on success rate and relative improvement
            success_rate = np.mean(self.success_history) if self.success_history else 0.5
            relative_improvement = abs(self.best_y - self.prev_best_y) / (abs(self.prev_best_y) + 1e-9)
            self.rho = np.clip(0.9 + (0.5 - success_rate) / 5 + (0.1 - relative_improvement), 0.7, 0.99)

            # Adjust trust region radius
            if success:
                self.trust_region_radius /= self.rho
                self.best_x = next_x.copy()
            else:
                self.trust_region_radius *= self.rho

            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)

            # Estimate noise level
            if len(self.y) > 5:
                self.noise_level = np.std(self.y[-5:])

            self.prev_best_y = self.best_y

        return self.best_y, self.best_x
