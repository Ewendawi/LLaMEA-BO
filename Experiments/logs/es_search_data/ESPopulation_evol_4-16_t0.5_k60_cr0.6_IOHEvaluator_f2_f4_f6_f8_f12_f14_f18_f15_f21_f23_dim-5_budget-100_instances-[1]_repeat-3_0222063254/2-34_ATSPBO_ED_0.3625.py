from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from collections import deque

class ATSPBO_ED:
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
        self.trust_region_radius = 2.5  # Initial trust region radius
        self.rho = 0.95  # Shrinking factor
        self.kappa = 2.0  # Exploration-exploitation trade-off for LCB
        self.success_history = deque(maxlen=10)  # Track recent success
        self.min_evals_for_rho = 5 # Minimum evals before adapting rho
        self.success_threshold = 0.7

    def _sample_points(self, n_points, center=None, radius=None):
        if center is None:
            center = (self.bounds[1] + self.bounds[0]) / 2
        if radius is None:
            radius = np.max(self.bounds[1] - self.bounds[0]) / 2

        sampler = qmc.Sobol(d=self.dim, scramble=True)
        points = sampler.random(n=n_points)
        points = qmc.scale(points, -1, 1)  # Scale to [-1, 1]

        lengths = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / lengths * np.random.uniform(0, 1, size=lengths.shape) ** (1/self.dim)

        points = points * radius + center
        points = np.clip(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y, patch_indices):
        # Fit the model on the stochastic patch
        X_patched = X[:, patch_indices]
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X_patched, y)
        return gp

    def _acquisition_function(self, X, gps, patch_indices_list):
        # Average LCB within the patch over the ensemble
        acq_values = np.zeros((X.shape[0], 1))
        for gp, patch_indices in zip(gps, patch_indices_list):
            X_patched = X[:, patch_indices]
            mu, sigma = gp.predict(X_patched, return_std=True)
            mu = mu.reshape(-1, 1)
            sigma = sigma.reshape(-1, 1)
            acq_values += (mu - self.kappa * sigma) / len(gps)
        return acq_values

    def _select_next_point(self, gps, patch_indices_list):
        n_samples = 100 * self.dim
        samples = self._sample_points(n_samples, center=self.best_x, radius=self.trust_region_radius)
        acq_values = self._acquisition_function(samples, gps, patch_indices_list)
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
            self.best_x = self.X[idx]
            self.success_history.append(True)
        else:
            self.success_history.append(False)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        self.best_x = self.X[np.argmin(self.y)].copy()

        while self.n_evals < self.budget:
            # Dynamic patch size
            remaining_evals = self.budget - self.n_evals
            success_rate = np.mean(self.success_history) if self.success_history else 0.5
            patch_size = max(1, min(self.dim, int(self.dim * (remaining_evals / self.budget + success_rate)/ 2) + 1))

            # Ensemble of Patches
            n_ensemble = 3 # Number of GPs in ensemble
            gps = []
            patch_indices_list = []
            for _ in range(n_ensemble):
                patch_indices = np.random.choice(self.dim, patch_size, replace=False)
                gp = self._fit_model(self.X, self.y, patch_indices)
                gps.append(gp)
                patch_indices_list.append(patch_indices)

            # Select next point
            next_x = self._select_next_point(gps, patch_indices_list)
            next_y = self._evaluate_points(func, next_x.reshape(1, -1))
            self._update_eval_points(next_x.reshape(1, -1), next_y)

            # Adjust trust region radius and kappa
            if len(self.success_history) >= self.min_evals_for_rho:
                success_rate = np.mean(self.success_history)
                if success_rate > self.success_threshold:
                    self.trust_region_radius /= self.rho  # Expand
                    self.kappa *= self.rho
                else:
                    self.trust_region_radius *= self.rho  # Shrink
                    self.kappa /= self.rho  # More exploration

            if next_y < self.best_y:
                self.best_x = next_x # Move Trust Region center

            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)
            self.kappa = np.clip(self.kappa, 0.1, 10.0)

        return self.best_y, self.best_x
