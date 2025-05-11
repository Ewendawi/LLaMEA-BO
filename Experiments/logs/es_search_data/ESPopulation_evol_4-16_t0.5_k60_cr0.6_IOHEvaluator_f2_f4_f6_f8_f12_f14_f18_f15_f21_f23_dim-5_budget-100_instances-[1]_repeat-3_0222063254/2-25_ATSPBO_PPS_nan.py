from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.spatial.distance import cdist

class ATSPBO_PPS:
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
        self.min_radius = 1e-2

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

    def _acquisition_function(self, X, gp, patch_indices):
        # LCB within the patch
        X_patched = X[:, patch_indices]
        mu, sigma = gp.predict(X_patched, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        return mu - self.kappa * sigma

    def _select_next_point(self, gp, patch_indices):
        n_samples = 100 * self.dim
        samples = self._sample_points(n_samples, center=self.best_x, radius=self.trust_region_radius)
        acq_values = self._acquisition_function(samples, gp, patch_indices)
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

    def _select_patch_indices(self, gp, patch_size):
        # Probabilistic patch selection based on GP variance
        _, sigma = gp.predict(self.X[:, :], return_std=True)  # Project back to full dim for variance
        if sigma is None or len(sigma) == 0:
            return np.random.choice(self.dim, patch_size, replace=False)

        dimension_variances = np.var(self.X, axis=0)
        if np.sum(dimension_variances) == 0:
             probabilities = np.ones(self.dim) / self.dim
        else:
            probabilities = dimension_variances / np.sum(dimension_variances) # Use the total variance among dimensions

        patch_indices = np.random.choice(self.dim, patch_size, replace=False, p=probabilities)
        return patch_indices

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        self.best_x = self.X[np.argmin(self.y)].copy()

        while self.n_evals < self.budget:
            # Dynamic patch size
            remaining_evals = self.budget - self.n_evals
            patch_size = max(1, min(self.dim, int(self.dim * remaining_evals / self.budget) + 1))

            # Fit model
            gp = self._fit_model(self.X, self.y, np.arange(self.dim)) # train on all dimensions for patch selection and uncertainty estimation
            # Select patch indices probabilistically
            patch_indices = self._select_patch_indices(gp, patch_size)

            # Select next point
            next_x = self._select_next_point(gp, patch_indices)
            next_y = self._evaluate_points(func, next_x.reshape(1, -1))
            self._update_eval_points(next_x.reshape(1, -1), next_y)

            # Adjust trust region radius and kappa based on uncertainty
            _, sigma = gp.predict(self.X[:, patch_indices], return_std=True) # check uncertainty in the sampled stochastic patch
            avg_variance = np.mean(sigma**2) if sigma is not None else 0.0 # average of the variance

            if next_y < self.best_y:
                self.best_x = next_x.copy()
                self.trust_region_radius /= self.rho
                self.kappa *= self.rho
            else:
                self.trust_region_radius *= self.rho
                self.kappa /= self.rho

            # Adaptive adjustment based on variance within the patch
            if avg_variance > 0.1: # tune the threshold here
                self.kappa *= 1.1 # explore more
                self.rho *= 0.95  # Shrink slower
            else:
                self.kappa *= 0.9
                self.rho /= 0.95 # Shrink faster

            self.trust_region_radius = np.clip(self.trust_region_radius, self.min_radius, np.max(self.bounds[1] - self.bounds[0]) / 2)
            self.kappa = np.clip(self.kappa, 0.1, 10.0)

        return self.best_y, self.best_x
