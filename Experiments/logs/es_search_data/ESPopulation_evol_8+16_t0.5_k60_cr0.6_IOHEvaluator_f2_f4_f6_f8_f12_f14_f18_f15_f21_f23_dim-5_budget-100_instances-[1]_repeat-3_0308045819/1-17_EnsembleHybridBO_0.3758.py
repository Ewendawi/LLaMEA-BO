from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

class EnsembleHybridBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(10 * dim, self.budget // 4)
        self.gp_ensemble = []
        self.ensemble_weights = []
        self.best_x = None
        self.best_y = float('inf')
        self.n_ensemble = 3
        self.trust_region_radius = 2.0  # Initial trust region radius

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        if not self.gp_ensemble:
            kernels = [
                ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed"),
                ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds="fixed", nu=0.5),
                ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds="fixed", nu=1.5)
            ]
            for i in range(self.n_ensemble):
                gp = GaussianProcessRegressor(kernel=kernels[i % len(kernels)], n_restarts_optimizer=0, alpha=1e-6)
                gp.fit(X_train, y_train)
                self.gp_ensemble.append(gp)
                self.ensemble_weights.append(1.0 / self.n_ensemble)
        else:
            for gp in self.gp_ensemble:
                gp.fit(X_train, y_train)

        val_errors = []
        for gp in self.gp_ensemble:
            y_pred, _ = gp.predict(X_val, return_std=True)
            error = np.mean((y_pred - y_val.flatten()) ** 2)
            val_errors.append(error)

        val_errors = np.array(val_errors)
        weights = np.exp(-val_errors) / np.sum(np.exp(-val_errors))
        self.ensemble_weights = weights

    def _acquisition_function(self, X):
        if not self.gp_ensemble:
            return np.random.normal(size=(len(X), 1))
        else:
            y_samples_ensemble = np.zeros((len(X), self.n_ensemble))
            for i, gp in enumerate(self.gp_ensemble):
                y_samples_ensemble[:, i] = gp.sample_y(X, n_samples=1).flatten()
            
            # Weighted average of Thompson samples from each GP
            acquisition_values = np.sum(self.ensemble_weights[i] * y_samples_ensemble[:, i] for i in range(self.n_ensemble)).reshape(-1, 1)
            return acquisition_values

    def _select_next_points(self, batch_size):
        n_candidates = min(100 * self.dim, self.budget - self.n_evals)
        
        # Sample candidates within the trust region
        if self.best_x is not None:
            candidates = self._sample_points(n_candidates)
            # Filter candidates within the trust region
            distances = np.linalg.norm(candidates - self.best_x, axis=1)
            candidates = candidates[distances <= self.trust_region_radius]
        else:
            candidates = self._sample_points(n_candidates)
        
        if len(candidates) == 0:
            candidates = self._sample_points(n_candidates)

        acquisition_values = self._acquisition_function(candidates)

        n_clusters = min(batch_size, len(candidates))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(candidates)
        cluster_ids = kmeans.labels_

        next_points = []
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_ids == i)[0]
            best_index = cluster_indices[np.argmin(acquisition_values[cluster_indices])]
            next_points.append(candidates[best_index])

        return np.array(next_points)

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
            # Adjust trust region radius based on success
            self.trust_region_radius *= 1.1
        else:
            # Reduce trust region radius if no improvement
            self.trust_region_radius *= 0.9
        self.trust_region_radius = np.clip(self.trust_region_radius, 0.1, 5.0)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        batch_size = min(10, self.dim)
        while self.n_evals < self.budget:
            self._fit_model(self.X, self.y)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
