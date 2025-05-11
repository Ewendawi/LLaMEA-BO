from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
import warnings
from sklearn.neighbors import NearestNeighbors

class DGAETBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.gradient_weight = 0.01
        self.diversity_weight = 0.01
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 1.5
        self.model_agreement_threshold = 0.7
        self.kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        self.model = None
        self.batch_size = 5
        self.min_batch_size = 3
        self.max_batch_size = 10
        self.step_size = 1e-6

    def _sample_points(self, n_points):
        sampler = qmc.Halton(d=self.dim, seed=42)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        try:
            model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5, alpha=1e-5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)
            self.kernel = model.kernel_
            return model
        except Exception as e:
            print(f"GP fitting failed: {e}. Returning None.")
            return None

    def _acquisition_function(self, X, model):
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if self.best_y is None:
            ei = np.zeros_like(mu)
        else:
            imp = self.best_y - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0

        # Gradient-based exploration
        if self.X is not None:
            dmu_dx = self._predict_gradient(X, model)
            gradient_norm = np.linalg.norm(dmu_dx, axis=1, keepdims=True)
            ei += self.gradient_weight * gradient_norm

            # Diversity enhancement
            diversity = self._calculate_diversity(X)
            diversity_weight = self.diversity_weight * (1 + sigma.mean()) * (1 - self.trust_region_radius / 5.0)
            ei += diversity_weight * diversity

        return ei

    def _predict_gradient(self, X, model):
        X = np.atleast_2d(X)
        dmu_dx = np.zeros((X.shape[0], self.dim))
        for i in range(self.dim):
            X_plus = X.copy()
            X_plus[:, i] += self.step_size
            X_minus = X.copy()
            X_minus[:, i] -= self.step_size
            dmu_dx[:, i] = (model.predict(X_plus) - model.predict(X_minus)) / (2 * self.step_size)
        return dmu_dx

    def _calculate_diversity(self, X):
        if self.X is None or len(self.X) < 2:
            return np.zeros(X.shape[0])

        knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        knn.fit(self.X)
        distances, _ = knn.kneighbors(X)
        return distances.flatten()

    def _select_next_points(self, batch_size, trust_region_center):
        sobol_engine = qmc.Sobol(d=self.dim, seed=42)
        samples = sobol_engine.random(n=100 * batch_size)
        scaled_samples = trust_region_center + self.trust_region_radius * (2 * samples - 1)
        scaled_samples = np.clip(scaled_samples, self.bounds[0], self.bounds[1])

        if self.model is None:
            return scaled_samples[:batch_size]

        acquisition_values = self._acquisition_function(scaled_samples, self.model)
        indices = np.argsort(-acquisition_values.flatten())[:batch_size]
        selected_points = scaled_samples[indices]

        return selected_points

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

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        trust_region_center = self.X[np.argmin(self.y)]
        
        while self.n_evals < self.budget:
            self.model = self._fit_model(self.X, self.y)
            if self.model is None:
                print("GP model fitting failed. Returning current best.")
                return self.best_y, self.best_x

            next_X = self._select_next_points(self.batch_size, trust_region_center)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            predicted_y = self.model.predict(next_X)
            agreement = np.corrcoef(next_y.flatten(), predicted_y.flatten())[0, 1]

            # Trust region update
            delta_y = self.best_y - np.min(next_y)
            if np.isnan(agreement) or agreement < self.model_agreement_threshold or delta_y < 1e-3:
                self.trust_region_radius *= self.trust_region_shrink_factor
                self.batch_size = max(self.min_batch_size, int(self.batch_size * self.trust_region_shrink_factor))
            else:
                self.trust_region_radius *= self.trust_region_expand_factor
                self.trust_region_radius = min(self.trust_region_radius, 5.0)
                self.batch_size = min(self.max_batch_size, int(self.batch_size * self.trust_region_expand_factor))
            
            trust_region_center = self.X[np.argmin(self.y)]

        return self.best_y, self.best_x
