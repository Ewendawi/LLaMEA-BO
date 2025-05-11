from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.metrics import pairwise_distances
import warnings

class GADETREBOPareto:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.initial_gradient_weight = 0.01
        self.initial_diversity_weight = 0.1
        self.initial_exploration_weight = 0.01
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 2.0
        self.model_agreement_threshold = 0.75
        self.acquisition_functions = ['ei', 'pi', 'ucb']
        self.ucb_kappa = 2.0
        self.ucb_kappa_max = 5.0
        self.ucb_kappa_min = 1.0
        self.reg_weight = 0.1
        self.exploration_factor = 0.01
        self.ei_weight = 0.5
        self.pi_weight = 0.25
        self.ucb_weight = 0.25
        self.kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        self.model = None

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, seed=42)
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

    def _acquisition_function(self, X, model, iteration, acq_type='ei'):
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if self.best_y is None:
            ei = np.zeros_like(mu)
        else:
            imp = self.best_y - mu
            Z = imp / sigma

            if acq_type == 'ei':
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma <= 1e-6] = 0.0
            elif acq_type == 'pi':
                ei = norm.cdf(Z)
            elif acq_type == 'ucb':
                ei = mu + self.ucb_kappa * sigma
            else:
                raise ValueError("Invalid acquisition function type.")

        reg_weight = self.reg_weight * (1 - (iteration / self.budget))
        regularization_term = -reg_weight * np.linalg.norm(mu / (sigma + 1e-6), axis=1, keepdims=True)**2
        ei = ei + regularization_term + self.exploration_factor * sigma

        if acq_type == 'ei' and self.X is not None:
            dmu_dx = self._predict_gradient(X, model)
            gradient_norm = np.linalg.norm(dmu_dx, axis=1, keepdims=True)
            ei = ei + self.initial_gradient_weight * gradient_norm

        if acq_type == 'ei' and self.X is not None:
            distances = pairwise_distances(X, self.X)
            min_distances = np.min(distances, axis=1).reshape(-1, 1)
            ei = ei + self.initial_diversity_weight * min_distances

        return ei

    def _predict_gradient(self, X, model):
        X = np.atleast_2d(X)
        dmu_dx = np.zeros((X.shape[0], self.dim))
        delta = 1e-6
        for i in range(self.dim):
            X_plus = X.copy()
            X_plus[:, i] += delta
            X_minus = X.copy()
            X_minus[:, i] -= delta
            dmu_dx[:, i] = (model.predict(X_plus) - model.predict(X_minus)) / (2 * delta)
        return dmu_dx

    def _select_next_points(self, batch_size, model, iteration, trust_region_center):
        sobol_engine = qmc.Sobol(d=self.dim, seed=42)
        samples = sobol_engine.random(n=100 * batch_size)
        scaled_samples = trust_region_center + self.trust_region_radius * (2 * samples - 1)
        scaled_samples = np.clip(scaled_samples, self.bounds[0], self.bounds[1])

        acquisition_values = np.zeros((scaled_samples.shape[0], len(self.acquisition_functions)))

        for i, acq_type in enumerate(self.acquisition_functions):
            acquisition_values[:, i] = self._acquisition_function(scaled_samples, model, iteration, acq_type).flatten()

        if self.trust_region_radius < 1.0 and self.model_agreement > self.model_agreement_threshold:
            self.ei_weight = 0.7
            self.pi_weight = 0.15
            self.ucb_weight = 0.15
        else:
            self.ei_weight = 0.5
            self.pi_weight = 0.25
            self.ucb_weight = 0.25

        weighted_acquisition_values = (
            self.ei_weight * acquisition_values[:, 0] +
            self.pi_weight * acquisition_values[:, 1] +
            self.ucb_weight * acquisition_values[:, 2]
        )

        indices = np.argsort(-weighted_acquisition_values)[:batch_size]
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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        trust_region_center = self.X[np.argmin(self.y)]
        iteration = self.n_init
        while self.n_evals < self.budget:
            self.model = self._fit_model(self.X, self.y)
            if self.model is None:
                print("GP model fitting failed. Returning current best.")
                return self.best_y, self.best_x

            # Dynamic batch size
            if self.trust_region_radius > 2.0:
                batch_size = 5
            elif self.trust_region_radius > 1.0:
                batch_size = 3
            else:
                batch_size = 1

            next_X = self._select_next_points(batch_size, self.model, iteration, trust_region_center)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            predicted_y = self.model.predict(next_X)
            self.model_agreement = np.corrcoef(next_y.flatten(), predicted_y.flatten())[0, 1]

            if np.isnan(self.model_agreement) or self.model_agreement < self.model_agreement_threshold:
                self.trust_region_radius *= self.trust_region_shrink_factor
                self.ucb_kappa = min(self.ucb_kappa * 1.1, self.ucb_kappa_max)
            else:
                self.trust_region_radius *= self.trust_region_expand_factor
                self.trust_region_radius = min(self.trust_region_radius, 5.0)
                self.ucb_kappa = max(self.ucb_kappa * 0.9, self.ucb_kappa_min)

            trust_region_center = self.X[np.argmin(self.y)]
            iteration += batch_size

        return self.best_y, self.best_x
