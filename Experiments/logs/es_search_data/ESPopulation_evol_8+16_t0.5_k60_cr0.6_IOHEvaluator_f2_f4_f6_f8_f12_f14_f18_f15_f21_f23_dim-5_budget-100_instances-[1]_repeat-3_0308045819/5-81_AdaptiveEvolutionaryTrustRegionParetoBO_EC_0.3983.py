from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize, differential_evolution

class AdaptiveEvolutionaryTrustRegionParetoBO_EC:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(5 * dim, self.budget // 10)
        self.gp_ensemble = []
        self.n_ensemble = 5  # Number of GPs in the ensemble
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0 * np.sqrt(dim)
        self.radius_decay = 0.95
        self.radius_increase = 1.1
        self.min_radius = 0.1
        self.success_ratio = 0.0
        self.random_restart_prob = 0.05

        self.knn = NearestNeighbors(n_neighbors=5)
        self.context_penalty = 0.1
        self.initial_context_penalty = 0.1
        self.context_penalty_decay = 0.95
        self.context_penalty_increase = 1.05
        self.min_context_penalty = 0.01
        self.max_context_penalty = 1.0

        self.de_pop_size = 10
        self.lcb_kappa = 2.0
        self.kappa_decay = 0.99
        self.min_kappa = 0.1
        self.noise_estimate = 1e-4

    def _sample_points(self, n_points):
        if self.best_x is None or np.random.rand() < self.random_restart_prob:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            points = []
            while len(points) < n_points:
                sample = np.random.uniform(-1, 1, size=(self.dim,))
                sample = self.best_x + self.trust_region_radius * sample
                sample = np.clip(sample, self.bounds[0], self.bounds[1])
                points.append(sample)
            return np.array(points)

    def _fit_model(self, X, y):
        # Kernel optimization
        def neg_log_likelihood(theta):
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=theta, length_scale_bounds="fixed")
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
            gp.fit(X, y)
            return -gp.log_marginal_likelihood()

        # Initial guess for kernel parameters
        initial_length_scale = 1.0

        # Optimize kernel parameters using L-BFGS-B
        result = minimize(neg_log_likelihood, initial_length_scale, method='L-BFGS-B', bounds=[(1e-5, 10.0)])
        optimized_length_scale = result.x[0]

        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=optimized_length_scale, length_scale_bounds="fixed")

        # Train the ensemble
        self.gp_ensemble = []
        for _ in range(self.n_ensemble):
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
            gp.fit(X, y)
            self.gp_ensemble.append(gp)

        self.knn.fit(X)
        return self.gp_ensemble

    def _lower_confidence_bound(self, X):
        mu_ensemble = np.zeros((X.shape[0], self.n_ensemble))
        sigma_ensemble = np.zeros((X.shape[0], self.n_ensemble))

        for i, gp in enumerate(self.gp_ensemble):
            mu, sigma = gp.predict(X, return_std=True)
            mu_ensemble[:, i] = mu
            sigma_ensemble[:, i] = sigma

        mu = np.mean(mu_ensemble, axis=1)
        sigma = np.mean(sigma_ensemble, axis=1) # Average standard deviation
        lcb = mu - self.lcb_kappa * sigma
        return lcb.reshape(-1, 1)

    def _diversity_metric(self, X):
        if self.X is None:
            return np.ones((len(X), 1))
        else:
            distances = np.min(cdist(X, self.X), axis=1)
            return distances.reshape(-1, 1)

    def _context_penalty_metric(self, X):
        if self.X is None:
            return np.zeros((len(X), 1))
        else:
            distances, _ = self.knn.kneighbors(X)
            context_penalty = np.mean(distances, axis=1).reshape(-1, 1)
            return context_penalty

    def _select_next_points(self, batch_size):
        if self.best_x is None:
            return self._sample_points(batch_size)

        def de_objective(x):
            X_candidate = x.reshape(1, -1)
            lcb = self._lower_confidence_bound(X_candidate)
            diversity = self._diversity_metric(X_candidate)
            context_penalty = self._context_penalty_metric(X_candidate) * self.context_penalty

            lcb_normalized = (lcb - np.min(lcb)) / (np.max(lcb) - np.min(lcb)) if np.max(lcb) != np.min(lcb) else np.zeros_like(lcb)
            diversity_normalized = (diversity - np.min(diversity)) / (np.max(diversity) - np.min(diversity)) if np.max(diversity) != np.min(diversity) else np.zeros_like(diversity)
            context_penalty_normalized = (context_penalty - np.min(context_penalty)) / (np.max(context_penalty) - np.min(context_penalty)) if np.max(context_penalty) != np.min(context_penalty) else np.zeros_like(context_penalty)

            F = np.hstack([lcb_normalized, diversity_normalized, context_penalty_normalized])

            # Pareto front calculation (minimization of all objectives)
            is_efficient = np.ones(F.shape[0], dtype=bool)
            for i, c in enumerate(F):
                if is_efficient[i]:
                    is_efficient[is_efficient] = np.any(F[is_efficient] <= c, axis=1)  # Changed to <= for minimization
                    is_efficient[i] = True

            # Return the negative LCB value for DE (DE is a minimizer)
            if np.any(is_efficient):
                lcb_pareto = self._lower_confidence_bound(X_candidate[is_efficient])
                return lcb_pareto[0, 0]  # Return LCB of the first Pareto point
            else:
                return lcb[0, 0] #Return LCB if no pareto point is found

        de_bounds = [(max(self.bounds[0][i], self.best_x[i] - self.trust_region_radius),
                      min(self.bounds[1][i], self.best_x[i] + self.trust_region_radius)) for i in range(self.dim)]

        remaining_evals = self.budget - self.n_evals
        maxiter = max(1, int(remaining_evals / (self.de_pop_size * self.dim * 2)))
        maxiter = min(maxiter, 100)

        result = differential_evolution(de_objective, de_bounds, popsize=self.de_pop_size, maxiter=maxiter, tol=0.01, disp=False)
        return result.x.reshape(1, -1)

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
            self.success_ratio = 1.0
        else:
            self.success_ratio *= 0.75

        self.noise_estimate = np.var(self.y)

    def _adjust_trust_region(self):
        if self.success_ratio > 0.5:
            self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)

    def _adjust_context_penalty(self):
        if self.success_ratio > 0.5:
            self.context_penalty = min(self.context_penalty * self.context_penalty_increase, self.max_context_penalty)
        else:
            self.context_penalty = max(self.context_penalty * self.context_penalty_decay, self.min_context_penalty)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        batch_size = min(1, self.dim)
        while self.n_evals < self.budget:
            self._fit_model(self.X, self.y)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self._adjust_trust_region()
            self._adjust_context_penalty()

            self.lcb_kappa = max(self.lcb_kappa * self.kappa_decay, self.min_kappa)
            self.min_kappa = 0.1 + np.sqrt(self.noise_estimate)

        return self.best_y, self.best_x
