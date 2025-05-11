from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist

class ParetoEvolutionaryLCBTrustRegionBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(5 * dim, self.budget // 10)
        self.gp = None
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0
        self.radius_decay = 0.95
        self.radius_increase = 1.1
        self.min_radius = 0.1
        self.success_ratio = 0.0
        self.random_restart_prob = 0.05
        self.de_pop_size = 10
        self.lcb_kappa = 2.0
        self.kappa_decay = 0.99
        self.min_kappa = 0.1
        self.diversity_weight = 0.1 # Initial diversity weight
        self.diversity_weight_decay = 0.95
        self.min_diversity_weight = 0.01

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
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
        self.gp.fit(X, y)
        return self.gp

    def _lower_confidence_bound(self, X, kappa=2.0):
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            mu, sigma = self.gp.predict(X, return_std=True)
            lcb = mu - kappa * sigma
            return lcb.reshape(-1, 1)

    def _diversity_metric(self, X):
        if self.X is None:
            return np.ones((len(X), 1))
        else:
            distances = np.min(cdist(X, self.X), axis=1)
            return distances.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        if self.best_x is None:
            return self._sample_points(batch_size)

        def de_objective(x):
            x = x.reshape(1, -1)
            lcb = self._lower_confidence_bound(x)
            diversity = self._diversity_metric(x)

            lcb_normalized = (lcb - np.min(lcb)) / (np.max(lcb) - np.min(lcb)) if np.max(lcb) != np.min(lcb) else np.zeros_like(lcb)
            diversity_normalized = (diversity - np.min(diversity)) / (np.max(diversity) - np.min(diversity)) if np.max(diversity) != np.min(diversity) else np.zeros_like(diversity)

            # Dynamic weighting of LCB and diversity
            weight_lcb = 1.0 - self.diversity_weight
            weight_diversity = self.diversity_weight

            F = np.hstack([weight_lcb * lcb_normalized, weight_diversity * diversity_normalized])

            is_efficient = np.ones(F.shape[0], dtype=bool)
            for i, c in enumerate(F):
                if is_efficient[i]:
                    is_efficient[is_efficient] = np.any(F[is_efficient] >= c, axis=1)
                    is_efficient[i] = True

            pareto_front_indices = np.where(is_efficient)[0]

            if len(pareto_front_indices) > 0:
                pareto_front = x[pareto_front_indices]
                lcb_pareto = self._lower_confidence_bound(pareto_front)
                next_point = pareto_front[np.argmin(lcb_pareto)].reshape(1, -1)
            else:
                next_point = self._sample_points(1)

            return next_point[0,0]

        de_bounds = [(max(self.bounds[0][i], self.best_x[i] - self.trust_region_radius),
                      min(self.bounds[1][i], self.best_x[i] + self.trust_region_radius)) for i in range(self.dim)]

        remaining_evals = self.budget - self.n_evals
        maxiter = max(1, int(remaining_evals / (self.de_pop_size * self.dim * 2)))
        maxiter = min(maxiter, 100)

        next_points = []
        for _ in range(batch_size):
            result = differential_evolution(de_objective, de_bounds, popsize=self.de_pop_size, maxiter=maxiter, tol=0.01, disp=False)
            next_points.append(result.x)
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
            self.success_ratio = 1.0
        else:
            self.success_ratio *= 0.75

    def _adjust_trust_region(self):
        if self.success_ratio > 0.5:
            self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)

    def _adjust_pareto_weights(self):
         self.diversity_weight = max(self.diversity_weight * self.diversity_weight_decay, self.min_diversity_weight)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        batch_size = min(4, self.dim)
        while self.n_evals < self.budget:
            self._fit_model(self.X, self.y)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self._adjust_trust_region()
            self._adjust_pareto_weights()
            self.lcb_kappa = max(self.lcb_kappa * self.kappa_decay, self.min_kappa)

        return self.best_y, self.best_x
