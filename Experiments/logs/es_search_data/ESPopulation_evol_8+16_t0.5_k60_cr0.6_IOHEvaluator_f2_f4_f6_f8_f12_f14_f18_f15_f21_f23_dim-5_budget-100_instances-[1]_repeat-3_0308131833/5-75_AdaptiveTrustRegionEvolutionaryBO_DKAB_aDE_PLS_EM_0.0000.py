from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution, minimize

class AdaptiveTrustRegionEvolutionaryBO_DKAB_aDE_PLS_EM:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.gp_ensemble = []
        self.n_ensemble = 3  # Number of GPs in the ensemble
        self.length_scales = [1.0, 0.5, 2.0]  # Different initial length scales
        self.gp_weights = np.ones(self.n_ensemble) / self.n_ensemble  # Initialize weights equally
        self.trust_region_radius = 2.0
        self.trust_region_shrink = 0.5
        self.trust_region_expand = 1.5
        self.success_threshold = 0.75
        self.failure_threshold = 0.25
        self.trust_region_center = np.zeros(dim)
        self.best_x = None
        self.best_y = np.inf
        self.de_popsize = 10
        self.de_mutation = 0.5
        self.de_crossover = 0.7
        self.mutation_adaptation_rate = 0.1
        self.crossover_adaptation_rate = 0.1
        self.success_threshold_de = 0.2
        self.exploration_weight = 0.1
        self.exploration_weight_decay = 0.99
        self.length_scale_bounds = (1e-2, 1e2)
        self.min_trust_region_radius = 0.1
        self.local_search_prob = 0.1
        self.local_search_success_prob_increase = 0.2
        self.local_search_failure_prob_decrease = 0.1

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        scaled_sample = qmc.scale(sample, -self.trust_region_radius, self.trust_region_radius)
        return scaled_sample + self.trust_region_center

    def _fit_model(self, X, y):
        # Fit and tune surrogate model ensemble
        self.gp_ensemble = []  # Clear the ensemble
        for i in range(self.n_ensemble):
            # Optimize the length_scale for each GP
            def neg_log_likelihood(length_scale):
                kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=length_scale, length_scale_bounds="fixed")
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
                gp.fit(X, y)
                return -gp.log_marginal_likelihood()

            res = minimize(neg_log_likelihood, x0=self.length_scales[i], bounds=[self.length_scale_bounds])
            self.length_scales[i] = res.x[0]

            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.length_scales[i], length_scale_bounds="fixed")
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
            gp.fit(X, y)
            self.gp_ensemble.append(gp)

    def _acquisition_function(self, X):
        # Implement acquisition function using the ensemble
        if not self.gp_ensemble:
            return np.zeros((len(X), 1))

        mu_ensemble = np.zeros((len(X), self.n_ensemble))
        sigma_ensemble = np.zeros((len(X), self.n_ensemble))

        for i, gp in enumerate(self.gp_ensemble):
            mu, sigma = gp.predict(X, return_std=True)
            mu_ensemble[:, i] = mu
            sigma_ensemble[:, i] = sigma

        # Weighted average of predictions
        mu = np.average(mu_ensemble, axis=1, weights=self.gp_weights)
        # Weighted average of standard deviations (approximation)
        sigma = np.average(sigma_ensemble, axis=1, weights=self.gp_weights)

        # Expected Improvement acquisition function
        improvement = self.best_y - mu
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Weighted combination of EI and exploration
        acquisition = (1 - self.exploration_weight) * ei.reshape(-1, 1) + self.exploration_weight * sigma.reshape(-1, 1)
        return acquisition

    def _select_next_points(self, batch_size):
        def de_objective(x):
            return -self._acquisition_function(x.reshape(1, -1))[0, 0]

        de_bounds = list(zip(np.maximum(self.bounds[0], self.trust_region_center - self.trust_region_radius),
                             np.minimum(self.bounds[1], self.trust_region_center + self.trust_region_radius)))

        de_result = differential_evolution(
            func=de_objective,
            bounds=de_bounds,
            popsize=self.de_popsize,
            mutation=self.de_mutation,
            recombination=self.de_crossover,
            maxiter=5,
            tol=0.01,
            seed=None,
            strategy='rand1bin',
            init='latinhypercube'
        )

        next_point = de_result.x.reshape(1, -1)
        return next_point

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
        if self.y[best_index, 0] < self.best_y:
            self.best_x = self.X[best_index]
            self.best_y = self.y[best_index, 0]

    def _local_search(self, func):
        def objective(x):
            return func(x)

        bounds = list(zip(np.maximum(self.bounds[0], self.trust_region_center - self.trust_region_radius),
                             np.minimum(self.bounds[1], self.trust_region_center + self.trust_region_radius)))

        x0 = self.best_x

        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

        return result.x, result.fun

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_X = np.clip(initial_X, self.bounds[0], self.bounds[1])
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        while self.n_evals < self.budget:
            # Optimization

            # Fit GP model ensemble
            self._fit_model(self.X, self.y)

            # Update GP weights based on performance on training data
            for i in range(self.n_ensemble):
                mu, _ = self.gp_ensemble[i].predict(self.X, return_std=True)
                error = np.mean((mu.flatten() - self.y.flatten())**2)  # Mean Squared Error
                self.gp_weights[i] = np.exp(-error)  # Weight based on inverse error
            self.gp_weights /= np.sum(self.gp_weights)  # Normalize weights

            # Select points by acquisition function using differential evolution
            batch_size = min(1, self.budget - self.n_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Calculate the ratio of improvement
            predicted_improvement = self.best_y - self._acquisition_function(self.best_x.reshape(1, -1))[0, 0]
            actual_improvement = self.best_y - min(self.y)[0]

            if abs(predicted_improvement) < 1e-9:
                ratio = 0.0
            else:
                ratio = actual_improvement / predicted_improvement

            # Adjust trust region size
            if ratio > self.success_threshold and abs(predicted_improvement) > 1e-3:
                self.trust_region_radius *= self.trust_region_expand
                self.local_search_prob = min(1.0, self.local_search_prob + self.local_search_success_prob_increase)
            elif ratio < self.failure_threshold:
                self.trust_region_radius *= self.trust_region_shrink
                self.trust_region_radius = max(self.trust_region_radius, self.min_trust_region_radius)
                self.local_search_prob = max(0.0, self.local_search_prob - self.local_search_failure_prob_decrease)

            # Update trust region center
            self.trust_region_center = self.best_x

            # Adapt mutation and crossover rate based on success
            if len(self.y) > self.de_popsize:
                recent_ys = self.y[-self.de_popsize:]
                improvement_ratios = (recent_ys[:-1] - recent_ys[1:]) / (recent_ys[:-1] + 1e-9)
                success_rate = np.sum(improvement_ratios > 0) / len(improvement_ratios)

                if success_rate > self.success_threshold_de:
                    self.de_mutation *= (1 - self.mutation_adaptation_rate)
                    self.de_crossover *= (1 + self.crossover_adaptation_rate)
                else:
                    self.de_mutation *= (1 + self.mutation_adaptation_rate)
                    self.de_crossover *= (1 - self.crossover_adaptation_rate)

                self.de_mutation = np.clip(self.de_mutation, 0.1, 1.99)
                self.de_crossover = np.clip(self.de_crossover, 0.1, 0.99)

            # Decay exploration weight
            self.exploration_weight *= self.exploration_weight_decay
            self.exploration_weight = max(self.exploration_weight, 0.01)

            # Perform local search with probability self.local_search_prob
            if np.random.rand() < self.local_search_prob and self.n_evals < self.budget:
                local_x, local_y = self._local_search(func)
                if local_y < self.best_y:
                    local_x = local_x.reshape(1, -1)
                    local_y = np.array([[local_y]])
                    self._update_eval_points(local_x, local_y)

        return self.best_y, self.best_x
