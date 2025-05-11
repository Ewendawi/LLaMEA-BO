from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution, minimize

class ATGREBO_DKAB:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.gp = None
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
        self.success_threshold_de = 0.2
        self.gradient_weight = 0.1
        self.finite_difference_step = 0.1
        self.ei_gradient_weight_ratio = 0.5
        self.ei_gradient_weight_adapt_rate = 0.1
        self.length_scale = 1.0
        self.length_scale_bounds = (1e-2, 1e2)
        self.gradient_history = []  # Store past gradients for refinement
        self.gradient_history_length = 5

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        scaled_sample = qmc.scale(sample, -self.trust_region_radius, self.trust_region_radius)
        return scaled_sample + self.trust_region_center

    def _fit_model(self, X, y):
        def neg_log_likelihood(length_scale):
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=length_scale, length_scale_bounds="fixed")
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
            gp.fit(X, y)
            return -gp.log_marginal_likelihood()

        res = minimize(neg_log_likelihood, x0=self.length_scale, bounds=[self.length_scale_bounds])
        self.length_scale = res.x[0]

        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.length_scale, length_scale_bounds="fixed")
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        gp.fit(X, y)
        return gp

    def _approximate_gradients(self, func, x):
        gradients = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.finite_difference_step
            x_minus[i] -= self.finite_difference_step
            x_plus = np.clip(x_plus, self.bounds[0], self.bounds[1])
            x_minus = np.clip(x_minus, self.bounds[0], self.bounds[1])
            gradients[i] = (func(x_plus) - func(x_minus)) / (2 * self.finite_difference_step)

        # Refine gradient using history
        if self.gradient_history:
            historical_gradients = np.array(self.gradient_history)
            gradients = 0.7 * gradients + 0.3 * np.mean(historical_gradients, axis=0)

        return gradients

    def _acquisition_function(self, X):
        if self.gp is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.gp.predict(X, return_std=True)
        improvement = self.best_y - mu
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        gradient_exploration = np.zeros_like(ei)
        for i in range(len(X)):
            gradients = self._approximate_gradients(lambda x: self.gp.predict(x.reshape(1, -1))[0], X[i])
            gradient_exploration[i] = self.gradient_weight * np.linalg.norm(gradients)

        ei_weight = self.ei_gradient_weight_ratio
        gradient_weight = 1 - self.ei_gradient_weight_ratio
        acquisition = (ei_weight * ei + gradient_weight * gradient_exploration).reshape(-1, 1)
        acquisition = (1 - self.exploration_weight) * acquisition + self.exploration_weight * sigma.reshape(-1, 1)
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

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_X = np.clip(initial_X, self.bounds[0], self.bounds[1])
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        while self.n_evals < self.budget:
            self.gp = self._fit_model(self.X, self.y)

            batch_size = min(1, self.budget - self.n_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)

            # Store gradient before updating eval points
            gradients = self._approximate_gradients(func, next_X[0])
            self.gradient_history.append(gradients)
            if len(self.gradient_history) > self.gradient_history_length:
                self.gradient_history.pop(0)

            self._update_eval_points(next_X, next_y)

            predicted_improvement = self.best_y - self.gp.predict(self.best_x.reshape(1, -1))[0]
            actual_improvement = self.best_y - min(self.y)[0]

            if abs(predicted_improvement) < 1e-9:
                ratio = 0.0
            else:
                ratio = actual_improvement / predicted_improvement

            if ratio > self.success_threshold:
                self.trust_region_radius *= self.trust_region_expand
            elif ratio < self.failure_threshold:
                self.trust_region_radius *= self.trust_region_shrink
                self.trust_region_radius = max(self.trust_region_radius, 0.1)

            self.trust_region_center = self.best_x

            if len(self.y) > self.de_popsize:
                recent_ys = self.y[-self.de_popsize:]
                improvement_ratios = (recent_ys[:-1] - recent_ys[1:]) / recent_ys[:-1]
                success_rate = np.sum(improvement_ratios > 0) / len(improvement_ratios)

                if success_rate > self.success_threshold_de:
                    self.de_mutation *= (1 + self.mutation_adaptation_rate)
                    self.de_crossover *= (1 - self.mutation_adaptation_rate)
                else:
                    self.de_mutation *= (1 - self.mutation_adaptation_rate)
                    self.de_crossover *= (1 + self.mutation_adaptation_rate)
                self.de_mutation = np.clip(self.de_mutation, 0.1, 1.99)
                self.de_crossover = np.clip(self.de_crossover, 0.1, 0.99)

            mean_sigma = np.mean(self.gp.predict(self.X, return_std=True)[1])
            self.ei_gradient_weight_ratio += self.ei_gradient_weight_adapt_rate * (1 - mean_sigma)
            self.ei_gradient_weight_ratio = np.clip(self.ei_gradient_weight_ratio, 0.1, 0.9)

            self.exploration_weight *= 0.99
            self.exploration_weight = max(self.exploration_weight, 0.01)

        return self.best_y, self.best_x
