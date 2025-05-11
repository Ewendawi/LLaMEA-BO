from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error

class ADBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(10 * dim, self.budget // 5)
        self.best_y = float('inf')
        self.best_x = None
        self.hall_of_fame_X = []
        self.hall_of_fame_y = []
        self.hall_of_fame_size = max(5, dim // 2)
        self.diversity_threshold = 0.5
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.trust_region_shrink_factor = 0.7
        self.trust_region_expand_factor = 1.2
        self.min_trust_region_radius = 0.1
        self.diversity_penalty_scale = 1.0

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, center=None, radius=None):
        # sample points
        # return array of shape (n_points, n_dims)
        if center is None:
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            points = sampler.random(n=n_points)
            return qmc.scale(points, self.bounds[0], self.bounds[1])
        else:
            # Sample within a ball around center with radius
            points = np.random.normal(loc=center, scale=radius/3, size=(n_points, self.dim))
            points = np.clip(points, self.bounds[0], self.bounds[1])
            return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement
        imp = self.best_y - mu - 1e-9  # Adding a small constant to avoid division by zero
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Diversity penalty
        if self.hall_of_fame_X:
            distances = np.array([np.linalg.norm(X - hof_x, axis=1) for hof_x in self.hall_of_fame_X]).T
            min_distances = np.min(distances, axis=1, keepdims=True)
            diversity_penalty = np.where(min_distances < self.diversity_threshold, -self.diversity_penalty_scale * sigma, 0)  # Penalize close points scaled by sigma
            ei += diversity_penalty
        return ei

    def _select_next_points(self, batch_size, gp):
        # Select the next points to evaluate
        # Optimization of acquisition function using L-BFGS-B within trust region
        x_starts = self._sample_points(batch_size // 2, center=self.best_x, radius=self.trust_region_radius) if self.best_x is not None else self._sample_points(batch_size // 2)
        x_next = []
        for x_start in x_starts:
            res = minimize(lambda x: -self._acquisition_function(x.reshape(1, -1), gp),
                           x_start,
                           bounds=[(max(self.bounds[0][i], self.best_x[i] - self.trust_region_radius) if self.best_x is not None else self.bounds[0][i],
                                    min(self.bounds[1][i], self.best_x[i] + self.trust_region_radius) if self.best_x is not None else self.bounds[1][i]) for i in range(self.dim)],
                           method='L-BFGS-B')
            x_next.append(res.x)

        # QMC sampling for exploration and diversity, outside trust region
        explore_samples = self._sample_points(batch_size - batch_size // 2)
        x_next.extend(explore_samples)

        return np.array(x_next)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        # Do not change the function signature
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

        # Update best seen value
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

            # Update Hall of Fame
            if not self.hall_of_fame_X:
                self.hall_of_fame_X.append(self.best_x)
                self.hall_of_fame_y.append(self.best_y)
            else:
                distances = np.array([np.linalg.norm(self.best_x - hof_x) for hof_x in self.hall_of_fame_X])
                if np.min(distances) > self.diversity_threshold:
                    self.hall_of_fame_X.append(self.best_x)
                    self.hall_of_fame_y.append(self.best_y)
                    if len(self.hall_of_fame_X) > self.hall_of_fame_size:
                        # Remove worst performing member
                        worst_idx = np.argmax(self.hall_of_fame_y)
                        self.hall_of_fame_X.pop(worst_idx)
                        self.hall_of_fame_y.pop(worst_idx)

    def _adjust_trust_region(self, gp):
        # Adjust trust region based on GP's performance
        if len(self.X) > 5:
            mu, _ = gp.predict(self.X, return_std=True)
            mse = mean_squared_error(self.y, mu)

            if mse < 0.01:  # If GP is accurate, expand trust region
                self.trust_region_radius = min(self.trust_region_radius * self.trust_region_expand_factor, 5.0)
            else:  # Otherwise, shrink trust region
                self.trust_region_radius = max(self.trust_region_radius * self.trust_region_shrink_factor, self.min_trust_region_radius)
        else:
            self.trust_region_radius = min(self.trust_region_radius * self.trust_region_expand_factor, 5.0)

    def _adjust_diversity_penalty(self):
         # Adaptive adjustment of diversity penalty
        if len(self.hall_of_fame_X) == self.hall_of_fame_size:
            self.diversity_penalty_scale *= 1.1
        else:
            self.diversity_penalty_scale *= 0.9
        self.diversity_penalty_scale = np.clip(self.diversity_penalty_scale, 0.1, 10)

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        while self.n_evals < self.budget:
            # Optimization
            gp = self._fit_model(self.X, self.y)

            # Dynamic batch size
            remaining_evals = self.budget - self.n_evals
            batch_size = min(max(1, int(remaining_evals / (self.dim * 0.1))), 20) # Ensure at least 1 point and limit to 20

            # select points by acquisition function
            next_X = self._select_next_points(batch_size, gp)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            self._adjust_trust_region(gp)
            self._adjust_diversity_penalty()

        return self.best_y, self.best_x
