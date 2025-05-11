from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution, minimize
from numpy.linalg import norm as linalg_norm

class AdaptiveEvolutionaryTrustRegionBO_DKAEBPA:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(5*dim, self.budget//10)

        # Do not add any other arguments without a default value
        self.gp = None
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.radius_decay = 0.95
        self.radius_increase = 1.1
        self.min_radius = 0.1
        self.success_ratio = 0.0 #ratio to track the success of trust region
        self.random_restart_prob = 0.05
        self.de_pop_size = 10  # Population size for differential evolution
        self.lcb_kappa = 2.0  # Kappa parameter for Lower Confidence Bound
        self.kappa_decay = 0.99 # Decay rate for kappa
        self.min_kappa = 0.1 # Minimum value for kappa
        self.initial_length_scale = 1.0
        self.length_scale = self.initial_length_scale
        self.length_scale_decay = 0.95
        self.length_scale_increase = 1.05
        self.min_length_scale = 0.01
        self.max_length_scale = 100.0
        self.initial_trust_region_radius = min(2.0, 5.0 / np.sqrt(dim)) # Scale initial radius with dimension
        self.diversity_weight = 0.5 # Initial weight for diversity
        self.diversity_weight_decay = 0.98 # Decay rate for diversity weight
        self.min_diversity_weight = 0.1
        self.batch_size = min(4, self.dim) # Initial batch size

    def _sample_points(self, n_points):
        # sample points within the trust region or randomly
        # return array of shape (n_points, n_dims)
        if self.best_x is None or np.random.rand() < self.random_restart_prob:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            points = []
            while len(points) < n_points:
                sample = np.random.uniform(-1, 1, size=(self.dim,))
                sample = self.best_x + self.trust_region_radius * sample
                # Clip to bounds
                sample = np.clip(sample, self.bounds[0], self.bounds[1])
                points.append(sample)
            return np.array(points)

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.length_scale, length_scale_bounds=(1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        self.gp.fit(X, y)

        # Optimize kernel lengthscale using L-BFGS-B
        def neg_log_likelihood(theta):
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=theta, length_scale_bounds=(1e-2, 1e2))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
            gp.fit(X, y)
            return -gp.log_marginal_likelihood()

        bounds = [(1e-2, 1e2)]  # Bounds for length_scale
        result = minimize(neg_log_likelihood, self.length_scale, method='L-BFGS-B', bounds=bounds)
        self.length_scale = result.x[0]

        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function: Lower Confidence Bound
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            mu, sigma = self.gp.predict(X, return_std=True)
            LCB = mu - self.lcb_kappa * sigma
            return LCB.reshape(-1, 1)

    def _calculate_diversity(self, X):
        # Calculate the diversity of points in X based on Euclidean distance to nearest existing point
        if self.X is None or len(self.X) == 0:
            return np.ones(len(X))  # Maximum diversity if no existing points

        diversity = []
        for x in X:
            distances = np.linalg.norm(self.X - x, axis=1)
            min_distance = np.min(distances)
            diversity.append(min_distance)
        return np.array(diversity)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using differential evolution within the trust region, considering Pareto optimality
        if self.best_x is None:
            return self._sample_points(batch_size)

        def de_objective(x):
            x = x.reshape(1, -1)
            lcb = self._acquisition_function(x)[0, 0]
            diversity = self._calculate_diversity(x)[0]

            # Normalize LCB and diversity to [0, 1]
            lcb_normalized = (lcb - np.min(self._acquisition_function(self.X))) / (np.max(self._acquisition_function(self.X)) - np.min(self._acquisition_function(self.X)) + 1e-9) if self.X is not None and len(self.X) > 0 else 0
            diversity_normalized = diversity / (self.trust_region_radius + 1e-9)

            # Weighted sum of LCB and diversity
            return self.diversity_weight * (-diversity_normalized) + (1 - self.diversity_weight) * lcb_normalized

        de_bounds = [(max(self.bounds[0][i], self.best_x[i] - self.trust_region_radius),
                      min(self.bounds[1][i], self.best_x[i] + self.trust_region_radius)) for i in range(self.dim)]

        # Adjust maxiter based on remaining budget and optimization progress
        remaining_evals = self.budget - self.n_evals
        maxiter = max(1, int(remaining_evals / (self.de_pop_size * self.dim * 2)))
        maxiter = min(maxiter, 100) #limit maxiter to prevent excessive computation

        result = differential_evolution(de_objective, de_bounds, popsize=self.de_pop_size, maxiter=maxiter, tol=0.01, disp=False)

        return result.x.reshape(1, -1)


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
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]
            self.success_ratio = 1.0 #reset success ratio
        else:
            self.success_ratio *= 0.75 #reduce success ratio if not improving

    def _adjust_trust_region(self):
        # Adjust the trust region size based on the success
        if self.gp is not None and self.best_x is not None:
            mu, sigma = self.gp.predict(self.X, return_std=True)
            error = np.mean(np.abs(mu.flatten() - self.y.flatten()))
        else:
            error = 0.0

        if self.success_ratio > 0.5:
            self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)

        # Adjust more aggressively if GP predictions are inaccurate
        if error > 0.5:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)

    def _adjust_kernel_bandwidth(self):
        # Adjust kernel bandwidth based on optimization progress and GP uncertainty
        if self.gp is not None and self.best_x is not None:
            # Estimate landscape complexity using the gradient norm of the GP mean function
            delta = 1e-4  # Small perturbation for gradient estimation
            gradient = np.zeros_like(self.best_x)
            for i in range(self.dim):
                x_plus = self.best_x.copy()
                x_minus = self.best_x.copy()
                x_plus[i] += delta
                x_minus[i] -= delta
                mu_plus, _ = self.gp.predict(x_plus.reshape(1, -1), return_std=True)
                mu_minus, _ = self.gp.predict(x_minus.reshape(1, -1), return_std=True)
                gradient[i] = (mu_plus - mu_minus) / (2 * delta)
            gradient_norm = linalg_norm(gradient)

            # Adjust bandwidth based on gradient norm and GP uncertainty
            _, sigma = self.gp.predict(self.best_x.reshape(1, -1), return_std=True)
            uncertainty = sigma[0]

            if gradient_norm > 1.0 or uncertainty > 0.5:  # High complexity or uncertainty, reduce bandwidth
                self.length_scale = max(self.length_scale * self.length_scale_decay, self.min_length_scale)
            else:  # Low complexity and uncertainty, increase bandwidth
                self.length_scale = min(self.length_scale * self.length_scale_increase, self.max_length_scale)
        else:
            self.length_scale = self.initial_length_scale

    def _adjust_batch_size(self):
        # Adjust batch size based on success ratio and remaining budget
        if self.success_ratio > 0.6:
            self.batch_size = min(self.batch_size + 1, self.dim, int(self.budget / (self.n_evals + 1)))
        elif self.success_ratio < 0.3:
            self.batch_size = max(1, self.batch_size - 1)
        else:
            self.batch_size = min(4, self.dim)

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Optimization loop
        self.trust_region_radius = self.initial_trust_region_radius
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Adjust batch size
            self._adjust_batch_size()

            # Select points by acquisition function
            next_X = self._select_next_points(self.batch_size)

            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)

            # Update the evaluated points
            self._update_eval_points(next_X, next_y)

            # Adjust trust region radius
            self._adjust_trust_region()

            # Decay kappa
            self.lcb_kappa = max(self.lcb_kappa * self.kappa_decay, self.min_kappa)

            # Adjust kernel bandwidth
            self._adjust_kernel_bandwidth()

            # Decay diversity weight
            self.diversity_weight = max(self.diversity_weight * self.diversity_weight_decay, self.min_diversity_weight)

        return self.best_y, self.best_x
