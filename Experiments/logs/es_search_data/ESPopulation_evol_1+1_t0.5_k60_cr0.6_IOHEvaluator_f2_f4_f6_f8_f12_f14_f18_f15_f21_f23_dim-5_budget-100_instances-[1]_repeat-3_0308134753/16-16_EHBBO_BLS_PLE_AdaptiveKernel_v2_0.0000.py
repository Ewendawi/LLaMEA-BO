from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
from scipy.optimize import fmin_l_bfgs_b

class EHBBO_BLS_PLE_AdaptiveKernel_v2:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim # initial samples
        self.gp = None
        self.best_x = None
        self.best_y = np.inf
        self.exploration_factor = 1.0 # initial exploration factor
        self.exploration_decay = 0.99 # decay rate for exploration factor
        self.kernel_length_scale = 1.0
        self.kernel_update_interval = dim # Update kernel every this many evaluations

    def _sample_points(self, n_points):
        # sample points using Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.kernel_length_scale, length_scale_bounds=(1e-2, 10))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-5) # Increased restarts for kernel tuning
        self.gp.fit(X, y)
        self.kernel_length_scale = self.gp.kernel_.k2.length_scale # Update the length scale
        return self.gp

    def _acquisition_function(self, X, beta):
        # Implement Upper Confidence Bound (UCB) acquisition function
        mu, sigma = self.gp.predict(X, return_std=True)
        ucb = mu - beta * sigma  # UCB acquisition function (minimization)
        return ucb

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate by maximizing the acquisition function
        # Use a simple optimization strategy: randomly sample points and choose the best one
        best_x = None
        best_acq = np.inf
        beta = 2 * np.log(self.n_evals + 1) # UCB exploration-exploitation parameter
        candidate_X = self._sample_points(10 * batch_size)
        acq_values = self._acquisition_function(candidate_X, beta)
        best_index = np.argmin(acq_values)
        best_x = candidate_X[best_index]

        return best_x

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

        # Update best_x and best_y
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

    def local_search(self, gp, x0):
        # Perform a local search around x0 using the GP prediction
        # Adjust the search radius based on the GP's predicted uncertainty
        _, sigma = gp.predict(x0.reshape(1, -1), return_std=True)
        search_radius = min(0.5 + sigma[0], 1.0)  # Limit the search radius

        def gp_objective(x):
            mu, _ = gp.predict(x.reshape(1, -1), return_std=True)
            return mu[0]

        bounds = [(max(self.bounds[0][i], x0[i] - search_radius), min(self.bounds[1][i], x0[i] + search_radius)) for i in range(self.dim)]
        res = minimize(gp_objective, x0, method='L-BFGS-B', bounds=bounds)
        return res.x

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Optimization loop
        while self.n_evals < self.budget:
            # Fit the GP model
            if self.n_evals % self.kernel_update_interval == 0:
                # Update kernel length scale using L-BFGS-B
                def neg_log_likelihood(length_scale):
                    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 10))
                    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-5)
                    gp.fit(self.X, self.y)
                    return -gp.log_marginal_likelihood()

                bounds = [(1e-2, 10)] * self.dim
                x0 = np.array([self.kernel_length_scale] * self.dim)
                result = fmin_l_bfgs_b(neg_log_likelihood, x0, bounds=bounds, approx_grad=True)
                self.kernel_length_scale = result[0][0] # Take the first length scale if dim > 1

                self._fit_model(self.X, self.y) # Refit with optimized kernel
            else:
                # Refit without kernel optimization for efficiency
                kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.kernel_length_scale, length_scale_bounds="fixed")
                self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-5)
                self.gp.fit(self.X, self.y)


            # Select next point(s) by acquisition function
            next_X = self._select_next_points(1)

            # Evaluate the new point(s)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Local search around the best point
            local_x = self.local_search(self.gp, self.best_x)
            local_y = func(local_x)
            self.n_evals += 1
            if local_y < self.best_y:
                self.best_y = local_y
                self.best_x = local_x


            # Decay exploration factor
            self.exploration_factor *= self.exploration_decay

        return self.best_y, self.best_x
