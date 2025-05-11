from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution

class AdaptiveEvolutionaryTrustRegionBO_DKAEL:
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
        self.trust_region_radius = 2.0 * np.sqrt(dim) # Initial trust region radius, scaled by dimension
        self.radius_decay = 0.95
        self.radius_increase = 1.1
        self.min_radius = 0.1
        self.success_ratio = 0.0 #ratio to track the success of trust region
        self.random_restart_prob = 0.05
        self.de_pop_size = 10  # Population size for differential evolution
        self.lcb_kappa = 2.0  # Kappa parameter for Lower Confidence Bound
        self.kappa_decay = 0.99 # Decay rate for kappa
        self.min_kappa = 0.1 # Minimum value for kappa
        self.gp_error_threshold = 0.1 # Threshold for GP error
        self.noise_estimate = 1e-4 #initial noise estimate
        self.length_scale = 1.0 # Initial length scale
        self.length_scale_decay = 0.95
        self.length_scale_increase = 1.05
        self.min_length_scale = 0.1
        self.gp_reinit_threshold = 0.5 # Threshold for GP re-initialization

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
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
        self.gp.fit(X, y)
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

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using differential evolution within the trust region
        # return array of shape (batch_size, n_dims)

        if self.best_x is None:
            return self._sample_points(batch_size)

        def de_objective(x):
            return self._acquisition_function(x.reshape(1, -1))[0, 0]

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

        #Update noise estimate
        self.noise_estimate = np.var(self.y)


    def _adjust_trust_region(self):
        # Adjust the trust region size based on the success
        if self.best_x is not None:
            #Calculate GP error within the trust region
            distances = np.linalg.norm(self.X - self.best_x, axis=1)
            within_tr = distances < self.trust_region_radius
            if np.any(within_tr):
                X_tr = self.X[within_tr]
                y_tr = self.y[within_tr]
                mu, _ = self.gp.predict(X_tr, return_std=True)
                gp_error = np.mean(np.abs(mu.reshape(-1,1) - y_tr))

                # Check for GP re-initialization
                if gp_error > self.gp_reinit_threshold:
                    # Re-initialize GP with new samples
                    new_X = self._sample_points(self.n_init)
                    new_y = self._evaluate_points(func, new_X)
                    self._update_eval_points(new_X, new_y)
                    self._fit_model(self.X, self.y)
                    return # Skip trust region and lengthscale adjustment this iteration

            else:
                gp_error = 0.0

            # Gradient-aware trust region adaptation
            if self.gp is not None:
                gradient = self.gp.predict(self.best_x.reshape(1, -1), return_std=False, return_cov=False)
                gradient_norm = np.linalg.norm(gradient)
            else:
                gradient_norm = 0.0

            if self.success_ratio > 0.5 and gp_error < self.gp_error_threshold:
                self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
            elif gradient_norm < 0.1: #flat region
                self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
            else:
                self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius) #shrink if no best_x

    def _adjust_length_scale(self):
        # Adjust length scale based on landscape complexity
        if self.y is not None and len(self.y) > 1:
            data_variance = np.var(self.y)
            if data_variance > self.noise_estimate * 10: #complex landscape
                self.length_scale = min(self.length_scale * self.length_scale_increase, 5.0)
            else: #smooth landscape
                self.length_scale = max(self.length_scale * self.length_scale_decay, self.min_length_scale)


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
        batch_size = min(4, self.dim) #increase batch size to 4 or dim, whichever is smaller
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Select points by acquisition function
            next_X = self._select_next_points(batch_size)

            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)

            # Update the evaluated points
            self._update_eval_points(next_X, next_y)

            # Adjust trust region radius
            self._adjust_trust_region()

            # Adjust length scale
            self._adjust_length_scale()

            # Decay kappa, adjust min_kappa based on noise estimate
            self.lcb_kappa = max(self.lcb_kappa * self.kappa_decay, self.min_kappa)
            self.min_kappa = 0.1 + np.sqrt(self.noise_estimate) #dynamic min kappa

        return self.best_y, self.best_x
