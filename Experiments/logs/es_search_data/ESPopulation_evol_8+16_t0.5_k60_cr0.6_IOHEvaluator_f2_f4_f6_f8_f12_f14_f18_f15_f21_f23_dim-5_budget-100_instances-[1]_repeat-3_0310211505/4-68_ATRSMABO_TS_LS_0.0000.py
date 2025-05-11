from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from scipy.optimize import minimize

class ATRSMABO_TS_LS:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(10*dim, self.budget//5)
        self.trust_region_width = 2.0  # Initial trust region width
        self.success_threshold = 0.1 # Threshold for increasing trust region
        self.best_y = np.inf # Initialize best_y with a large value
        self.best_x = None
        self.thompson_temperature = 1.0
        self.thompson_decay = 0.95
        self.min_thompson_temperature = 0.1
        self.local_search_frequency = 5
        self.local_search_step_size = 0.1

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, center=None, width=None):
        # sample points within the trust region
        # return array of shape (n_points, n_dims)
        if center is None:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            # Sample within the trust region
            lower_bound = np.maximum(self.bounds[0], center - width / 2)
            upper_bound = np.minimum(self.bounds[1], center + width / 2)
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            scaled_sample = qmc.scale(sample, lower_bound, upper_bound)
            return scaled_sample

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        
        # Define kernels
        kernel_rbf = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        kernel_matern = ConstantKernel(1.0) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        # Initialize models
        gp_rbf = GaussianProcessRegressor(kernel=kernel_rbf, n_restarts_optimizer=5)
        gp_matern = GaussianProcessRegressor(kernel=kernel_matern, n_restarts_optimizer=5)
        
        # Fit models
        gp_rbf.fit(X, y)
        gp_matern.fit(X, y)
        
        return gp_rbf, gp_matern

    def _acquisition_function(self, X, model):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1)) # Return zeros if no data is available

        mu, sigma = model.predict(X, return_std=True)

        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement
        best = np.min(self.y)
        imp = best - mu
        Z = imp / (sigma + 1e-9)  # Adding a small constant to avoid division by zero
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0 # avoid division by zero

        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Generate candidate points within the trust region
        n_candidates = max(1000, batch_size * 100)
        X_cand = self._sample_points(n_candidates, center=self.best_x, width=self.trust_region_width)

        # Thompson Sampling for Acquisition Function Selection
        # Sample a function value from each GPR model
        mu_rbf, sigma_rbf = self.model_rbf.predict(self.X, return_std=True)
        mu_matern, sigma_matern = self.model_matern.predict(self.X, return_std=True)

        # Sample from the posterior predictive distribution
        try:
            sampled_rbf = np.random.normal(mu_rbf, sigma_rbf * self.thompson_temperature)
            sampled_matern = np.random.normal(mu_matern, sigma_matern * self.thompson_temperature)
        except Exception as e:
            sampled_rbf = mu_rbf
            sampled_matern = mu_matern

        # Calculate acquisition function values for each model
        ei_rbf = self._acquisition_function(X_cand, self.model_rbf)
        ei_matern = self._acquisition_function(X_cand, self.model_matern)

        # Select the model with the highest sampled value
        if np.mean(sampled_rbf) > np.mean(sampled_matern):
            acq_values = ei_rbf
        else:
            acq_values = ei_matern

        # Select top-k points based on acquisition function values
        top_indices = np.argsort(acq_values.flatten())[::-1][:batch_size]
        return X_cand[top_indices]

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        # Do not change the function signature
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.concatenate((self.X, new_X), axis=0)
            self.y = np.concatenate((self.y, new_y), axis=0)

    def _local_search(self, func):
        # Perform local search around the current best solution
        # Use finite differences to estimate gradients
        def obj_func(x):
            return func(x)

        # Define bounds for the local search
        lower_bound = np.maximum(self.bounds[0], self.best_x - self.trust_region_width / 2)
        upper_bound = np.minimum(self.bounds[1], self.best_x + self.trust_region_width / 2)
        bounds = list(zip(lower_bound, upper_bound))

        # Perform local optimization using L-BFGS-B
        result = minimize(obj_func, self.best_x, method='L-BFGS-B', bounds=bounds)

        # Update best solution if a better one is found
        if result.fun < self.best_y:
            self.best_y = result.fun
            self.best_x = result.x
            self.n_evals += result.nfev
            return True
        return False

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_x = X_init[np.argmin(y_init)][0]
        self.best_y = np.min(y_init)

        # Optimization loop
        batch_size = max(1, self.dim // 2)
        iteration = 0
        while self.n_evals < self.budget:
            # Fit the model
            self.model_rbf, self.model_matern = self._fit_model(self.X, self.y)

            # Select next points
            X_next = self._select_next_points(batch_size)

            # Evaluate points
            y_next = self._evaluate_points(func, X_next)

            # Update evaluated points
            self._update_eval_points(X_next, y_next)

            # Update best solution
            new_best_y = np.min(self.y)
            new_best_x = self.X[np.argmin(self.y)]

            # Adjust trust region width
            if (self.best_y - new_best_y) / self.best_y > self.success_threshold:
                self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)  # Increase if successful
            else:
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)  # Decrease if not successful

            self.best_y = new_best_y
            self.best_x = new_best_x

            # Dynamic Thompson temperature
            self.thompson_temperature = max(self.thompson_temperature * self.thompson_decay, self.min_thompson_temperature)

            # Perform local search periodically
            if iteration % self.local_search_frequency == 0:
                self._local_search(func)

            iteration += 1

        # Return best solution
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
