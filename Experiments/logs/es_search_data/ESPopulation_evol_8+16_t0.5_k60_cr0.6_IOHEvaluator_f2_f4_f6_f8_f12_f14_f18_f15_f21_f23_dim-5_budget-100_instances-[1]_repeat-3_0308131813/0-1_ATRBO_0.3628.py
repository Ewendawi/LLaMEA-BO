from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ATRBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim

        self.best_y = np.inf
        self.best_x = None

        self.trust_region_center = np.zeros(dim)
        self.trust_region_radius = 2.5  # Initial trust region radius
        self.radius_min = 0.1
        self.radius_max = 5.0
        self.gamma_inc = 2.0
        self.gamma_dec = 0.5
        self.eta_good = 0.9
        self.eta_bad = 0.1

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        scaled_sample = qmc.scale(sample, self.bounds[0], self.bounds[1])

        # Clip to trust region
        for i in range(n_points):
            if np.linalg.norm(scaled_sample[i] - self.trust_region_center) > self.trust_region_radius:
                direction = scaled_sample[i] - self.trust_region_center
                direction = direction / np.linalg.norm(direction)
                scaled_sample[i] = self.trust_region_center + direction * self.trust_region_radius

        return scaled_sample

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0
        return ei

    def _select_next_point(self):
        # Select the next point to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (1, n_dims)

        def obj_func(x):
            x = x.reshape(1, -1)
            return -self._acquisition_function(x)[0][0]

        x0 = self.trust_region_center  # Start from the trust region center
        
        # Define the bounds for each dimension within the trust region
        bounds = [(max(self.bounds[0][i], self.trust_region_center[i] - self.trust_region_radius),
                   min(self.bounds[1][i], self.trust_region_center[i] + self.trust_region_radius)) for i in range(self.dim)]

        res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds)
        next_point = res.x.reshape(1, -1)
        return next_point

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
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

        # Update best seen value
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Optimization
            # select points by acquisition function
            next_X = self._select_next_point()
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Update the trust region
            predicted_y = self.model.predict(next_X)[0]
            actual_y = next_y[0][0]
            rho = (self.y[-1][0] - actual_y) / (self.y[-1][0] - predicted_y) if (self.y[-1][0] - predicted_y) !=0 else 0

            if rho < self.eta_bad:
                self.trust_region_radius = max(self.radius_min, self.gamma_dec * self.trust_region_radius)
            else:
                self.trust_region_center = next_X[0]
                if rho > self.eta_good:
                    self.trust_region_radius = min(self.radius_max, self.gamma_inc * self.trust_region_radius)

            self.model = self._fit_model(self.X, self.y)

        return self.best_y, self.best_x
