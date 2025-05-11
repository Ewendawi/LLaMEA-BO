from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import minimize, Bounds
import warnings

class AGATBO_LS:
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
        self.gradient_weight = 0.01
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 2.0
        self.model_agreement_threshold = 0.75
        self.kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        self.model = None

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.Halton(d=self.dim, seed=42)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        try:
            model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5, alpha=1e-5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)
            self.kernel = model.kernel_  # Update kernel with optimized parameters
            return model
        except Exception as e:
            print(f"GP fitting failed: {e}. Returning None.")
            return None

    def _acquisition_function(self, X, model):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if self.best_y is None:
            ei = np.zeros_like(mu)
        else:
            imp = self.best_y - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0  # handle zero sigma

        # Add gradient-based exploration term
        if self.X is not None:
            dmu_dx = self._predict_gradient(X, model)
            gradient_norm = np.linalg.norm(dmu_dx, axis=1, keepdims=True)
            ei = ei + self.gradient_weight * gradient_norm

        return ei

    def _predict_gradient(self, X, model):
        # Predict the gradient of the GPR mean function
        # return array of shape (n_points, n_dims)
        X = np.atleast_2d(X)
        dmu_dx = np.zeros((X.shape[0], self.dim))
        for i in range(self.dim):
            def obj(x):
                x_prime = x.copy()
                x_prime[i] += 1e-6
                return model.predict(x_prime.reshape(1, -1))[0]
            def obj0(x):
                return model.predict(x.reshape(1, -1))[0]
            dmu_dx[:, i] = (np.array([obj(x) for x in X]) - np.array([obj0(x) for x in X]))/1e-6
        return dmu_dx

    def _select_next_points(self, batch_size, trust_region_center):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # return array of shape (batch_size, n_dims)

        # Sample within the trust region using Sobol sequence
        sobol_engine = qmc.Sobol(d=self.dim, seed=42)
        samples = sobol_engine.random(n=100 * batch_size)

        # Scale and shift samples to be within the trust region
        scaled_samples = trust_region_center + self.trust_region_radius * (2 * samples - 1)

        # Clip samples to stay within the problem bounds
        scaled_samples = np.clip(scaled_samples, self.bounds[0], self.bounds[1])

        # Calculate acquisition function values
        if self.model is None:
            return scaled_samples[:batch_size]
        acquisition_values = self._acquisition_function(scaled_samples, self.model)

        # Select top batch_size points based on acquisition function values
        indices = np.argsort(-acquisition_values.flatten())[:batch_size]
        selected_points = scaled_samples[indices]

        return selected_points

    def _local_search(self, func, x0):
        # Perform local search around x0 within the trust region
        # return array of shape (n_dims,)
        
        # Define bounds for local search based on trust region
        lower_bounds = np.maximum(self.bounds[0], x0 - self.trust_region_radius)
        upper_bounds = np.minimum(self.bounds[1], x0 + self.trust_region_radius)
        bounds = Bounds(lower_bounds, upper_bounds)

        # Perform local search using SLSQP
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(func, x0, method='SLSQP', bounds=bounds)

        return result.x

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
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        trust_region_center = self.X[np.argmin(self.y)] # Initialize trust region center
        batch_size = 5
        while self.n_evals < self.budget:
            # Optimization
            self.model = self._fit_model(self.X, self.y)
            if self.model is None:
                print("GP model fitting failed. Returning current best.")
                return self.best_y, self.best_x

            # select points by acquisition function
            next_X = self._select_next_points(batch_size, trust_region_center)

            # Apply local search to refine selected points
            next_X_refined = np.array([self._local_search(func, x) for x in next_X])
            
            next_y = self._evaluate_points(func, next_X_refined)
            self._update_eval_points(next_X_refined, next_y)

            # Model agreement check (simplified)
            predicted_y = self.model.predict(next_X_refined)
            agreement = np.corrcoef(next_y.flatten(), predicted_y.flatten())[0, 1]

            # Adjust trust region size
            if np.isnan(agreement) or agreement < self.model_agreement_threshold:
                self.trust_region_radius *= self.trust_region_shrink_factor
            else:
                self.trust_region_radius *= self.trust_region_expand_factor
                self.trust_region_radius = min(self.trust_region_radius, 5.0) # Limit expansion

            # Update trust region center
            trust_region_center = self.X[np.argmin(self.y)]

        return self.best_y, self.best_x
