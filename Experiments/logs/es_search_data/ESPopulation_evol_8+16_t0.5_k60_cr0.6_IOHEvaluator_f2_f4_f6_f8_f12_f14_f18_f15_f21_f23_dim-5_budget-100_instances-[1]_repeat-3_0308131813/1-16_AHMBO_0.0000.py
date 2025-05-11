from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class AHMBO:
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

        self.exploration_factor = 0.1 # Initial exploration factor
        self.kernel_length_scale = 1.0 # Initial kernel length scale

        # Hyperparameter tuning parameters
        self.n_hypers = 3 # Number of hyperparameter sets to try in each meta-optimization iteration
        self.meta_iter = 3 # Number of meta-optimization iterations
        self.batch_size = min(10, dim)

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        kernel = C(1.0, (1e-3, 1e3)) * RBF(self.kernel_length_scale, (1e-2, 1e2))
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

        # Distance-based exploration term
        min_dist = np.min(np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2), axis=1, keepdims=True)
        exploration = min_dist / np.max(min_dist)

        # Hybrid acquisition function
        acquisition = ei + self.exploration_factor * exploration
        return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Generate candidate points
        candidate_points = self._sample_points(100 * batch_size)

        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_points)

        # Select the top batch_size points with the highest acquisition values
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]

        return next_points

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

    def _meta_optimize(self, func):
        # Meta-optimization loop to tune hyperparameters
        best_hypers = (self.exploration_factor, self.kernel_length_scale)
        best_aocc = np.inf

        # Define a grid of hyperparameter values to try
        exploration_factors = np.linspace(0.01, 0.2, self.n_hypers)
        kernel_length_scales = np.linspace(0.5, 2.0, self.n_hypers)

        for exploration_factor in exploration_factors:
            for kernel_length_scale in kernel_length_scales:
                # Create a temporary BO object with the current hyperparameters
                temp_bo = AHMBO(self.budget, self.dim)
                temp_bo.exploration_factor = exploration_factor
                temp_bo.kernel_length_scale = kernel_length_scale
                temp_bo.X = self.X.copy() if self.X is not None else None
                temp_bo.y = self.y.copy() if self.y is not None else None
                temp_bo.n_evals = self.n_evals
                temp_bo.best_y = self.best_y
                temp_bo.best_x = self.best_x

                # Initial fit of the model
                if temp_bo.X is not None and temp_bo.y is not None:
                    temp_bo.model = temp_bo._fit_model(temp_bo.X, temp_bo.y)
                else:
                    # Handle the case where there's no data to fit the model
                    continue

                # Run the temporary BO object for a few iterations
                while temp_bo.n_evals < min(self.n_evals + self.n_init, self.budget):
                    remaining_evals = self.budget - temp_bo.n_evals
                    batch_size = min(temp_bo.batch_size, remaining_evals)
                    next_X = temp_bo._select_next_points(batch_size)
                    next_y = temp_bo._evaluate_points(func, next_X)
                    temp_bo._update_eval_points(next_X, next_y)
                    temp_bo.model = temp_bo._fit_model(temp_bo.X, temp_bo.y)

                # Calculate the AOCC for the current hyperparameters
                aocc = temp_bo.best_y # Use best_y as a proxy for AOCC

                # Update the best hyperparameters if the current ones are better
                if aocc < best_aocc:
                    best_aocc = aocc
                    best_hypers = (exploration_factor, kernel_length_scale)

        return best_hypers

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

        # Meta-optimization loop
        for i in range(self.meta_iter):
            # Tune hyperparameters
            self.exploration_factor, self.kernel_length_scale = self._meta_optimize(func)
            self.model = self._fit_model(self.X, self.y) # Refit the model with new kernel length scale

            # Optimization loop
            remaining_evals = self.budget - self.n_evals
            batch_size = min(self.batch_size, remaining_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self.model = self._fit_model(self.X, self.y)

            if self.n_evals >= self.budget:
                break

        return self.best_y, self.best_x
