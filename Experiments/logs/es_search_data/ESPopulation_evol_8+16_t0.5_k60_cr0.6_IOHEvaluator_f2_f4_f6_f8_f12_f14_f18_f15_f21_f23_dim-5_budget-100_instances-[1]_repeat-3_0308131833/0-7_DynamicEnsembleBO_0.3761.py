from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist

class DynamicEnsembleBO:
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
        self.n_models = 3 # Number of GP models in the ensemble
        self.gps = []
        self.weights = np.ones(self.n_models) / self.n_models # Initial weights
        self.weight_decay = 0.95
        self.novelty_weight = 0.1
        self.best_x = None
        self.best_y = np.inf
        
        # Define different kernels for the ensemble
        kernels = [
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed"),
            ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds="fixed", nu=1.5),
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=0.5, length_scale_bounds="fixed") # More local
        ]
        for kernel in kernels:
            self.gps.append(GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6))

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
        for gp in self.gps:
            gp.fit(X, y)

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.X is None:
            return np.zeros((len(X), 1))

        mu = np.zeros((len(X), 1))
        sigma = np.zeros((len(X), 1))
        
        # Weighted average of GP predictions
        for i, gp in enumerate(self.gps):
            m, s = gp.predict(X, return_std=True)
            mu += self.weights[i] * m.reshape(-1, 1)
            sigma += self.weights[i] * s.reshape(-1, 1)

        # Expected Improvement acquisition function
        improvement = self.best_y - mu
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # Avoid division by zero
        acquisition = ei

        # Novelty search component
        if len(self.X) > 0:
            distances = cdist(X, self.X)
            min_distances = np.min(distances, axis=1)
            acquisition += self.novelty_weight * min_distances.reshape(-1, 1)
        
        return acquisition.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Optimize the acquisition function to find the next points
        x_tries = self._sample_points(batch_size * 10) # Generate more candidates
        acq_values = self._acquisition_function(x_tries)

        # Select the top batch_size points based on the acquisition function values
        indices = np.argsort(acq_values.flatten())[::-1][:batch_size]
        return x_tries[indices]

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
        
        # Update best observed solution
        best_index = np.argmin(self.y)
        if self.y[best_index, 0] < self.best_y:
            self.best_x = self.X[best_index]
            self.best_y = self.y[best_index, 0]
            
        # Update GP model weights based on recent performance
        if len(self.X) > self.n_init:
            for i, gp in enumerate(self.gps):
                # Calculate the error on the existing points
                y_pred, sigma = gp.predict(self.X, return_std=True)
                error = np.mean((y_pred.reshape(-1, 1) - self.y)**2)
                
                # Update weights based on the inverse of the error
                self.weights[i] = np.exp(-error)
            
            # Normalize the weights
            self.weights /= np.sum(self.weights)
            
            # Decay the weights to encourage exploration
            self.weights *= self.weight_decay
            self.weights += (1 - self.weight_decay) / self.n_models
            self.weights /= np.sum(self.weights)

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

            # Fit GP models
            self._fit_model(self.X, self.y)

            # Select points by acquisition function
            batch_size = min(10, self.budget - self.n_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
