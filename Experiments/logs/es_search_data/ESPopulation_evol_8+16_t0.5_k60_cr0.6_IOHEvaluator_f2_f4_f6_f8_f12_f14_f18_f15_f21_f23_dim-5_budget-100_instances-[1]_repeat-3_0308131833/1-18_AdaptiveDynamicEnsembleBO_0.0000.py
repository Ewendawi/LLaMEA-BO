from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

class AdaptiveDynamicEnsembleBO:
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
        self.local_search_iters = 5

        # Define different kernels for the ensemble
        kernels = [
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)),
            ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5),
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=0.5, length_scale_bounds=(0.1, 5.0)) # More local
        ]
        for kernel in kernels:
            self.gps.append(GaussianProcessRegressor(kernel=kernel + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-7, 1e-5)), n_restarts_optimizer=5))

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
            avg_distances = np.mean(distances, axis=1)
            acquisition += self.novelty_weight * avg_distances.reshape(-1, 1)
        
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
                # Calculate the log-likelihood of the data under the GP model
                log_likelihood = gp.log_marginal_likelihood(gp.kernel_.theta, clone_kernel=False)
                
                # Update weights based on the log-likelihood
                self.weights[i] = np.exp(log_likelihood)
            
            # Normalize the weights
            self.weights /= np.sum(self.weights)
            
            # Decay the weights to encourage exploration
            self.weights *= self.weight_decay
            self.weights += (1 - self.weight_decay) / self.n_models
            self.weights /= np.sum(self.weights)

    def _local_search(self, func, x0):
        # Perform local search around x0
        bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
        res = minimize(func, x0, method="L-BFGS-B", bounds=bounds, options={'maxiter': self.local_search_iters})
        return res.fun, res.x

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
            batch_size = min(5, self.budget - self.n_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Local search around the best point
            best_y_local, best_x_local = self._local_search(func, self.best_x)
            if best_y_local < self.best_y:
                self.best_y = best_y_local
                self.best_x = best_x_local
                self.n_evals += self.local_search_iters # Approximate number of evaluations
                if self.X is not None:
                    self.X = np.vstack((self.X, best_x_local.reshape(1, -1)))
                    self.y = np.vstack((self.y, best_y_local))
                else:
                    self.X = best_x_local.reshape(1, -1)
                    self.y = np.array(best_y_local).reshape(-1,1)

        return self.best_y, self.best_x
