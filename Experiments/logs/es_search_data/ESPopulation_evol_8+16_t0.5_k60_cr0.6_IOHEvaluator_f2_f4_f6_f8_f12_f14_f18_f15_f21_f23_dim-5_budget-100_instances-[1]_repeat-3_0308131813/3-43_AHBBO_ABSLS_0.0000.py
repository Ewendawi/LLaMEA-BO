from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class AHBBO_ABSLS:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim

        self.best_y = np.inf
        self.best_x = None

        self.max_batch_size = min(10, dim)
        self.min_batch_size = 1
        self.exploration_weight = 0.1
        self.exploration_decay = 0.995
        self.min_exploration = 0.01
        self.uncertainty_threshold = 0.5
        self.local_search_prob = 0.1  # Probability of performing local search
        self.local_search_step_size = 0.1 # Initial step size for local search

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        min_dist = np.min(np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2), axis=1, keepdims=True)
        exploration = min_dist / np.max(min_dist)

        # Distance to best solution
        dist_to_best = np.linalg.norm(X - self.best_x, axis=1, keepdims=True)
        # Scale the distance to best solution
        scaled_dist_to_best = dist_to_best / np.max(np.linalg.norm(self.bounds[1] - self.bounds[0]))

        acquisition = ei + self.exploration_weight * exploration - 0.01 * scaled_dist_to_best # Favor points closer to best

        return acquisition

    def _select_next_points(self, batch_size):
        candidate_points = self._sample_points(100 * batch_size)
        acquisition_values = self._acquisition_function(candidate_points)
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]
        return next_points

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)
    
    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))
        
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

    def _local_search(self, func):
        # Perform local search around the best solution
        
        def obj_func(x):
            return func(x)  # Evaluate the black-box function

        # Define the bounds for the local search
        bounds = [(max(self.bounds[0][i], self.best_x[i] - self.local_search_step_size),
                   min(self.bounds[1][i], self.best_x[i] + self.local_search_step_size)) for i in range(self.dim)]

        # Perform the local optimization using a bounded optimization algorithm
        res = minimize(obj_func, self.best_x, method='L-BFGS-B', bounds=bounds)

        # If the local search finds a better solution, update the best solution
        if res.fun < self.best_y:
            self.best_y = res.fun
            self.best_x = res.x

        return self.best_y, self.best_x

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Adjust batch size based on uncertainty
            _, sigma = self.model.predict(self.X, return_std=True)
            avg_sigma = np.mean(sigma)
            
            if avg_sigma > self.uncertainty_threshold:
                batch_size = self.max_batch_size
                exploration_decay = 0.99
            else:
                batch_size = self.min_batch_size
                exploration_decay = 0.999
            
            remaining_evals = self.budget - self.n_evals
            batch_size = min(batch_size, remaining_evals)

            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self.model = self._fit_model(self.X, self.y)
            
            # Update exploration weight
            self.exploration_weight = max(self.exploration_weight * exploration_decay, self.min_exploration)

            # Perform local search with a probability
            if np.random.rand() < self.local_search_prob:
                # Adjust local search step size based on uncertainty
                self.local_search_step_size = 0.1 * avg_sigma # Smaller steps when more certain
                self._local_search(func)
                

        return self.best_y, self.best_x
