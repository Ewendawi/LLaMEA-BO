from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ADEHBBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(20 * dim, self.budget // 5) # Increased initial exploration
        self.best_y = float('inf')
        self.best_x = None
        self.hall_of_fame_X = []
        self.hall_of_fame_y = []
        self.hall_of_fame_size = max(5, dim // 2)  # Hall of Fame size
        self.diversity_threshold = 0.5  # Initial diversity threshold
        self.diversity_threshold_decay = 0.995 # Decay the diversity threshold over time

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, center=None, radius=None):
        # sample points
        # return array of shape (n_points, n_dims)
        if center is None:
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            points = sampler.random(n=n_points)
            return qmc.scale(points, self.bounds[0], self.bounds[1])
        else:
            # Sample within a ball around center with radius
            points = np.random.normal(loc=center, scale=radius/3, size=(n_points, self.dim))
            points = np.clip(points, self.bounds[0], self.bounds[1])
            return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement
        imp = self.best_y - mu - 1e-9  # Adding a small constant to avoid division by zero
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Diversity penalty
        if self.hall_of_fame_X:
            distances = np.array([np.linalg.norm(X - hof_x, axis=1) for hof_x in self.hall_of_fame_X]).T
            min_distances = np.min(distances, axis=1, keepdims=True)
            diversity_penalty = np.where(min_distances < self.diversity_threshold, -100, 0)  # Penalize close points
            ei += diversity_penalty
        return ei

    def _select_next_points(self, batch_size, gp):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Optimization of acquisition function using L-BFGS-B
        n_lbfgs = batch_size // 3
        n_local = batch_size // 3
        n_random = batch_size - n_lbfgs - n_local
        
        x_next = []
        
        # L-BFGS-B optimization
        x_starts = self._sample_points(n_lbfgs)  # Multiple starting points
        for x_start in x_starts:
            res = minimize(lambda x: -self._acquisition_function(x.reshape(1, -1), gp),
                           x_start,
                           bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)],
                           method='L-BFGS-B')
            x_next.append(res.x)

        # Random sampling around the best point
        if self.best_x is not None:
            random_samples = self._sample_points(n_local, center=self.best_x, radius=0.5)
            x_next.extend(random_samples)

        # Add more random samples for exploration
        random_samples = self._sample_points(n_random)
        x_next.extend(random_samples)

        return np.array(x_next)

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

            # Update Hall of Fame
            if not self.hall_of_fame_X:
                self.hall_of_fame_X.append(self.best_x)
                self.hall_of_fame_y.append(self.best_y)
            else:
                distances = np.array([np.linalg.norm(self.best_x - hof_x) for hof_x in self.hall_of_fame_X])
                if np.min(distances) > self.diversity_threshold:
                    self.hall_of_fame_X.append(self.best_x)
                    self.hall_of_fame_y.append(self.best_y)
                    if len(self.hall_of_fame_X) > self.hall_of_fame_size:
                        # Remove worst performing member
                        worst_idx = np.argmax(self.hall_of_fame_y)
                        self.hall_of_fame_X.pop(worst_idx)
                        self.hall_of_fame_y.pop(worst_idx)
        
        # Decay diversity threshold
        self.diversity_threshold *= self.diversity_threshold_decay
    
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
            gp = self._fit_model(self.X, self.y)

            # Dynamic batch size
            remaining_evals = self.budget - self.n_evals
            batch_size = min(max(1, int(remaining_evals / (self.dim * 0.1))), 20) # Ensure at least 1 point and limit to 20

            # select points by acquisition function
            next_X = self._select_next_points(batch_size, gp)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
