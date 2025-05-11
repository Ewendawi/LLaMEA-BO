from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

class DynamicResourceAllocationBO:
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
        self.exploration_phase = True # Start in exploration phase
        self.improvement_threshold = 1e-3 # Threshold for switching phases
        self.improvement_rate = 0.0
        self.last_best_y = float('inf')
        self.phase_switch_interval = 10 # Number of iterations between phase switch checks

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
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function_exploration(self, X):
        # Implement acquisition function: Thompson Sampling (Exploration)
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            y_samples = self.gp.sample_y(X, n_samples=1)
            return y_samples

    def _acquisition_function_exploitation(self, X):
        # Implement acquisition function: Expected Improvement (Exploitation)
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            mu, sigma = self.gp.predict(X, return_std=True)
            sigma = np.clip(sigma, 1e-9, np.inf)
            imp = self.best_y - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        best_points = []
        for _ in range(batch_size):
            x0 = self._sample_points(1)  # Start from a random point

            def obj(x):
                if self.exploration_phase:
                    return -self._acquisition_function_exploration(x.reshape(1, -1))[0, 0]
                else:
                    return -self._acquisition_function_exploitation(x.reshape(1, -1))[0, 0]  # Negative EI for minimization

            # Limit the number of L-BFGS-B iterations based on the remaining budget
            max_iter = min(50, (self.budget - self.n_evals) // batch_size)
            res = minimize(obj, x0, bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)], method='L-BFGS-B', options={'maxiter': max_iter})
            best_points.append(res.x)

        return np.array(best_points)

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

    def _check_phase_switch(self):
        # Check if the algorithm should switch between exploration and exploitation phases
        if self.best_y < self.last_best_y:
            self.improvement_rate = (self.last_best_y - self.best_y) / self.last_best_y
        else:
            self.improvement_rate = 0.0

        if self.improvement_rate < self.improvement_threshold and not self.exploration_phase:
            self.exploration_phase = True  # Switch to exploration
            print("Switching to exploration phase")
        elif self.improvement_rate >= self.improvement_threshold and self.exploration_phase:
            self.exploration_phase = False # Switch to exploitation
            print("Switching to exploitation phase")

        self.last_best_y = self.best_y

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
        batch_size = min(2, self.dim)
        iteration = 0
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Select points by acquisition function
            next_X = self._select_next_points(batch_size)

            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)

            # Update the evaluated points
            self._update_eval_points(next_X, next_y)

            # Check if the algorithm should switch phases
            if iteration % self.phase_switch_interval == 0:
                self._check_phase_switch()

            iteration += 1

        return self.best_y, self.best_x
