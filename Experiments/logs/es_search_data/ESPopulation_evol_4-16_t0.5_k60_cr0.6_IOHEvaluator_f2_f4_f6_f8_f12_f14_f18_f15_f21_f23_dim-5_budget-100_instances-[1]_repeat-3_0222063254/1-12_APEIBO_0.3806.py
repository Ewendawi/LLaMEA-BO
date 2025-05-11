from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.spatial.distance import cdist

class APEIBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(10 * dim, self.budget // 5)
        self.best_y = float('inf')
        self.best_x = None
        self.diversity_threshold = 0.1 # Minimum distance for diversity penalty
        self.diversity_weight = 0.1  # Initial weight for diversity penalty

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        points = sampler.random(n=n_points)
        return qmc.scale(points, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp, patch_indices):
        # Implement Expected Improvement on a stochastic patch with diversity penalty

        mu, sigma = gp.predict(X, return_std=True)  # Predict on the full space
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu - 1e-9
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Diversity penalty
        if self.X is not None:
            distances = cdist(X[:, patch_indices], self.X[:, patch_indices])
            min_distances = np.min(distances, axis=1)
            diversity_penalty = np.where(min_distances < self.diversity_threshold,
                                           (min_distances - self.diversity_threshold)**2,
                                           0) # Only penalize if too close
            ei = ei - self.diversity_weight * diversity_penalty.reshape(-1, 1)

        return ei

    def _select_next_points(self, batch_size, gp):
        # Select the next points to evaluate
        next_X = []
        for _ in range(batch_size):
            # Dynamic patch size
            remaining_evals = self.budget - self.n_evals
            patch_size = max(1, min(self.dim, int(self.dim * remaining_evals / self.budget) + 1))

            # Randomly select a patch of dimensions
            patch_indices = np.random.choice(self.dim, patch_size, replace=False)

            # Optimization of acquisition function using L-BFGS-B
            x_starts = self._sample_points(10)  # Generate multiple starting points
            best_x = None
            best_acq = float('-inf')

            for x_start in x_starts:
                res = minimize(lambda x: -self._acquisition_function(x.reshape(1, -1), gp, patch_indices),
                               x_start,
                               bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)],
                               method='L-BFGS-B')

                if -res.fun > best_acq:
                    best_acq = -res.fun
                    best_x = res.x

            next_X.append(best_x)

        return np.array(next_X)

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

        #Adapt diversity weight
        self.diversity_weight = min(0.5, self.diversity_weight + 0.01)


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
            batch_size = min(max(1, int(remaining_evals / (self.dim * 0.1))), 20)

            # select points by acquisition function
            next_X = self._select_next_points(batch_size, gp)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
