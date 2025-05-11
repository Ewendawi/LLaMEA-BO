from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.model_selection import train_test_split

class SurrogateEnsembleBO:
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
        self.gp_ensemble = []
        self.ensemble_weights = []
        self.best_x = None
        self.best_y = float('inf')
        self.n_ensemble = 3 # Number of surrogate models in the ensemble

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate models in the ensemble
        # return the models
        # Do not change the function signature

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        if not self.gp_ensemble:
            # Initialize the ensemble with different kernels
            kernels = [
                ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed"),
                ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds="fixed", nu=0.5),
                ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds="fixed", nu=1.5)
            ]
            for i in range(self.n_ensemble):
                gp = GaussianProcessRegressor(kernel=kernels[i % len(kernels)], n_restarts_optimizer=0, alpha=1e-6)
                gp.fit(X_train, y_train)
                self.gp_ensemble.append(gp)
                self.ensemble_weights.append(1.0 / self.n_ensemble)  # Initialize with equal weights
        else:
            # Update the existing models
            for gp in self.gp_ensemble:
                gp.fit(X_train, y_train)

        # Adjust weights based on validation performance
        val_errors = []
        for gp in self.gp_ensemble:
            y_pred, _ = gp.predict(X_val, return_std=True)
            error = np.mean((y_pred - y_val.flatten())**2)  # Mean Squared Error
            val_errors.append(error)

        # Convert errors to weights using softmax
        val_errors = np.array(val_errors)
        weights = np.exp(-val_errors) / np.sum(np.exp(-val_errors))
        self.ensemble_weights = weights

    def _acquisition_function(self, X):
        # Implement acquisition function: Upper Confidence Bound
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if not self.gp_ensemble:
            return np.random.normal(size=(len(X), 1))
        else:
            mu_ensemble = np.zeros((len(X), 1))
            sigma_ensemble = np.zeros((len(X), 1))

            for i, gp in enumerate(self.gp_ensemble):
                mu, sigma = gp.predict(X, return_std=True)
                mu_ensemble += self.ensemble_weights[i] * mu.reshape(-1, 1)
                sigma_ensemble += self.ensemble_weights[i] * sigma.reshape(-1, 1)

            # UCB calculation
            kappa = 2.0  # Exploration-exploitation trade-off
            ucb = mu_ensemble + kappa * sigma_ensemble
            return ucb

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        candidates = self._sample_points(100*batch_size)
        acquisition_values = self._acquisition_function(candidates)
        best_indices = np.argsort(acquisition_values.flatten())[-batch_size:]  # Select top batch_size points
        return candidates[best_indices]

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
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Select points by acquisition function
            next_X = self._select_next_points(batch_size)

            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)

            # Update the evaluated points
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
