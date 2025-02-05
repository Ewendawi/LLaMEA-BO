from typing import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
from scipy.special import expit

class GP_Matern_EI_MSL_SobolBOv1:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X = None
        self.y = None
        self.model = None
        self.best_y = np.inf
        self.best_x = None
        self.n_initial_points = min(10 + self.dim, self.budget // 2)
        self.n_restarts_optimizer = min(5 + self.dim, self.budget // 4)

    def _sample_points(self, n_points):
        # sample points
        # the return has a shape (n_points, n_dims)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        u_sample = sampler.random(n_points)
        sample = qmc.scale(u_sample, self.bounds[0], self.bounds[1])
        return sample

    def _fit_model(self, X, y):
        # Fit surrogate model
        # Auto-tuning the parameters of model
        # return the model
        kernel = Matern(nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=self.n_restarts_optimizer, alpha=1e-6
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # the return has a shape (n_points, 1)
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        gamma = (self.best_y - mu) / (sigma + 1e-9)
        ei = sigma * (gamma * 0.5 * (1 + expit(gamma / np.sqrt(2))) + (1 / np.sqrt(2 * np.pi)) * np.exp(-gamma**2 / 2))
        return -ei
    
    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Choose a suitable selection strategy, such as methods in scipy.optimize.minimize or any other strategies.
        # show me your options and justify the final decision in the comment.
        # the return has a shape (batch_size, n_dims)

        # Option 1: Random Sampling.
        # Pros: Simple and can help with exploration.
        # Cons: Not very efficient in exploiting the information from the surrogate model.

        # Option 2: Using scipy.optimize.minimize with a single starting point (e.g., the best point found so far).
        # Pros: Can work well for simple acquisition functions.
        # Cons: May get stuck in local optima of the acquisition function, especially in high dimensions.

        # Option 3: Using scipy.optimize.minimize with multiple starting points (multi-start local search).
        # Pros: Increases the chance of finding the global optimum of the acquisition function.
        # Cons: Computationally more expensive.

        # Option 4: Using a genetic algorithm to optimize the acquisition function.
        # Pros: Can handle complex, multi-modal acquisition functions.
        # Cons: Requires careful tuning of the genetic algorithm parameters.

        # Option 5: Latin Hypercube Sampling (LHS)
        # Pros: LHS ensures good space-filling properties.
        # Cons: May lead to premature convergence if not handled carefully.

        # Final Decision Justification:
        # I choose Option 3 (multi-start L-BFGS-B) with Option 5 as the starting points because it provides a good balance between
        # exploration and exploitation. Multi-start helps to avoid getting stuck in local optima,
        # and L-BFGS-B is efficient for smooth, continuous optimization problems which is often the case for acquisition functions.
        # Sobol sequence is used to generate starting points to improve space filling property.

        x_next = None
        best_acq = np.inf
        
        x_seeds = self._sample_points(self.n_restarts_optimizer)

        for x_try in x_seeds:
            res = minimize(
                lambda x: self._acquisition_function(x.reshape(1, -1)),
                x_try.reshape(1, -1),
                bounds=self.bounds.T,
                method="L-BFGS-B",
            )

            if res.fun < best_acq:
                best_acq = res.fun
                x_next = res.x

        return x_next.reshape(1, -1)

    def _update_sample_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(
        self, func: Callable[[np.ndarray], np.float64]
    ) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # Do not change the function signature
        # Return a tuple (best_y, best_x)

        X = self._sample_points(self.n_initial_points)
        y = np.array([[func(x)] for x in X])
        self._update_sample_points(X, y)

        self.best_y = np.min(self.y)
        self.best_x = self.X[np.argmin(self.y)]
        
        rest_of_budget = self.budget - self.n_initial_points
        while rest_of_budget > 0:
            # Optimization
            self.model = self._fit_model(self.X, self.y)
            x_next = self._select_next_points(1)
            
            y_next = func(x_next[0])
            if y_next < self.best_y:
                self.best_y = y_next
                self.best_x = x_next[0]

            self._update_sample_points(x_next, y_next)
            
            rest_of_budget -= 1
        return self.best_y, self.best_x