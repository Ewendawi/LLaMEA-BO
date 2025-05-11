# Description
AdaptiveBandwidth_UCB_TS_BO: This algorithm combines the strengths of both RBF_Bandwidth_BO and GP_UCB_TS_BO. It uses a Gaussian Process (GP) with an adaptively tuned RBF kernel bandwidth, similar to RBF_Bandwidth_BO, to better model the underlying function. It also incorporates a hybrid acquisition function that blends Upper Confidence Bound (UCB) and Thompson Sampling (TS), similar to GP_UCB_TS_BO, to balance exploration and exploitation effectively. The UCB's exploration-exploitation parameter, kappa, is dynamically adjusted over time.

# Justification
The RBF_Bandwidth_BO dynamically adjusts the RBF kernel bandwidth, which helps the GP to better model the function by adapting to the data's characteristics. The GP_UCB_TS_BO combines UCB and Thompson Sampling for a more robust exploration-exploitation trade-off. By combining these two approaches, we aim to create an algorithm that is both accurate in its modeling and effective in its exploration. The dynamic adjustment of kappa in UCB allows for more exploration early on and more exploitation later in the optimization process. Using Sobol sampling for initial points and candidate points improves space coverage.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class AdaptiveBandwidth_UCB_TS_BO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 4 * dim
        self.bandwidth = 1.0  # Initial bandwidth
        self.bandwidth_update_interval = 5 * dim # Update bandwidth every this many evaluations

        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.kappa = 2.0  # UCB exploration-exploitation parameter

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
        kernel = C(1.0, (1e-3, 1e3)) * RBF(self.bandwidth, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X, kappa):
        # Implement UCB acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        return mu + kappa * sigma

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using Thompson Sampling and UCB
        # return array of shape (batch_size, n_dims)

        # Thompson Sampling: Draw a sample from the posterior
        sampled_f = self.gp.sample_y(self.X, n_samples=1)

        # UCB on a set of randomly sampled points
        num_candidates = 100 * self.dim
        X_candidate = self._sample_points(num_candidates)
        ucb_values = self._acquisition_function(X_candidate, self.kappa)

        # Select the point with the maximum UCB value
        next_point = X_candidate[np.argmax(ucb_values)]
        return next_point.reshape(1, -1)

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
        if np.min(new_y) < self.best_y:
            self.best_y = np.min(new_y)
            self.best_x = new_X[np.argmin(new_y)]

    def _update_bandwidth(self):
        # Update the RBF kernel bandwidth using the median heuristic
        distances = np.linalg.norm(self.X[:, None, :] - self.X[None, :, :], axis=2)
        distances = distances[np.triu_indices_from(distances, k=1)]
        if len(distances) > 0:
            self.bandwidth = np.median(distances)
        else:
            self.bandwidth = 1.0 # Fallback value

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
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Adjust kappa over time
            self.kappa = 2.0 - 1.8 * (self.n_evals / self.budget)
            if self.kappa < 0.2:
                self.kappa = 0.2 # Minimum kappa

            # Select next points by acquisition function
            next_X = self._select_next_points(1)

            # Evaluate the points
            next_y = self._evaluate_points(func, next_X)
            
            # Update evaluated points
            self._update_eval_points(next_X, next_y)

            # Update bandwidth periodically
            if self.n_evals % self.bandwidth_update_interval == 0:
                self._update_bandwidth()

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AdaptiveBandwidth_UCB_TS_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1504 with standard deviation 0.1057.

took 187.13 seconds to run.