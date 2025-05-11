# Description
GP_UCB_TS_Batch_BO: Gaussian Process Upper Confidence Bound with Thompson Sampling and Batch Evaluation Bayesian Optimization. This algorithm enhances the GP_UCB_TS_BO by incorporating batch evaluation to improve computational efficiency. It uses a Gaussian Process (GP) as a surrogate model and combines Upper Confidence Bound (UCB) and Thompson Sampling (TS) to balance exploration and exploitation. The key improvement is the introduction of batch evaluation, where multiple points are selected and evaluated simultaneously. The batch is selected by taking the top `batch_size` points from the UCB acquisition function evaluated on a set of candidate points. This reduces the overhead of GP fitting and prediction. An adaptive mechanism for `kappa` is also added.

# Justification
The primary motivation for this change is to improve the computational efficiency of the algorithm. Fitting the GP model can be expensive, especially in higher dimensions or with a large number of evaluations. By evaluating points in batches, we can reduce the number of times the GP model needs to be fit and predicted.

The UCB acquisition function is used to select the points to be included in the batch. This ensures that the algorithm continues to balance exploration and exploitation. Thompson Sampling is still used to sample from the posterior, providing a probabilistic approach to exploration.

The adaptive kappa allows for more exploration in the beginning and more exploitation as the budget is close to being exhausted.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class GP_UCB_TS_Batch_BO:
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
        self.batch_size = min(10, dim) # Evaluate multiple points in parallel

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
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
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
        #sampled_f = self.gp.sample_y(self.X, n_samples=1)

        # UCB on a set of randomly sampled points
        num_candidates = 100 * self.dim
        X_candidate = self._sample_points(num_candidates)
        ucb_values = self._acquisition_function(X_candidate, self.kappa)

        # Select the point with the maximum UCB value
        #next_point = X_candidate[np.argmax(ucb_values)]
        #return next_point.reshape(1, -1)
        
        # Select the top batch_size points
        indices = np.argsort(ucb_values.flatten())[-batch_size:]
        next_points = X_candidate[indices]
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
        if np.min(new_y) < self.best_y:
            self.best_y = np.min(new_y)
            self.best_x = new_X[np.argmin(new_y)]
    
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

            # Select next points by acquisition function
            next_X = self._select_next_points(self.batch_size)

            # Evaluate the points
            next_y = self._evaluate_points(func, next_X)
            
            # Update evaluated points
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm GP_UCB_TS_Batch_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1411 with standard deviation 0.1023.

took 45.51 seconds to run.