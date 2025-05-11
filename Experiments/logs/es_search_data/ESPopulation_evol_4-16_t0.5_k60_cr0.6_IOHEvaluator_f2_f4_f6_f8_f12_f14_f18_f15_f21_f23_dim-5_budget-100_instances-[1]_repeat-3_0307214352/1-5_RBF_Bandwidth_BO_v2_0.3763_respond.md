# Description
RBF_Bandwidth_BO_v2: Bayesian Optimization with dynamically adjusted RBF kernel bandwidth using a more robust median heuristic and adaptive restarts for acquisition function optimization. This algorithm employs a Gaussian Process (GP) as a surrogate model and Expected Improvement (EI) as the acquisition function. The key improvements are: 1) A refined median heuristic for bandwidth tuning that considers only the nearest neighbors to avoid over-smoothing, and 2) An adaptive number of restarts for the L-BFGS-B optimizer based on the GP's uncertainty, allowing for more exploration in uncertain regions and exploitation in well-modeled regions.

# Justification
The original RBF_Bandwidth_BO uses a simple median heuristic for bandwidth selection, which can be sensitive to outliers and may lead to over-smoothing, especially in high-dimensional spaces. By considering only the distances to the *k* nearest neighbors, the updated heuristic provides a more localized estimate of the data density, leading to a more appropriate bandwidth. The adaptive restarts for the acquisition function optimization are crucial for balancing exploration and exploitation. In regions where the GP has high uncertainty (high sigma), more restarts are used to explore the space more thoroughly. Conversely, in regions where the GP is confident (low sigma), fewer restarts are used to exploit the current best solution. This adaptive strategy improves the efficiency of the optimization process.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors

class RBF_Bandwidth_BO_v2:
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
        self.knn = NearestNeighbors(n_neighbors=min(10, self.n_init-1)) # Use k-NN for bandwidth update

        self.gp = None
        self.best_y = float('inf')
        self.best_x = None

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
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

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0 # avoid nan values
        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Optimize the acquisition function using multiple random restarts
        best_x = None
        best_ei = -np.inf

        # Adaptive restarts based on GP uncertainty
        _, sigma = self.gp.predict(self.X, return_std=True)
        avg_sigma = np.mean(sigma)
        n_restarts = max(1, int(5 * (avg_sigma / 0.5)))  # Scale restarts based on uncertainty, minimum 1
        n_restarts = min(n_restarts, 10) # Cap the number of restarts

        for _ in range(n_restarts):  # Multiple restarts
            x0 = self._sample_points(1)  # Random initial point
            
            # Define the objective function to minimize (negative EI)
            def objective(x):
                x = x.reshape(1, -1)
                return -self._acquisition_function(x)[0, 0]
            
            # Perform the optimization
            res = minimize(objective, x0, bounds=[(-5, 5)] * self.dim, method='L-BFGS-B')
            
            # Check if this restart found a better solution
            if -res.fun > best_ei:
                best_ei = -res.fun
                best_x = res.x
        
        return best_x.reshape(1, -1)

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
        self.knn.fit(self.X)
        distances, _ = self.knn.kneighbors(self.X)
        distances = distances[:, 1:]  # Exclude the distance to itself
        self.bandwidth = np.median(distances)

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        self.knn.fit(self.X) # Fit k-NN after initial sampling
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

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
 The algorithm RBF_Bandwidth_BO_v2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1508 with standard deviation 0.1035.

took 163.63 seconds to run.