# Description
**Adaptive Hybrid Bayesian Optimization with Dynamic Exploration, Adaptive Batch Size, and Distance-Aware Uncertainty (AHBBO-ABSDU):** This algorithm combines the strengths of AHBBO and AHBBO_ABS while introducing a more refined uncertainty measure that considers the distance to existing points. It adaptively adjusts the batch size based on model uncertainty, promoting exploration when uncertainty is high and exploitation when uncertainty is low. The exploration weight is also dynamically adjusted. The key improvement lies in using a distance-aware uncertainty measure to better estimate the true uncertainty, especially in regions far from existing data points. This helps in making more informed decisions about batch size and exploration-exploitation trade-off.

# Justification
This algorithm builds upon the existing AHBBO and AHBBO_ABS algorithms. AHBBO_ABS already improves upon AHBBO by introducing an adaptive batch size strategy based on uncertainty. However, a simple average of the sigma values might not accurately represent the uncertainty, especially in regions far from the evaluated points. To address this, the new algorithm introduces a distance-aware uncertainty measure. This measure combines the predicted variance from the GPR model with a term that increases with the distance to the nearest evaluated point. This allows the algorithm to be more confident in regions close to existing data and more exploratory in regions far away. The adaptive batch size and exploration weight are then adjusted based on this distance-aware uncertainty. The selection of the next points uses the same hybrid acquisition function as AHBBO, balancing Expected Improvement and distance-based exploration. This combination of adaptive batch size, dynamic exploration, and distance-aware uncertainty should lead to a more robust and efficient optimization algorithm.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.spatial.distance import cdist

class AHBBO_ABSDU:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim # Initial number of points

        self.best_y = np.inf
        self.best_x = None

        self.max_batch_size = min(10, dim) # Maximum batch size for selecting points
        self.min_batch_size = 1
        self.exploration_weight = 0.1 # Initial exploration weight
        self.exploration_decay = 0.995 # Decay factor for exploration weight
        self.min_exploration = 0.01 # Minimum exploration weight
        self.uncertainty_threshold = 0.5  # Threshold for adjusting batch size
        self.distance_scale = 1.0 # Scale for distance-based uncertainty

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
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Distance-based exploration term
        min_dist = np.min(np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2), axis=1, keepdims=True)
        exploration = min_dist / np.max(min_dist)

        # Hybrid acquisition function
        acquisition = ei + self.exploration_weight * exploration
        return acquisition

    def _distance_aware_uncertainty(self, X):
        # Calculate distance-aware uncertainty
        _, sigma = self.model.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        
        # Calculate minimum distance to existing points
        distances = cdist(X, self.X)
        min_distances = np.min(distances, axis=1, keepdims=True)
        
        # Combine variance and distance
        uncertainty = sigma + self.distance_scale * min_distances
        return np.mean(uncertainty)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Generate candidate points
        candidate_points = self._sample_points(100 * batch_size)  # Generate more candidates
        
        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_points)
        
        # Select the top batch_size points with the highest acquisition values
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]
        
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
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Optimization
            # select points by acquisition function
            
            # Adjust batch size based on uncertainty
            avg_sigma = self._distance_aware_uncertainty(self.X)
            
            if avg_sigma > self.uncertainty_threshold:
                batch_size = self.max_batch_size
                exploration_decay = 0.99
            else:
                batch_size = self.min_batch_size
                exploration_decay = 0.999
            
            remaining_evals = self.budget - self.n_evals
            batch_size = min(batch_size, remaining_evals) # Adjust batch size to budget
            
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self.model = self._fit_model(self.X, self.y)
            
            # Update exploration weight
            self.exploration_weight = max(self.exploration_weight * exploration_decay, self.min_exploration)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AHBBO_ABSDU got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1612 with standard deviation 0.0984.

took 199.02 seconds to run.