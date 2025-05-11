# Description
**EATRBO-V2: Enhanced Adaptive Trust Region Bayesian Optimization with Volume-Aware Exploration.** This algorithm refines EATRBO by incorporating a volume-aware exploration strategy into the acquisition function. It maintains the adaptive trust region and batch size mechanisms of EATRBO, while enhancing the exploration-exploitation balance by considering the volume of the search space that is far from existing data points. This encourages exploration in potentially promising, yet unexplored regions.

# Justification
The key improvements focus on the acquisition function:

1.  **Volume-Aware Exploration:** The original EATRBO used a simple minimum distance to encourage diversity. EATRBO-V2 incorporates a term that considers the "volume" of unexplored space around a candidate point. This is achieved by calculating the average distance to the k-nearest neighbors in the existing data. A larger average distance indicates a more isolated point, and thus a higher potential for exploration. This volume-aware term is added to the LCB acquisition function, promoting exploration in less-visited regions.

2.  **Adaptive Exploration Factor:** The exploration factor is adaptively adjusted based on both the remaining budget and the diversity of the evaluated points. A higher diversity (measured by the average distance between points) leads to a lower exploration factor, favoring exploitation. Conversely, a lower diversity increases the exploration factor.

3.  **Trust Region Refinement**: The trust region adaptation is refined by considering the uncertainty of the GP model in addition to the prediction error. If the model has high uncertainty in the region, the trust region is increased to allow for more exploration.

These changes aim to improve the algorithm's ability to escape local optima and converge to the global optimum more effectively, especially in high-dimensional and complex search spaces.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

class EATRBO_V2:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * self.dim # Initial samples
        self.trust_region_size = 2.0  # Initial trust region size
        self.exploration_factor = 2.0 # Initial exploration factor
        self.epsilon = 1e-6
        self.knn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree')


        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        samples = sampler.random(n=n_points)
        return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, nu=2.5)
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

        # Lower Confidence Bound
        lcb = mu - self.exploration_factor * sigma

        # Volume-aware exploration
        if self.X is not None:
            distances, _ = self.knn.kneighbors(X)
            avg_distances = np.mean(distances, axis=1).reshape(-1, 1)
            lcb -= 0.01 * self.exploration_factor * avg_distances

        return lcb

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Optimize the acquisition function within the trust region using L-BFGS-B
        best_idx = np.argmin(self.y)
        best_x = self.X[best_idx]
        
        x_starts = best_x + np.random.normal(0, 0.1, size=(batch_size, self.dim))
        x_starts = np.clip(x_starts, self.bounds[0], self.bounds[1])

        candidates = []
        values = []
        for x_start in x_starts:
            # Define trust region bounds
            lower_bound = np.maximum(x_start - self.trust_region_size / 2, self.bounds[0])
            upper_bound = np.minimum(x_start + self.trust_region_size / 2, self.bounds[1])
            
            res = minimize(lambda x: self._acquisition_function(x.reshape(1, -1)),
                           x_start,
                           bounds=np.array([lower_bound, upper_bound]).T,
                           method="L-BFGS-B")
            candidates.append(res.x)
            values.append(res.fun)
        
        return np.array(candidates)

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
        self.knn.fit(self.X)
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        self.model = self._fit_model(self.X, self.y)
        
        while self.n_evals < self.budget:
            # Adaptive batch size
            batch_size = min(int(np.ceil(self.trust_region_size)), 4)  # Adjust batch size based on trust region
            batch_size = max(1, batch_size) # Ensure batch size is at least 1

            # Optimization
            X_next = self._select_next_points(batch_size)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

            # Adaptive trust region adjustment
            y_pred, sigma = self.model.predict(X_next, return_std=True)
            y_pred = y_pred.reshape(-1, 1)
            sigma = sigma.reshape(-1, 1)

            # Agreement between prediction and actual value
            agreement = np.abs(y_pred - y_next) / (sigma.reshape(-1, 1) + self.epsilon)

            if np.mean(agreement) < 1.0:
                self.trust_region_size *= 1.1  # Increase trust region if model is accurate
            else:
                self.trust_region_size *= 0.9  # Decrease trust region if model is inaccurate
            
            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0) # Clip trust region size

            # Dynamic exploration factor adjustment
            diversity = 0
            if self.X is not None and len(self.X) > 1:
                distances = cdist(self.X, self.X)
                diversity = np.mean(np.min(distances + np.eye(len(self.X)) * 1000, axis=1))  # Avoid distance to self
            
            self.exploration_factor = 1.0 + (self.budget - self.n_evals) / self.budget + 0.5*(1-diversity/5)
            self.exploration_factor = max(0.1, self.exploration_factor) # Ensure exploration factor is at least 0.1
            
            self.model = self._fit_model(self.X, self.y) # Refit the model with new data
            
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        return best_y, best_x
```
## Feedback
 The algorithm EATRBO_V2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1824 with standard deviation 0.1032.

took 436.45 seconds to run.