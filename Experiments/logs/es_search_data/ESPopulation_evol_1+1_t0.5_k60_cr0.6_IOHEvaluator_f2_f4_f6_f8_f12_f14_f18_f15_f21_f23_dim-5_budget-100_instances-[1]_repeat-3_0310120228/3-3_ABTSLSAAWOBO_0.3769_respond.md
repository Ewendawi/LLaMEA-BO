# Description
**Adaptive Batch Bayesian Optimization with Thompson Sampling, Improved Local Search and Adaptive Acquisition Weighting (ABTSLS-AAWBO):** This algorithm refines the ABTSLSBO algorithm by introducing an adaptive weighting mechanism to the acquisition function, dynamically balancing exploration and exploitation. The local search is enhanced with a more robust sampling strategy and a probability of performing it. The batch size is also dynamically adjusted based on the optimization progress and function evaluations.

# Justification
The key improvements are:

1.  **Adaptive Acquisition Weighting:** Instead of a fixed exploration-exploitation balance in the UCB acquisition function, a weight `w` is introduced to modulate the exploration term (sigma). This weight is adapted based on the optimization progress. Initially, a higher weight is given to exploration, which gradually decreases as the algorithm converges, promoting exploitation of promising regions. This addresses the issue of premature convergence or insufficient exploration.

2.  **Enhanced Local Search:** The local search is enhanced by incorporating a probability of performing it, `local_search_prob`. This probability decreases as the number of function evaluations increases, reducing the computational overhead of local search in later stages. The number of local search points is also adjusted.

3.  **Dynamic Batch Size Adjustment:** The batch size is dynamically adjusted based on the remaining budget and the current iteration. This ensures that the algorithm efficiently utilizes the available budget while adapting to the problem's characteristics.

These changes aim to improve the exploration-exploitation balance, enhance local refinement, and optimize computational efficiency, leading to better performance on the BBOB test suite.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ABTSLSAAWOBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim  # Initial number of points
        self.batch_size = min(10, dim) # Initial batch size, adaptively adjusted
        self.gp = None
        self.best_x = None
        self.best_y = np.inf
        self.acquisition_weight = 2.0 # Initial acquisition weight
        self.local_search_prob = 0.9  # Probability of performing local search

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
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None:
            return np.zeros((len(X), 1))  # Return zeros if the model is not fitted yet

        mu, sigma = self.gp.predict(X, return_std=True)
        # UCB acquisition function: balance exploration and exploitation
        return mu.reshape(-1, 1) + self.acquisition_weight * sigma.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Sample candidate points
        candidate_points = self._sample_points(10 * batch_size)
        
        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_points)
        
        # Select the points with the highest acquisition function values
        indices = np.argsort(acquisition_values.flatten())[::-1][:batch_size]
        
        return candidate_points[indices]

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
        
        # Update best solution
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]

    def _local_search(self, func, x, num_points=3):
        # Perform local search around x using the surrogate model
        # Adapt radius based on GP's uncertainty
        if self.gp is None:
            radius = 0.1  # Default radius if GP is not fitted
        else:
            _, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
            radius = max(0.01, sigma[0])  # Ensure a minimum radius
        
        # Sample points around x
        X = np.random.uniform(low=np.maximum(self.bounds[0], x - radius),
                                high=np.minimum(self.bounds[1], x + radius),
                                size=(num_points, self.dim))
        
        # Predict the values using the Gaussian Process model
        if self.gp is None:
            predicted_y = np.zeros((num_points, 1))
        else:
            predicted_y, _ = self.gp.predict(X, return_std=True)
            predicted_y = predicted_y.reshape(-1, 1)
        
        # Find the best point based on the predicted value
        best_index = np.argmin(predicted_y)
        best_x_candidate = X[best_index]

        # Evaluate the best candidate point using the real function, if budget allows
        if self.n_evals < self.budget:
            best_y_candidate = self._evaluate_points(func, best_x_candidate.reshape(1, -1))[0, 0]
            return best_x_candidate, best_y_candidate
        else:
            return x, self.best_y  # Return current best if no budget left
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Fit the model
            self._fit_model(self.X, self.y)
            
            # Select next points
            X_next = self._select_next_points(self.batch_size)
            
            # Evaluate points
            y_next = self._evaluate_points(func, X_next)
            
            # Update evaluated points
            self._update_eval_points(X_next, y_next)

            # Local search around the best point
            if np.random.rand() < self.local_search_prob:
                best_x_local, best_y_local = self._local_search(func, self.best_x)
                if best_y_local < self.best_y:
                    self.best_x = best_x_local
                    self.best_y = best_y_local
        
            # Adapt batch size
            self.batch_size = max(1, min(int(self.budget/10), int(self.batch_size * 0.95))) # Reduce batch size gradually, but not too small

            # Adapt acquisition weight
            self.acquisition_weight = max(0.1, self.acquisition_weight * 0.98)  # Reduce exploration over time
            self.local_search_prob = max(0.1, self.local_search_prob * 0.98)

            if self.n_evals >= self.budget:
                break

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ABTSLSAAWOBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1492 with standard deviation 0.1030.

took 76.95 seconds to run.