# Description
**Adaptive Hybrid Bayesian Optimization with Dynamic Exploration and Local Refinement (AHBBODELR):** This algorithm combines the strengths of AEEHBBO and AHBBO by incorporating adaptive exploration-exploitation balancing with dynamic exploration and local refinement using a combination of Gaussian process regression (GPR) and a distance-based exploration term. The exploration weight is dynamically adjusted based on the optimization progress, decreasing as the number of evaluations increases, but with a lower bound to prevent premature convergence, similar to AHBBO. It refines the local search strategy of AEEHBBO by adaptively adjusting the scale of the local search based on the GPR's uncertainty estimates. Furthermore, it introduces a dynamic elitist local refinement strategy where the best few points are refined using local search with a scale proportional to the GPR's uncertainty in those regions.

# Justification
The key components are justified as follows:

1.  **Adaptive Exploration-Exploitation Balancing:** Inspired by AEEHBBO and AHBBO, this allows the algorithm to explore the search space effectively in the early stages and exploit promising regions later on. The exploration weight is decayed as the optimization progresses, but a minimum exploration weight is maintained.
2.  **Hybrid Acquisition Function:** The acquisition function balances Expected Improvement (EI) and a distance-based exploration term. EI focuses on exploitation, while the distance-based term encourages exploration of regions far from previously evaluated points.
3.  **Dynamic Exploration Weight Decay:** The exploration weight is decayed based on the number of evaluations, similar to AHBBO and AEEHBBO. This allows for a smooth transition from exploration to exploitation.
4.  **Adaptive Local Refinement:** The scale of the local search around the best point is adjusted based on the GPR's uncertainty estimates. This allows the algorithm to perform more focused local search in regions where the GPR is less confident.
5.  **Dynamic Elitist Local Refinement:** Refines the best few points found so far using local search with a scale proportional to the GPR's uncertainty. This helps to escape local optima and improve the overall performance. This helps to refine the best solutions.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class AHBBODELR:
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

        self.batch_size = min(10, dim) # Batch size for selecting points
        self.exploration_weight = 0.2 # Initial exploration weight
        self.exploration_decay = 0.995 # Decay factor for exploration weight
        self.min_exploration = 0.01 # Minimum exploration weight
        self.local_search_scale = 0.1
        self.n_elite = 3

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
        if self.X is not None:
            min_dist = np.min(np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2), axis=1, keepdims=True)
            exploration = min_dist / np.max(min_dist)
        else:
            exploration = np.ones(X.shape[0])[:,None]

        # Hybrid acquisition function
        acquisition = ei + self.exploration_weight * exploration
        return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Generate candidate points
        candidate_points = self._sample_points(50 * batch_size)  # Generate more candidates

        # Add points around the best solution (local search)
        if self.best_x is not None:
            _, best_sigma = self.model.predict(self.best_x.reshape(1, -1), return_std=True)
            local_scale = max(self.local_search_scale, best_sigma) # Adaptive local search scale
            local_points = np.random.normal(loc=self.best_x, scale=local_scale, size=(50 * batch_size, self.dim))
            local_points = np.clip(local_points, self.bounds[0], self.bounds[1])
            candidate_points = np.vstack((candidate_points, local_points))
        
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
    
    def _dynamic_elitist_local_refinement(self, func):
        # Refine the best few points using local search with adaptive scale
        num_refine = min(self.n_elite, len(self.X))
        indices = np.argsort(self.y.flatten())[:num_refine]
        elite_points = self.X[indices]

        for i, elite_point in enumerate(elite_points):
            _, sigma = self.model.predict(elite_point.reshape(1, -1), return_std=True)
            local_scale = max(self.local_search_scale, sigma)
            new_point = np.random.normal(loc=elite_point, scale=local_scale, size=self.dim)
            new_point = np.clip(new_point, self.bounds[0], self.bounds[1])
            new_y = self._evaluate_points(func, new_point.reshape(1, -1))
            
            if new_y[0, 0] < self.y[indices[i], 0]:
                self.X[indices[i]] = new_point
                self.y[indices[i]] = new_y[0, 0]
                if new_y[0, 0] < self.best_y:
                    self.best_y = new_y[0, 0]
                    self.best_x = new_point
                    
    
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
            
            remaining_evals = self.budget - self.n_evals
            batch_size = min(self.batch_size, remaining_evals) # Adjust batch size to budget
            
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self.model = self._fit_model(self.X, self.y)

            # Update exploration weight
            self.exploration_weight = max(self.exploration_weight * self.exploration_decay, self.min_exploration)
            
            # Dynamic elitist local refinement
            self._dynamic_elitist_local_refinement(func)
            self.model = self._fit_model(self.X, self.y) # Refit model after refinement

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<AHBBODELR>", line 174, in __call__
 174->             self._dynamic_elitist_local_refinement(func)
  File "<AHBBODELR>", line 135, in _dynamic_elitist_local_refinement
 135->             new_y = self._evaluate_points(func, new_point.reshape(1, -1))
  File "<AHBBODELR>", line 104, in _evaluate_points
 104->         y = np.array([func(x) for x in X])
  File "<AHBBODELR>", line 104, in <listcomp>
 102 |         # return array of shape (n_points, 1)
 103 |         
 104->         y = np.array([func(x) for x in X])
 105 |         self.n_evals += len(X)
 106 |         return y.reshape(-1, 1)
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
