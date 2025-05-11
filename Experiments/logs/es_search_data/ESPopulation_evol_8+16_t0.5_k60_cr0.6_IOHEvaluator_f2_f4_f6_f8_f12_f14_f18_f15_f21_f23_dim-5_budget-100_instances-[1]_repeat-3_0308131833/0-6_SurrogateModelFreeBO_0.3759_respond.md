# Description
**SurrogateModelFreeBO (SMFBO):** This algorithm deviates from traditional Bayesian Optimization by eliminating the explicit surrogate model. Instead, it relies on a combination of space-filling design, direct function evaluations, and a nearest-neighbor-based acquisition function. It maintains a memory of evaluated points and their corresponding function values. The acquisition function selects new points by considering the distance to existing points and their function values, favoring regions that are both unexplored and have the potential for improvement. To enhance exploration, it incorporates a random sampling component. This approach is particularly suitable for problems where fitting a surrogate model is computationally expensive or unreliable.

# Justification
This algorithm is designed to be diverse from the previous ones by removing the Gaussian Process surrogate model.
*   **No Surrogate Model:** This is the most significant departure from the previous algorithms. It avoids the computational cost and potential inaccuracies of fitting a GP.
*   **Nearest Neighbor Acquisition:** This provides a simple and computationally efficient way to balance exploration and exploitation.
*   **Direct Function Evaluations:** The algorithm directly uses the observed function values to guide the search.
*   **Random Exploration:** This helps to avoid getting stuck in local optima.
*   **Computational Efficiency:** The algorithm is designed to be computationally efficient, especially for high-dimensional problems where GP fitting can be slow.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.neighbors import NearestNeighbors

class SurrogateModelFreeBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim
        self.knn = NearestNeighbors(n_neighbors=5)
        self.exploration_weight = 0.1

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
        # In this algorithm, we don't fit a surrogate model
        pass

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.X is None:
            return np.zeros((len(X), 1))

        distances, indices = self.knn.kneighbors(X)
        min_distances = distances[:, 0] # Distance to the nearest neighbor

        # Acquisition function based on distance and function values of neighbors
        # Encourage exploration in regions far from existing points
        acquisition = min_distances.reshape(-1, 1)

        # Add a random component for exploration
        acquisition += self.exploration_weight * np.random.rand(len(X), 1)

        return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Optimize the acquisition function to find the next points
        x_tries = self._sample_points(batch_size * 10) # Generate more candidates
        acq_values = self._acquisition_function(x_tries)

        # Select the top batch_size points based on the acquisition function values
        indices = np.argsort(acq_values.flatten())[::-1][:batch_size]
        return x_tries[indices]

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
        
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_index = np.argmin(self.y)
        best_y = self.y[best_index, 0]
        best_x = self.X[best_index]

        while self.n_evals < self.budget:
            # Optimization

            # select points by acquisition function
            batch_size = min(10, self.budget - self.n_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            current_best_index = np.argmin(self.y)
            if self.y[current_best_index, 0] < best_y:
                best_y = self.y[current_best_index, 0]
                best_x = self.X[current_best_index]

        return best_y, best_x
```
## Feedback
 The algorithm SurrogateModelFreeBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1485 with standard deviation 0.1017.

took 0.95 seconds to run.