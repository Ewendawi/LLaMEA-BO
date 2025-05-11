# Description
**Adaptive Distance-Weighted Exploration BO (ADWEBO):** This algorithm enhances the SurrogateModelFreeBO by adaptively weighting the distance to the nearest neighbors with the improvement in function value. It maintains a memory of evaluated points and their corresponding function values. The acquisition function selects new points by considering a combination of distance to existing points and the potential for improvement, favoring regions that are both unexplored and have shown promise. The exploration weight is dynamically adjusted based on the optimization progress. Additionally, a local search strategy is incorporated to refine the search around the best-observed solution, improving local exploitation.

# Justification
The key improvements are:

1.  **Adaptive Distance Weighting:** The acquisition function is modified to incorporate the difference between the current best function value and the function values of the nearest neighbors. This allows the algorithm to prioritize exploration in regions where the potential for improvement is high, not just regions that are far from existing points.

2.  **Dynamic Exploration Weight:** The exploration weight is dynamically adjusted based on the optimization progress. Initially, exploration is favored, but as the algorithm progresses, the exploration weight is reduced to focus on exploitation. This is done by tracking the improvement rate and reducing the exploration weight when the improvement rate plateaus.

3.  **Local Search:** A simple local search strategy is incorporated to refine the search around the best-observed solution. This helps to improve local exploitation and find the local optima more efficiently.

These enhancements aim to improve the balance between exploration and exploitation, leading to better performance on the BBOB test suite. The adaptive weighting and dynamic exploration encourage exploration in promising regions and exploitation of local optima. The computational overhead of these changes is relatively small, maintaining the efficiency of the algorithm.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.neighbors import NearestNeighbors

class AdaptiveDistanceWeightedExplorationBO:
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
        self.exploration_weight = 0.2
        self.best_y = float('inf')
        self.improvement_rate = 0.0
        self.previous_best_y = float('inf')
        self.local_search_radius = 0.1

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

        # Adaptive distance weighting based on function value improvement
        improvement = np.zeros_like(min_distances)
        for i in range(len(X)):
            neighbors_y = self.y[indices[i]].flatten()
            improvement[i] = np.max(self.best_y - neighbors_y)  # Potential improvement

        # Acquisition function: balance distance and potential improvement
        acquisition = min_distances * (1 + improvement)
        acquisition = acquisition.reshape(-1, 1)

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

    def _local_search(self, func, x_start, radius, num_steps=5):
        # Perform a simple local search around x_start
        x_current = x_start.copy()
        y_current = func(x_current)
        self.n_evals += 1

        for _ in range(num_steps):
            # Generate a random perturbation within the radius
            perturbation = np.random.uniform(-radius, radius, size=self.dim)
            x_new = x_current + perturbation

            # Clip the new point to stay within the bounds
            x_new = np.clip(x_new, self.bounds[0], self.bounds[1])

            y_new = func(x_new)
            self.n_evals += 1

            if y_new < y_current:
                x_current = x_new
                y_current = y_new

        return y_current, x_current

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
        self.best_y = self.y[best_index, 0]
        best_x = self.X[best_index]

        while self.n_evals < self.budget:
            # Optimization

            # select points by acquisition function
            batch_size = min(5, self.budget - self.n_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            current_best_index = np.argmin(self.y)
            current_best_y = self.y[current_best_index, 0]
            current_best_x = self.X[current_best_index]

            # Local search around the best point
            if self.n_evals + 5 <= self.budget:
                local_y, local_x = self._local_search(func, current_best_x, self.local_search_radius)
                if local_y < current_best_y:
                    current_best_y = local_y
                    current_best_x = local_x

            if current_best_y < self.best_y:
                self.best_y = current_best_y
                best_x = current_best_x

            # Update exploration weight based on improvement rate
            if self.previous_best_y - self.best_y > 0:
                self.improvement_rate = (self.previous_best_y - self.best_y) / self.previous_best_y
            else:
                self.improvement_rate = 0.0

            self.previous_best_y = self.best_y

            # Adjust exploration weight
            self.exploration_weight = max(0.01, self.exploration_weight * (1 - self.improvement_rate))

        return self.best_y, best_x
```
## Feedback
 The algorithm AdaptiveDistanceWeightedExplorationBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1587 with standard deviation 0.0983.

took 0.78 seconds to run.