# Description
**ContextAwareBO (CABO):** This algorithm introduces a context-aware approach to Bayesian Optimization by incorporating information about the local landscape around previously evaluated points. It uses a Gaussian Process (GP) surrogate model with a modified acquisition function that combines Expected Improvement (EI) with a context-aware term. This term penalizes points that are too close to existing points in regions where the GP model has high confidence (low uncertainty), encouraging exploration of less-explored areas. To estimate the local landscape, the algorithm uses a simple k-nearest neighbors (k-NN) approach. The initial sampling is done using a Sobol sequence to ensure good space coverage.

# Justification
This algorithm aims to address the limitations of standard Bayesian Optimization methods, which can sometimes get stuck in local optima or fail to adequately explore the search space. The context-aware term in the acquisition function encourages exploration by penalizing points that are too similar to existing points, especially in regions where the GP model is already confident. This helps to diversify the search and avoid premature convergence. The use of a Sobol sequence for initial sampling ensures good space coverage and reduces the risk of missing important regions of the search space. The k-NN approach is computationally efficient and provides a simple way to estimate the local landscape. This algorithm is diverse from the existing ones by using a context-aware acquisition function and a k-NN approach to estimate the local landscape. It also uses a Sobol sequence for initial sampling, which is different from the Latin Hypercube Sampling used in the other algorithms.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors

class ContextAwareBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(5*dim, self.budget//10)

        # Do not add any other arguments without a default value
        self.gp = None
        self.best_x = None
        self.best_y = float('inf')
        self.knn = NearestNeighbors(n_neighbors=5) # k-NN for context awareness
        self.context_penalty = 0.1 # Weight for context penalty in acquisition function

    def _sample_points(self, n_points):
        # sample points using Sobol sequence
        # return array of shape (n_points, n_dims)
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        self.gp.fit(X, y)
        self.knn.fit(X) # Update k-NN model with new data
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function: Expected Improvement + Context Awareness
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            mu, sigma = self.gp.predict(X, return_std=True)
            sigma = np.clip(sigma, 1e-9, np.inf)
            imp = self.best_y - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

            # Context-aware penalty: penalize points close to existing points
            distances, _ = self.knn.kneighbors(X)
            context_penalty = np.mean(distances, axis=1).reshape(-1, 1)
            acquisition = ei - self.context_penalty * sigma * context_penalty # Penalize by distance and uncertainty
            return acquisition.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        n_candidates = min(100 * self.dim, self.budget - self.n_evals)
        candidates = self._sample_points(n_candidates)
        acquisition_values = self._acquisition_function(candidates)
        best_indices = np.argsort(acquisition_values.flatten())[-batch_size:]  # Select top batch_size points
        return candidates[best_indices]

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

        # Update best seen value
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]

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
        batch_size = min(1, self.dim)
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Select points by acquisition function
            next_X = self._select_next_points(batch_size)

            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)

            # Update the evaluated points
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<ContextAwareBO>", line 118, in __call__
 118->             next_X = self._select_next_points(batch_size)
  File "<ContextAwareBO>", line 74, in _select_next_points
  72 |         acquisition_values = self._acquisition_function(candidates)
  73 |         best_indices = np.argsort(acquisition_values.flatten())[-batch_size:]  # Select top batch_size points
  74->         return candidates[best_indices]
  75 | 
  76 |     def _evaluate_points(self, func, X):
IndexError: index 445 is out of bounds for axis 0 with size 90
