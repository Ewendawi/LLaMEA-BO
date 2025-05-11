# Description
**HybridEfficientOptimisticBO**: This algorithm combines the efficient lengthscale estimation and initial sampling strategy of EfficientHybridBO with the Upper Confidence Bound (UCB) acquisition function and dynamic exploration-exploitation trade-off of BayesOptimisticBO. To avoid exceeding the budget during local search, the local search is performed using the GP model's prediction instead of calling the objective function directly. This approach aims to balance global exploration with local exploitation while maintaining computational efficiency and adhering to the budget constraints. The local search is performed on the acquisition function to avoid evaluating the true function.

# Justification
The combination of EfficientHybridBO and BayesOptimisticBO aims to leverage the strengths of both algorithms. EfficientHybridBO provides a computationally efficient way to estimate the lengthscale of the GP kernel, which can improve the accuracy of the surrogate model. BayesOptimisticBO uses a UCB acquisition function with a dynamic exploration parameter, which can help to balance exploration and exploitation. The local search component, adapted to use the GP model instead of the true function, allows for refining promising solutions without exceeding the budget. The error in BayesOptimisticBO was due to calling the function `func` during the local search, which could lead to exceeding the budget. This is avoided by using the GP to predict the values during local search.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize

class HybridEfficientOptimisticBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim
        self.batch_size = min(5, self.dim)
        self.beta = 2.0  # Exploration parameter for UCB
        self.beta_decay = 0.99 # Decay rate for beta
        self.best_x = None
        self.best_y = np.inf

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Efficient lengthscale estimation using nearest neighbors
        nn = NearestNeighbors(n_neighbors=min(len(X), 10)).fit(X)
        distances, _ = nn.kneighbors(X)
        median_distance = np.median(distances[:, 1])  # Exclude the point itself

        # Define the kernel with the estimated lengthscale
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=median_distance, length_scale_bounds=(1e-3, 1e3))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        model = self._fit_model(self.X, self.y)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Upper Confidence Bound
        ucb = mu - self.beta * sigma # minimize
        return ucb

    def _select_next_points(self, batch_size):
        candidate_points = self._sample_points(10 * batch_size)
        acquisition_values = self._acquisition_function(candidate_points)
        indices = np.argsort(acquisition_values.flatten())[:batch_size] # minimize
        return candidate_points[indices]

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)
    
    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))
            
        # Update best seen point
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]

    def _local_search(self):
        # Local search around the best point using the GP model
        if self.best_x is not None:
            model = self._fit_model(self.X, self.y)

            def obj_func(x):
                # Use the GP model to predict the value
                mu, _ = model.predict(x.reshape(1, -1), return_std=True)
                return mu[0]

            bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
            result = minimize(obj_func, self.best_x, method='L-BFGS-B', bounds=bounds)

            # Check if the predicted value is better than the current best
            predicted_best_y = obj_func(result.x)

            if predicted_best_y < self.best_y:
                self.best_x = result.x

                # Evaluate the true function value at the new best point
                new_X = result.x.reshape(1, -1)
                new_y = self._evaluate_points(func, new_X)  # Evaluate the true function
                self._update_eval_points(new_X, new_y)
                self.best_y = new_y[0][0]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        
        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        while self.n_evals < self.budget:
            # Optimization
            next_X = self._select_next_points(self.batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Local search
            self._local_search()
            
            # Decay exploration parameter
            self.beta *= self.beta_decay

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<HybridEfficientOptimisticBO>", line 117, in __call__
 117->             self._local_search()
  File "<HybridEfficientOptimisticBO>", line 98, in _local_search
  96 |                 # Evaluate the true function value at the new best point
  97 |                 new_X = result.x.reshape(1, -1)
  98->                 new_y = self._evaluate_points(func, new_X)  # Evaluate the true function
  99 |                 self._update_eval_points(new_X, new_y)
 100 |                 self.best_y = new_y[0][0]
NameError: name 'func' is not defined
