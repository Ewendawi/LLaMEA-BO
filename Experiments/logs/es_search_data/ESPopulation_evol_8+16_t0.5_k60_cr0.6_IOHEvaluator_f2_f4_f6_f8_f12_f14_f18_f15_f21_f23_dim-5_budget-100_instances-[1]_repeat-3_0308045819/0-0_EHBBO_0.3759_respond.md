# Description
**Efficient Hybrid Bayesian Optimization (EHBBO):** This algorithm combines an initial space-filling design using Latin Hypercube Sampling (LHS) with a Gaussian Process (GP) surrogate model and an acquisition function that balances exploration and exploitation. A computationally efficient batch selection strategy based on k-means clustering is used to diversify the search and improve parallelization. The acquisition function is Thompson Sampling, which is known for its efficiency and good performance in high-dimensional spaces.

# Justification
1.  **Initial Space-Filling Design (LHS):** LHS is used to efficiently explore the search space initially, providing a good starting point for the GP model.
2.  **Gaussian Process (GP) Surrogate Model:** GPs are powerful non-parametric models that can capture complex relationships between the input variables and the objective function. They also provide uncertainty estimates, which are crucial for balancing exploration and exploitation.
3.  **Thompson Sampling Acquisition Function:** Thompson Sampling is a probabilistic acquisition function that samples from the posterior distribution of the GP model. It is computationally efficient and has been shown to perform well in high-dimensional spaces.
4.  **k-means Clustering for Batch Selection:** k-means clustering is used to select a diverse batch of points to evaluate in parallel. This helps to avoid evaluating points that are too close to each other, which can waste function evaluations. It promotes exploration by selecting points from different regions of the search space.
5.  **Computational Efficiency:** The algorithm is designed to be computationally efficient by using a relatively small number of initial points, a fast GP implementation, and a simple batch selection strategy.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.cluster import KMeans

class EHBBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(10*dim, self.budget//4) # Number of initial random samples

        # Do not add any other arguments without a default value
        self.gp = None
        self.best_x = None
        self.best_y = float('inf')

    def _sample_points(self, n_points):
        # sample points using Latin Hypercube Sampling
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function: Thompson Sampling
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))  # Return random values if GP is not fitted yet
        else:
            y_samples = self.gp.sample_y(X, n_samples=1)
            return y_samples

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        n_candidates = min(100 * self.dim, self.budget - self.n_evals)
        candidates = self._sample_points(n_candidates)
        acquisition_values = self._acquisition_function(candidates)

        # Cluster the candidates using k-means
        n_clusters = min(batch_size, n_candidates)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(candidates)
        cluster_ids = kmeans.labels_

        # Select the best candidate from each cluster
        next_points = []
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_ids == i)[0]
            best_index = cluster_indices[np.argmin(acquisition_values[cluster_indices])]
            next_points.append(candidates[best_index])

        return np.array(next_points)

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
        batch_size = min(10, self.dim) # Adjust batch size as needed
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
## Feedback
 The algorithm EHBBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1497 with standard deviation 0.0996.

took 4.99 seconds to run.