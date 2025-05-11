# Description
**Adaptive Diversity Enhanced Bayesian Optimization (ADEBO):** This algorithm refines DEBO by incorporating adaptive diversity weighting and dynamic cluster adjustment. The diversity weight in the acquisition function is adjusted based on the optimization progress, promoting exploration early on and exploitation later. The number of clusters used for identifying diverse regions is also dynamically adjusted based on the distribution of sampled points to better capture the search space structure. A Matern kernel is used for the Gaussian Process Regression (GPR) model to better handle non-smooth functions.

# Justification
The key improvements are:

1.  **Adaptive Diversity Weight:** The original DEBO used a fixed diversity weight. However, balancing exploration and exploitation requires adjusting this weight during the optimization process. Initially, a higher diversity weight is beneficial for exploring the search space. As the optimization progresses and the algorithm converges towards promising regions, the diversity weight should be reduced to focus on exploitation within those regions. This adaptive strategy allows for a more efficient search. The diversity weight is updated based on the current iteration number and the maximum number of iterations.
2.  **Dynamic Cluster Adjustment:** The number of clusters used for identifying diverse regions was fixed. However, the optimal number of clusters can vary depending on the function being optimized and the current sampling density. By dynamically adjusting the number of clusters based on the number of evaluations performed so far, the algorithm can better adapt to the search space structure. The number of clusters is increased gradually as more evaluations are performed, allowing for a more refined exploration of the search space.
3. **Matern Kernel:** The original DEBO used an RBF kernel. Matern kernel is more flexible and can better handle non-smooth functions, which are common in the BBOB suite.

These changes aim to improve the exploration-exploitation balance and adapt to the characteristics of different optimization problems, leading to better performance.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

class ADEBO:
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
        self.initial_diversity_weight = 0.2 # Initial weight for the diversity term
        self.final_diversity_weight = 0.01 # Final weight for the diversity term
        self.n_clusters = 2 # Initial number of clusters
        self.max_clusters = 10 # Maximum number of clusters
        self.cluster_update_frequency = 10 # update cluster every n iterations

        # Do not add any other arguments without a default value
        self.best_x = None
        self.best_y = float('inf')

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.Sobol(d=self.dim, seed=42)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X, model):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if self.best_y is None:
            ei = np.zeros_like(mu)
        else:
            imp = self.best_y - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0  # handle zero sigma

        # Add diversity term with adaptive weight
        diversity_weight = self.initial_diversity_weight + (self.final_diversity_weight - self.initial_diversity_weight) * (self.n_evals / self.budget)
        if self.X is not None:
            distances = pairwise_distances(X, self.X)
            min_distances = np.min(distances, axis=1).reshape(-1, 1)
            ei = ei + diversity_weight * min_distances

        return ei

    def _select_next_points(self, batch_size, model):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Identify diverse regions using clustering
        if self.X is not None:
            # Update the number of clusters dynamically
            self.n_clusters = min(self.max_clusters, 2 + self.n_evals // self.cluster_update_frequency)
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.X)
            cluster_centers = kmeans.cluster_centers_
        else:
            cluster_centers = self._sample_points(self.n_clusters)

        # Sample candidate points from each cluster
        candidate_points = []
        for center in cluster_centers:
            # Sample points around the cluster center
            candidates = np.random.normal(loc=center, scale=0.5, size=(100, self.dim))
            candidates = np.clip(candidates, self.bounds[0], self.bounds[1])
            candidate_points.extend(candidates)
        candidate_points = np.array(candidate_points)

        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_points, model)

        # Select top batch_size points based on acquisition function values
        indices = np.argsort(-acquisition_values.flatten())[:batch_size]
        selected_points = candidate_points[indices]

        return selected_points

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
        
        batch_size = 5
        while self.n_evals < self.budget:
            # Optimization
            model = self._fit_model(self.X, self.y)
            
            # select points by acquisition function
            next_X = self._select_next_points(batch_size, model)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ADEBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1619 with standard deviation 0.1033.

took 77.21 seconds to run.