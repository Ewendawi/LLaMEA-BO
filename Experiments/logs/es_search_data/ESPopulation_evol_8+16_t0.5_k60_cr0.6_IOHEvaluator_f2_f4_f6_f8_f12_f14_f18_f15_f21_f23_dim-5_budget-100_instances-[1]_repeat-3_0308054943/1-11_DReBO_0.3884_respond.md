# Description
**DReBO: Diversity Regularized Bayesian Optimization:** This algorithm combines the diversity-enhancing features of DEBO with the regularization strategy of ReBO. It employs a Gaussian Process Regression (GPR) model with an RBF kernel for surrogate modeling. The acquisition function incorporates both a distance-based diversity term (like DEBO) and an adaptive regularization term based on the L2 norm of the GPR's predicted mean (like ReBO). The diversity term encourages exploration of unexplored regions, while the regularization term prevents overfitting. The initial exploration is performed using a Sobol sequence (like DEBO) to ensure good coverage of the search space. A clustering-based approach is used to guide the sampling of diverse regions, similar to DEBO, but with a modification to sample more points from the cluster with the best-observed function value. This aims to balance exploration and exploitation more effectively.

# Justification
The key idea is to combine the strengths of DEBO and ReBO to achieve a better balance between exploration and exploitation. DEBO's diversity term helps prevent premature convergence to local optima, while ReBO's regularization term reduces the risk of overfitting the GPR model, especially in high-dimensional spaces.

1.  **Diversity and Regularization:** Combining the diversity term and regularization term in the acquisition function allows the algorithm to simultaneously explore diverse regions and avoid overfitting.
2.  **Adaptive Regularization:** The regularization weight is adapted during the optimization process, gradually decreasing as the algorithm progresses. This allows for more exploration in the initial stages and more exploitation in the later stages.
3.  **Sobol Sequence for Initial Exploration:** Sobol sequence provides a more uniform coverage of the search space compared to Latin Hypercube Sampling, especially in higher dimensions.
4.  **Clustering-Guided Sampling with Exploitation:** Modifying the cluster-guided sampling to sample more points from the cluster containing the best-observed function value allows the algorithm to focus on promising regions while still maintaining diversity.
5.  **Computational Efficiency:** The algorithm is designed to be computationally efficient by using relatively simple and fast techniques for diversity enhancement and regularization.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

class DReBO:
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
        self.diversity_weight = 0.1 # Weight for the diversity term in the acquisition function
        self.reg_weight = 0.1 # Initial weight for the regularization term
        self.n_clusters = 5 # Number of clusters for region identification

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
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X, model, iteration):
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

        # Add diversity term
        if self.X is not None:
            distances = pairwise_distances(X, self.X)
            min_distances = np.min(distances, axis=1).reshape(-1, 1)
            ei = ei + self.diversity_weight * min_distances
        
        # Adaptive Regularization
        reg_weight = self.reg_weight * (1 - (iteration / self.budget))
        regularization_term = -reg_weight * np.linalg.norm(mu, axis=1, keepdims=True)**2
        ei = ei + regularization_term

        return ei

    def _select_next_points(self, batch_size, model, iteration):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # return array of shape (batch_size, n_dims)
        
        # Identify diverse regions using clustering
        if self.X is not None:
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.X)
            cluster_centers = kmeans.cluster_centers_

            # Find the cluster with the best-observed function value
            best_cluster = cluster_labels[np.argmin(self.y)]
        else:
            cluster_centers = self._sample_points(self.n_clusters)
            best_cluster = None

        # Sample candidate points from each cluster
        candidate_points = []
        for i, center in enumerate(cluster_centers):
            # Sample more points from the best cluster
            if self.X is not None and i == best_cluster:
                num_points = 200
            else:
                num_points = 100

            # Sample points around the cluster center
            candidates = np.random.normal(loc=center, scale=0.5, size=(num_points, self.dim))
            candidates = np.clip(candidates, self.bounds[0], self.bounds[1])
            candidate_points.extend(candidates)
        candidate_points = np.array(candidate_points)

        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_points, model, iteration)

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
        iteration = self.n_init
        while self.n_evals < self.budget:
            # Optimization
            model = self._fit_model(self.X, self.y)
            
            # select points by acquisition function
            next_X = self._select_next_points(batch_size, model, iteration)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            iteration += batch_size

        return self.best_y, self.best_x
```
## Feedback
 The algorithm DReBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1616 with standard deviation 0.0976.

took 65.16 seconds to run.