# Description
**Adaptive Diversity Enhanced Bayesian Optimization (ADEBO):** This algorithm refines DEBO by adaptively adjusting the diversity weight and the number of clusters based on the optimization progress. It also incorporates a more robust sampling strategy within each cluster using Gaussian Mixture Models (GMM) to better represent the potential distribution of promising points. Furthermore, it employs a dynamic batch size to balance exploration and exploitation.

# Justification
The key improvements are:

1.  **Adaptive Diversity Weight:** The original DEBO uses a fixed `diversity_weight`. However, the importance of diversity changes over the optimization process. Initially, exploration is crucial, so a higher diversity weight is beneficial. As the algorithm converges, exploitation becomes more important, and the diversity weight should decrease. The adaptive diversity weight is calculated based on the current iteration and the total budget, linearly decreasing from an initial value to a final value.

2.  **Adaptive Number of Clusters:** The number of clusters in the original DEBO is fixed. However, a fixed number of clusters might not be optimal for all stages of the optimization. The adaptive number of clusters is calculated based on the number of evaluations performed, increasing from a minimum value to a maximum value as the algorithm progresses.

3. **Gaussian Mixture Model (GMM) Sampling:** Instead of sampling points uniformly around the cluster center using a normal distribution, GMM is used to model the distribution of points within each cluster. This allows the algorithm to sample points more intelligently, focusing on regions within the cluster that are more likely to be promising. This provides a better representation of the data and potentially leads to better exploration.

4. **Dynamic Batch Size:** The batch size is dynamically adjusted based on the iteration number. Initially, a larger batch size is used to promote exploration, and as the algorithm progresses, the batch size is reduced to focus on exploitation.

These changes aim to improve the balance between exploration and exploitation, leading to better performance across a wider range of optimization problems.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances

class ADEBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.initial_diversity_weight = 0.5
        self.final_diversity_weight = 0.01
        self.min_clusters = 2
        self.max_clusters = 10
        self.gmm_n_components = 3 # Number of components in GMM

        self.best_x = None
        self.best_y = float('inf')

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, seed=42)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _adaptive_diversity_weight(self):
        # Linearly decrease diversity weight from initial to final value
        return self.initial_diversity_weight + (self.final_diversity_weight - self.initial_diversity_weight) * (self.n_evals / self.budget)

    def _adaptive_n_clusters(self):
        # Linearly increase the number of clusters
        return min(self.min_clusters + int((self.max_clusters - self.min_clusters) * (self.n_evals / self.budget)), self.max_clusters)

    def _acquisition_function(self, X, model):
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if self.best_y is None:
            ei = np.zeros_like(mu)
        else:
            imp = self.best_y - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0

        # Add diversity term
        if self.X is not None:
            distances = pairwise_distances(X, self.X)
            min_distances = np.min(distances, axis=1).reshape(-1, 1)
            diversity_weight = self._adaptive_diversity_weight()
            ei = ei + diversity_weight * min_distances

        return ei

    def _select_next_points(self, batch_size, model):
        n_clusters = self._adaptive_n_clusters()

        if self.X is not None:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.X)
            cluster_centers = kmeans.cluster_centers_
        else:
            cluster_centers = self._sample_points(n_clusters)

        candidate_points = []
        for i in range(n_clusters):
            if self.X is not None:
                cluster_data = self.X[cluster_labels == i]
                if len(cluster_data) > 1:
                    gmm = GaussianMixture(n_components=min(self.gmm_n_components, len(cluster_data)), random_state=42, covariance_type='full')
                    gmm.fit(cluster_data)
                    candidates = gmm.sample(n_samples=100)[0]
                else:
                    # If only one point in cluster, sample around it
                    candidates = np.random.normal(loc=cluster_centers[i], scale=0.5, size=(100, self.dim))
            else:
                candidates = np.random.normal(loc=cluster_centers[i], scale=0.5, size=(100, self.dim))
            
            candidates = np.clip(candidates, self.bounds[0], self.bounds[1])
            candidate_points.extend(candidates)

        candidate_points = np.array(candidate_points)
        acquisition_values = self._acquisition_function(candidate_points, model)
        indices = np.argsort(-acquisition_values.flatten())[:batch_size]
        selected_points = candidate_points[indices]

        return selected_points

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y
    
    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))
        
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        
        while self.n_evals < self.budget:
            model = self._fit_model(self.X, self.y)
            
            # Dynamic batch size
            batch_size = max(1, int(5 * (1 - self.n_evals / self.budget)))
            next_X = self._select_next_points(batch_size, model)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ADEBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1628 with standard deviation 0.1077.

took 414.88 seconds to run.