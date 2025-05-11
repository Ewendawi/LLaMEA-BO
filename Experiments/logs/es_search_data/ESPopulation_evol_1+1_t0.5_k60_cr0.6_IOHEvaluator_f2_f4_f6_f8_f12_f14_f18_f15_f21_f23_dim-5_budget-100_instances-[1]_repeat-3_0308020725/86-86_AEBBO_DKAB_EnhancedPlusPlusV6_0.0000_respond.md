# Description
**AEBBO_DKAB_EnhancedPlusPlusV6:** This algorithm builds upon AEBBO_DKAB_EnhancedPlusPlusV5 by incorporating a more robust kernel length scale adaptation strategy and a refined clustering approach in the batch selection process. The kernel length scale adaptation now considers the gradient of the acquisition function near the best point to better capture the local landscape. The clustering strategy is enhanced by using a combination of KMeans and DBSCAN, where KMeans is used to pre-cluster the data, and DBSCAN is applied within each KMeans cluster to identify finer-grained clusters. This hybrid approach aims to leverage the strengths of both algorithms for more effective batch selection. Finally, a dynamic adjustment of the DBSCAN eps parameter based on the local density is also implemented.

# Justification
- **Gradient-Aware Kernel Length Scale Adaptation:** The kernel length scale is crucial for the performance of the Gaussian Process. By incorporating the gradient of the acquisition function near the best point, the algorithm can better adapt the kernel to the local landscape and improve the accuracy of the GP model.
- **Hybrid Clustering Approach:** KMeans is used to provide a coarse clustering of the data, while DBSCAN is used to identify finer-grained clusters within each KMeans cluster. This hybrid approach aims to leverage the strengths of both algorithms for more effective batch selection. KMeans is good at finding roughly spherical clusters, while DBSCAN is good at finding clusters of arbitrary shape.
- **Dynamic DBSCAN eps Adjustment:** The eps parameter of DBSCAN is dynamically adjusted based on the local density to better adapt to the data. This helps to improve the accuracy of DBSCAN and the overall performance of the algorithm.
- **Exploration-Exploitation Balance:** The exploration weight is adaptively adjusted based on the uncertainty of the GP model and the progress of the optimization. This helps to balance exploration and exploitation and improve the overall performance of the algorithm.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc, norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize, Bounds, approx_fprime
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.linalg import solve

class AEBBO_DKAB_EnhancedPlusPlusV6:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim # number of initial samples

        # Do not add any other arguments without a default value
        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.acq_strategy = "EI+UCB"
        self.initial_exploration_weight = 0.2 + 0.05 * (dim/10)
        self.exploration_weight = self.initial_exploration_weight
        self.batch_size = 2
        self.kernel_length_scale = 1.0
        self.local_search_restarts = 3
        self.length_scale_weight = 0.5 # Weight for previous length scale in adaptation
        self.gradient_local_search = True # Flag to use gradient-based local search
        self.length_scale_history = []
        self.length_scale_history_length = 10
        self.reevaluate_interval = 10 * dim # Re-evaluate best point every this many evaluations
        self.variance_weight = 0.2 # Weight for variance in kernel length scale adaptation
        self.length_scale_momentum = 0.7 # Momentum for kernel length scale adaptation
        self.previous_length_scale_change = 0.0
        self.gradient_weight = 0.1

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        if self.X is None or len(self.X) < self.dim + 1:
            sampler = qmc.LatinHypercube(d=self.dim)
            samples = sampler.random(n=n_points)
            return qmc.scale(samples, self.bounds[0], self.bounds[1])
        else:
            try:
                kde = gaussian_kde(self.X.T)
                samples = kde.resample(n_points)
                samples = np.clip(samples.T, self.bounds[0], self.bounds[1])
                return samples
            except np.linalg.LinAlgError:
                sampler = qmc.LatinHypercube(d=self.dim)
                samples = sampler.random(n=n_points)
                return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature

        # Dynamic kernel adaptation
        distances = cdist(X, X, metric='euclidean')
        distances = np.triu(distances, k=1)
        distances = distances[distances > 0]
        mean_distance = np.mean(distances) if len(distances) > 0 else 1.0
        _, variances = self.gp.predict(X, return_std=True) if self.gp else (None, np.ones(len(X)))
        mean_variance = np.mean(variances)

        # Gradient-based length scale adaptation
        if self.best_x is not None and self.gradient_local_search:
            def obj_for_grad(x):
                return self._acquisition_function(x.reshape(1, -1))[0, 0]
            gradient = approx_fprime(self.best_x, obj_for_grad, epsilon=1e-6)
            gradient_norm = np.linalg.norm(gradient)
        else:
            gradient_norm = 0.0

        # Smooth kernel length scale adaptation with momentum
        new_length_scale = mean_distance * (1 + self.variance_weight * mean_variance) * (1 + self.gradient_weight * gradient_norm)
        self.length_scale_history.append(new_length_scale)
        if len(self.length_scale_history) > self.length_scale_history_length:
            self.length_scale_history.pop(0)
        smoothed_length_scale = np.mean(self.length_scale_history)

        length_scale_change = (1 - self.length_scale_weight) * smoothed_length_scale - self.kernel_length_scale
        length_scale_change = self.length_scale_momentum * self.previous_length_scale_change + (1 - self.length_scale_momentum) * length_scale_change
        self.kernel_length_scale += length_scale_change
        self.previous_length_scale_change = length_scale_change

        # Use a spectral mixture kernel
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.kernel_length_scale, length_scale_bounds="fixed")
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.acq_strategy == "EI":
            return self._expected_improvement(X)
        elif self.acq_strategy == "UCB":
            return self._upper_confidence_bound(X)
        elif self.acq_strategy == "EI+UCB":
            ei = self._expected_improvement(X)
            ucb = self._upper_confidence_bound(X)
            return ei + self.exploration_weight * ucb
        else:
            raise ValueError("Invalid acquisition function strategy.")

    def _expected_improvement(self, X):
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-6)  # avoid division by zero
        gamma = (self.best_y - mu) / (sigma + 1e-9) # avoid division by zero
        ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return ei.reshape(-1, 1)

    def _upper_confidence_bound(self, X):
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu.reshape(-1, 1) + 2 * sigma.reshape(-1, 1)

    def _select_next_points(self, batch_size):
         # Select the next points to evaluate using a clustering-based approach
        candidates = self._sample_points(100 * self.dim)
        acq_values = self._acquisition_function(candidates)
        _, variances = self.gp.predict(candidates, return_std=True)
        variances = variances.reshape(-1, 1)

        # Scale the features for clustering
        scaler = StandardScaler()
        scaled_candidates = scaler.fit_transform(candidates)

        # Use KMeans for pre-clustering
        kmeans = KMeans(n_clusters=min(batch_size * 5, len(candidates)), random_state=0, n_init='auto')
        kmeans.fit(np.hstack((acq_values, cdist(candidates, self.best_x.reshape(1, -1), metric='euclidean'), variances)))
        kmeans_labels = kmeans.labels_
        n_kmeans_clusters = len(set(kmeans_labels))

        cluster_labels = np.zeros(len(candidates), dtype=int) - 1  # Initialize with -1 for noise

        # Apply DBSCAN within each KMeans cluster
        current_label = 0
        for i in range(n_kmeans_clusters):
            cluster_indices = np.where(kmeans_labels == i)[0]
            cluster_data = scaled_candidates[cluster_indices]

            # Dynamic eps adjustment based on local density
            distances = cdist(cluster_data, cluster_data, metric='euclidean')
            distances = np.triu(distances, k=1)
            distances = distances[distances > 0]
            if len(distances) > 0:
                eps = np.percentile(distances, 25)  # Use 25th percentile as a measure of local density
            else:
                eps = 0.5 # Default value if no distances

            dbscan = DBSCAN(eps=eps, min_samples=max(2, len(cluster_data) // 10))  # Adjust min_samples as needed
            dbscan_labels = dbscan.fit_predict(cluster_data)

            # Remap DBSCAN labels to global cluster labels
            for j in range(len(dbscan_labels)):
                if dbscan_labels[j] != -1:
                    cluster_labels[cluster_indices[j]] = current_label + dbscan_labels[j]
            current_label += len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

        # Handle noise points (unclustered points)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        for i in range(len(cluster_labels)):
            if cluster_labels[i] == -1:
                cluster_labels[i] = n_clusters
                n_clusters += 1

        # Calculate cluster weights based on average acquisition function value and distance to best
        cluster_weights = np.zeros(n_clusters)
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            acq_mean = np.mean(acq_values[cluster_indices])
            dist_mean = np.mean(cdist(candidates[cluster_indices], self.best_x.reshape(1, -1), metric='euclidean'))
            variance_mean = np.mean(variances[cluster_indices])
            cluster_weights[i] = (acq_mean + variance_mean) / (1 + dist_mean) # Favor clusters with high acq, high variance and close to best

        cluster_weights = np.exp(cluster_weights - np.max(cluster_weights))  # Softmax for stability
        cluster_weights /= np.sum(cluster_weights)

        # Sample clusters based on weights and select best point from each
        selected_points = []
        sampled_clusters = np.random.choice(n_clusters, size=batch_size, replace=len(range(n_clusters)) < batch_size, p=cluster_weights)
        for i in sampled_clusters:
            cluster_indices = np.where(cluster_labels == i)[0]
            best_index = cluster_indices[np.argmax(acq_values[cluster_indices])]
            selected_points.append(candidates[best_index])

        return np.array(selected_points)

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
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Dynamic batch size adjustment
            mean_sigma = np.mean(self.gp.predict(self.X, return_std=True)[1])
            self.batch_size = max(1, min(self.batch_size + (1 if mean_sigma > 0.5 else -1), 5))
            next_X = self._select_next_points(min(self.batch_size, self.budget - self.n_evals))

            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Adaptive acquisition balancing (Thompson Sampling inspired)
            mean_sigma = np.mean(self.gp.predict(self.X, return_std=True)[1])
            range_y = np.max(self.y) - np.min(self.y) if len(self.y) > 1 and np.max(self.y) != np.min(self.y) else 1.0
            self.exploration_weight = self.initial_exploration_weight * (mean_sigma / range_y) * (1 - (self.n_evals / self.budget)**2) # More aggressive decay
            self.exploration_weight = np.clip(self.exploration_weight, 0.01, self.initial_exploration_weight)

            # Periodic re-evaluation of best point
            if self.n_evals % self.reevaluate_interval == 0 and self.n_evals > self.n_init:
                best_y = self._evaluate_points(func, self.best_x.reshape(1, -1))
                if best_y[0][0] < self.best_y:
                    self.best_y = best_y[0][0]
                    self._update_eval_points(self.best_x.reshape(1, -1), best_y)


        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<AEBBO_DKAB_EnhancedPlusPlusV6>", line 239, in __call__
 239->             self._fit_model(self.X, self.y)
  File "<AEBBO_DKAB_EnhancedPlusPlusV6>", line 80, in _fit_model
  80->             gradient = approx_fprime(self.best_x, obj_for_grad, epsilon=1e-6)
  File "<AEBBO_DKAB_EnhancedPlusPlusV6>", line 79, in obj_for_grad
  79->                 return self._acquisition_function(x.reshape(1, -1))[0, 0]
  File "<AEBBO_DKAB_EnhancedPlusPlusV6>", line 112, in _acquisition_function
 112->             ei = self._expected_improvement(X)
  File "<AEBBO_DKAB_EnhancedPlusPlusV6>", line 119, in _expected_improvement
 117 | 
 118 |     def _expected_improvement(self, X):
 119->         mu, sigma = self.gp.predict(X, return_std=True)
 120 |         sigma = np.maximum(sigma, 1e-6)  # avoid division by zero
 121 |         gamma = (self.best_y - mu) / (sigma + 1e-9) # avoid division by zero
AttributeError: 'NoneType' object has no attribute 'predict'
