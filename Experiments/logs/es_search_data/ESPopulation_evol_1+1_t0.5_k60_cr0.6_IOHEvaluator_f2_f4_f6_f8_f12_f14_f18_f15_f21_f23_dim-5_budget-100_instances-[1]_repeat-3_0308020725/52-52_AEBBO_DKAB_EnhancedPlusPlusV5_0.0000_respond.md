# Description
**AEBBO_DKAB_EnhancedPlusPlusV5:** This algorithm refines AEBBO_DKAB_EnhancedPlusPlusV4 by introducing a more sophisticated approach to kernel length scale adaptation and batch selection. The kernel length scale adaptation now incorporates a dynamic weighting based on the age of the data points, giving more weight to recent observations. The batch selection process is enhanced by using Thompson sampling within each cluster to balance exploration and exploitation more effectively. Additionally, a local search is performed around the best point of each batch to refine the search.

# Justification
The key improvements are:

1.  **Age-Based Kernel Length Scale Adaptation:** The kernel length scale is crucial for GP performance. By weighting the length scale adaptation based on the age of the data points, the algorithm becomes more responsive to recent changes in the objective function landscape. This helps to adapt the GP model more quickly to local features.

2.  **Thompson Sampling in Batch Selection:** Instead of simply selecting the point with the highest acquisition function value within each cluster, Thompson sampling is used to draw a sample from the posterior distribution of the acquisition function. This provides a more probabilistic approach to batch selection and balances exploration and exploitation more effectively.

3.  **Local Search Around Batch Best Points:** Performing a local search around the best point of each batch helps to refine the search and potentially find better solutions. This local search is gradient-free to avoid issues with non-differentiable functions.

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
from sklearn.cluster import KMeans
from scipy.optimize import minimize

class AEBBO_DKAB_EnhancedPlusPlusV5:
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
        self.age_weight = 0.1 # Weight for age in kernel length scale adaptation

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

        # Age-based weighting
        age_weights = np.linspace(1.0, 0.1, len(X))  # Newer points have higher weight
        age_weights /= np.sum(age_weights)

        # Smooth kernel length scale adaptation
        new_length_scale = mean_distance * (1 + self.variance_weight * mean_variance)
        self.length_scale_history.append(new_length_scale)
        if len(self.length_scale_history) > self.length_scale_history_length:
            self.length_scale_history.pop(0)
        smoothed_length_scale = np.mean(self.length_scale_history)

        self.kernel_length_scale = self.length_scale_weight * self.kernel_length_scale + (1 - self.length_scale_weight) * smoothed_length_scale

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
        mu, sigma = self.gp.predict(candidates, return_std=True)
        variances = sigma.reshape(-1, 1)


        # Cluster the candidates based on their acquisition function values and distance to best
        n_clusters = min(batch_size, len(candidates))  # Ensure we don't request more clusters than candidates
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init = 'auto')
        kmeans.fit(np.hstack((acq_values, cdist(candidates, self.best_x.reshape(1, -1), metric='euclidean'), variances)))
        cluster_labels = kmeans.labels_

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
            # Thompson Sampling within the cluster
            mu_cluster = mu[cluster_indices]
            sigma_cluster = sigma[cluster_indices]
            sampled_acq_values = np.random.normal(mu_cluster, sigma_cluster)
            best_index = cluster_indices[np.argmax(sampled_acq_values)]
            selected_points.append(candidates[best_index])

        # Local search around selected points
        refined_points = []
        for point in selected_points:
            refined_point = self._local_search(point, func)
            refined_points.append(refined_point)

        return np.array(refined_points)

    def _local_search(self, x_start, func):
        # Perform a local search around x_start
        bounds = Bounds(self.bounds[0], self.bounds[1])
        result = minimize(func, x_start, method='L-BFGS-B', bounds=bounds)
        return result.x

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
            self.exploration_weight = self.initial_exploration_weight * (mean_sigma / range_y) * (1 - (self.n_evals / self.budget)**1) # Less aggressive decay
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
  File "<AEBBO_DKAB_EnhancedPlusPlusV5>", line 213, in __call__
 213->             next_X = self._select_next_points(min(self.batch_size, self.budget - self.n_evals))
  File "<AEBBO_DKAB_EnhancedPlusPlusV5>", line 159, in _select_next_points
 157 |         refined_points = []
 158 |         for point in selected_points:
 159->             refined_point = self._local_search(point, func)
 160 |             refined_points.append(refined_point)
 161 | 
NameError: name 'func' is not defined
