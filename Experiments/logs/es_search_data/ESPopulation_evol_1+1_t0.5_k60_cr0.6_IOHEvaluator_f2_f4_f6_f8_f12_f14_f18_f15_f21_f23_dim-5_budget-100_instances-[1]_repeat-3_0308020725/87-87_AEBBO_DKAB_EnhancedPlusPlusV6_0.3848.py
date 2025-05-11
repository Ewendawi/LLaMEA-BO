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
from sklearn.neighbors import NearestNeighbors

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
        self.success_rate = 0.0
        self.success_history_length = 10
        self.success_history = []
        self.previous_gp = None

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
        self.previous_gp = self.gp

        # Dynamic kernel adaptation
        distances = cdist(X, X, metric='euclidean')
        distances = np.triu(distances, k=1)
        distances = distances[distances > 0]
        mean_distance = np.mean(distances) if len(distances) > 0 else 1.0
        _, variances = self.gp.predict(X, return_std=True) if self.gp else (None, np.ones(len(X)))
        mean_variance = np.mean(variances)

        # Smooth kernel length scale adaptation with momentum
        new_length_scale = mean_distance * (1 + self.variance_weight * mean_variance)
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

        # Uncertainty reduction
        if self.previous_gp is not None:
            old_variances = self.previous_gp.predict(X, return_std=True)[1]
            new_variances = self.gp.predict(X, return_std=True)[1]
            uncertainty_reduction = np.mean(old_variances - new_variances)
            self.kernel_length_scale *= np.exp(0.1 * uncertainty_reduction)

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

        # Adaptive DBSCAN parameter selection
        nn = NearestNeighbors(n_neighbors=min(10, len(candidates)-1))
        nn.fit(scaled_candidates)
        distances, _ = nn.kneighbors(scaled_candidates)
        eps = np.mean(distances[:, 1])  # Average distance to the nearest neighbor
        min_samples = max(2, int(0.05 * len(candidates)))

        # Use DBSCAN for clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # Adjust eps and min_samples as needed
        cluster_labels = dbscan.fit_predict(scaled_candidates)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)  # Number of clusters, ignoring noise

        # If DBSCAN finds no clusters or fails, fall back to KMeans
        if n_clusters <= 0:
            n_clusters = min(batch_size, len(candidates))
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
            kmeans.fit(np.hstack((acq_values, cdist(candidates, self.best_x.reshape(1, -1), metric='euclidean'), variances)))
            cluster_labels = kmeans.labels_
        else:
            # For DBSCAN, remap noise points (-1) to their own cluster
            max_label = np.max(cluster_labels)
            for i in range(len(cluster_labels)):
                if cluster_labels[i] == -1:
                    max_label += 1
                    cluster_labels[i] = max_label
            n_clusters = len(set(cluster_labels))

        # Calculate cluster weights based on average acquisition function value and distance to best
        cluster_weights = np.zeros(n_clusters)
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            acq_mean = np.mean(acq_values[cluster_indices])
            dist_mean = np.mean(cdist(candidates[cluster_indices], self.best_x.reshape(1, -1), metric='euclidean'))
            variance_mean = np.mean(variances[cluster_indices])
            cluster_size = len(cluster_indices)
            cluster_diversity = np.mean(cdist(candidates[cluster_indices], candidates[cluster_indices], metric='euclidean')) if cluster_size > 1 else 0.0
            cluster_weights[i] = (acq_mean + variance_mean) / (1 + dist_mean) * cluster_size * (1 + cluster_diversity) # Favor clusters with high acq, high variance, close to best, large size and high diversity

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
        new_best_y = self.y[best_index][0]
        improvement = self.best_y - new_best_y
        self.best_y = new_best_y
        self.best_x = self.X[best_index]

        # Update success rate
        self.success_history.append(1 if improvement > 0 else 0)
        if len(self.success_history) > self.success_history_length:
            self.success_history.pop(0)
        self.success_rate = np.mean(self.success_history)

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
            self.exploration_weight *= (1 - self.success_rate) # Reduce exploration if success rate is high
            self.exploration_weight = np.clip(self.exploration_weight, 0.01, self.initial_exploration_weight)

            # Periodic re-evaluation of best point
            if self.n_evals % self.reevaluate_interval == 0 and self.n_evals > self.n_init:
                best_y = self._evaluate_points(func, self.best_x.reshape(1, -1))
                if best_y[0][0] < self.best_y:
                    self.best_y = best_y[0][0]
                    self._update_eval_points(self.best_x.reshape(1, -1), best_y)


        return self.best_y, self.best_x
