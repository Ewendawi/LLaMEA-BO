from collections.abc import Callable
from scipy.stats import qmc, norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize, Bounds, approx_fprime
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans

class AEBBO_DKAB_EnhancedPlusPlusV2:
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

        # Gradient-aware kernel length scale adaptation
        if self.gp:
            gradients = []
            for x in X:
                grad = approx_fprime(x, lambda x: self.gp.predict(x.reshape(1, -1))[0], epsilon=1e-6)
                gradients.append(np.linalg.norm(grad))
            mean_gradient = np.mean(gradients) if gradients else 0.0
        else:
            mean_gradient = 0.0

        # Smooth kernel length scale adaptation
        self.kernel_length_scale = self.length_scale_weight * self.kernel_length_scale + (1 - self.length_scale_weight) * mean_distance * (1 + mean_variance) * (1 + mean_gradient)

        # Use a spectral mixture kernel
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.kernel_length_scale, length_scale_bounds="fixed")
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        ei = self._expected_improvement(X)
        ucb = self._upper_confidence_bound(X)

        # Dynamic acquisition balancing
        if self.gp:
            _, sigma = self.gp.predict(self.X, return_std=True)
            mean_sigma = np.mean(sigma)
            ei_weight = 1 / (1 + mean_sigma)
            ucb_weight = mean_sigma
        else:
            ei_weight = 0.5
            ucb_weight = 0.5

        return ei_weight * ei + ucb_weight * self.exploration_weight * ucb

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
        # Select the next points to evaluate using a clustering-based approach with diversity
        candidates = self._sample_points(100 * self.dim)
        acq_values = self._acquisition_function(candidates)

        # Cluster the candidates based on their acquisition function values
        n_clusters = min(batch_size, len(candidates))  # Ensure we don't request more clusters than candidates
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init = 'auto')
        kmeans.fit(acq_values)
        cluster_labels = kmeans.labels_

        # Select the point with the highest acquisition function value from each cluster
        selected_points = []
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            best_index = cluster_indices[np.argmax(acq_values[cluster_indices])]
            selected_points.append(candidates[best_index])

        selected_points = np.array(selected_points)

        # Add diversity metric: penalize points that are too close to each other
        if len(selected_points) > 1:
            distances = cdist(selected_points, selected_points, 'euclidean')
            diversity_penalty = np.sum(np.triu(distances, k=1))
            # Sort by acquisition value minus diversity penalty
            acq_values_selected = self._acquisition_function(selected_points)
            penalized_acq = acq_values_selected.flatten() - diversity_penalty
            selected_points = selected_points[np.argsort(penalized_acq)[::-1]]

        return selected_points[:batch_size]

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
            self.exploration_weight = self.initial_exploration_weight * (mean_sigma / range_y) * (1 - self.n_evals / self.budget)
            self.exploration_weight = np.clip(self.exploration_weight, 0.01, self.initial_exploration_weight)

        return self.best_y, self.best_x
