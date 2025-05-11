from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


class D_EHBBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.diversity_weight = 0.1
        self.n_clusters = 5
        self.best_x = None
        self.best_y = float('inf')

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

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
            ei = ei + self.diversity_weight * min_distances

        return ei

    def _select_next_points(self, batch_size, model):
        # Thompson Sampling component
        num_candidates_ts = 50 * batch_size
        candidates_ts = self._sample_points(num_candidates_ts)
        mu, sigma = model.predict(candidates_ts, return_std=True)
        sampled_values = np.random.normal(mu, sigma)
        indices_ts = np.argsort(sampled_values)[:batch_size // 2]  # Half of the batch from TS
        selected_points_ts = candidates_ts[indices_ts]

        # Clustering-based sampling component
        if self.X is not None:
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.X)
            cluster_centers = kmeans.cluster_centers_
        else:
            cluster_centers = self._sample_points(self.n_clusters)

        candidate_points_cluster = []
        for center in cluster_centers:
            candidates = np.random.normal(loc=center, scale=0.5, size=(20, self.dim))
            candidates = np.clip(candidates, self.bounds[0], self.bounds[1])
            candidate_points_cluster.extend(candidates)
        candidate_points_cluster = np.array(candidate_points_cluster)

        acquisition_values = self._acquisition_function(candidate_points_cluster, model)
        indices_cluster = np.argsort(-acquisition_values.flatten())[:batch_size - batch_size // 2]  # Other half from clustering
        selected_points_cluster = candidate_points_cluster[indices_cluster]

        # Combine selected points
        selected_points = np.vstack((selected_points_ts, selected_points_cluster))

        # Local search refinement
        refined_points = []
        for point in selected_points:
            res = minimize(lambda x: model.predict(x.reshape(1, -1))[0], point,
                           bounds=[(-5, 5)] * self.dim, method='L-BFGS-B')
            refined_points.append(res.x)

        return np.array(refined_points)

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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        batch_size = 5
        while self.n_evals < self.budget:
            model = self._fit_model(self.X, self.y)
            next_X = self._select_next_points(batch_size, model)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
