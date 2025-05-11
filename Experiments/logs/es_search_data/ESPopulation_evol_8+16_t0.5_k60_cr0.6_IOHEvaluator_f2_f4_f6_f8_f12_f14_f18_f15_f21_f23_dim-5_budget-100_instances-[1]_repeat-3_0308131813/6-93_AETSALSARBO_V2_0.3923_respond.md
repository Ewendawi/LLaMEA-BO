# Description
**AETSALSARBO-V2: Adaptive Ensemble with Thompson Sampling, Dynamic Exploration, Adaptive Local Search Radius, and Enhanced Exploration-Exploitation Trade-off Bayesian Optimization.** This enhanced version of AETSALSARBO refines the exploration-exploitation trade-off by introducing a dynamic adjustment of the exploration weight based on the *improvement rate* in addition to optimization progress and GPR uncertainty. It also incorporates a more sophisticated mechanism for adjusting the local search radius, taking into account the *gradient norm* of the predicted mean from the GPR models. This allows for a more targeted local search, focusing on regions where the function is changing rapidly.

# Justification
The key improvements in AETSALSARBO-V2 are:

1.  **Dynamic Exploration Weight Adjustment:** The exploration weight is now adjusted based on the improvement rate (change in best function value). If the improvement rate is low, the exploration weight is increased to encourage exploration. This helps to escape local optima and find better solutions.

2.  **Gradient-Aware Local Search Radius:** The local search radius is adjusted based on the gradient norm of the predicted mean from the GPR models. A larger gradient norm indicates a region where the function is changing rapidly, so the local search radius is increased to explore this region more thoroughly. This allows for a more targeted local search and can lead to faster convergence.

3.  **Ensemble Diversity:** To encourage diversity within the ensemble, a small amount of noise is added to the length scales of the RBF kernels during initialization.

These changes aim to improve the exploration-exploitation balance and the efficiency of the local search, leading to better performance on the BBOB test suite.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KernelDensity, NearestNeighbors
from scipy.optimize import minimize
from scipy.optimize import approx_fprime


class AETSALSARBO_V2:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.k_neighbors = min(10, 2 * dim)
        self.best_y = np.inf
        self.best_x = None
        self.kde_bandwidth = 0.5
        self.max_batch_size = min(10, dim)  # Maximum batch size for selecting points
        self.min_batch_size = 1
        self.exploration_weight = 0.2  # Initial exploration weight
        self.exploration_weight_min = 0.01  # Minimum exploration weight
        self.uncertainty_threshold = 0.5  # Threshold for adjusting batch size
        self.initial_local_search_radius = 0.2
        self.local_search_radius = self.initial_local_search_radius
        self.local_search_radius_min = 0.01
        self.local_search_step_size_factor = 0.1
        self.previous_best_y = np.inf
        self.improvement_rate = 0.0

        self.max_models = 5  # Maximum number of surrogate models in the ensemble
        self.min_models = 1  # Minimum number of surrogate models in the ensemble
        self.models = []
        for i in range(self.max_models):
            length_scale = 1.0 * (i + 1) / self.max_models + np.random.normal(0, 0.05)  # Add noise
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
            model = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=5, alpha=1e-5
            )
            self.models.append(model)

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Adapt ensemble size
        n_models = max(
            self.min_models, int(self.max_models * (1 - self.n_evals / self.budget))
        )
        active_models = self.models[:n_models]
        for model in active_models:
            model.fit(X, y)
        return active_models

    def _acquisition_function(self, X, active_models):
        # Thompson Sampling
        sampled_values = np.zeros((X.shape[0], len(active_models)))
        sigmas = np.zeros((X.shape[0], len(active_models)))
        for i, model in enumerate(active_models):
            mu, sigma = model.predict(X, return_std=True)
            mu = mu.reshape(-1, 1)
            sigma = sigma.reshape(-1, 1)
            sampled_values[:, i] = np.random.normal(mu.flatten(), sigma.flatten())
            sigmas[:, i] = sigma.flatten()

        acquisition_ts = np.mean(sampled_values, axis=1, keepdims=True)
        acquisition_ts = acquisition_ts.reshape(-1, 1)

        # Hybrid acquisition function (EI + exploration)
        mu = np.mean([model.predict(X) for model in active_models], axis=0).reshape(
            -1, 1
        )
        sigma = np.mean(sigmas, axis=1).reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Distance-based exploration term
        if self.X is not None and len(self.X) > 0:
            min_dist = np.min(
                np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2),
                axis=1,
                keepdims=True,
            )
            exploration = min_dist / np.max(min_dist)
        else:
            exploration = np.ones(X.shape[0]).reshape(-1, 1)

        # Hybrid acquisition function
        acquisition = ei + self.exploration_weight * exploration
        return acquisition

    def _adaptive_bandwidth(self):
        if self.X is None or len(self.X) < self.k_neighbors:
            return self.kde_bandwidth

        nbrs = NearestNeighbors(
            n_neighbors=self.k_neighbors, algorithm="ball_tree"
        ).fit(self.X)
        distances, _ = nbrs.kneighbors(self.X)
        k_distances = distances[:, -1]
        bandwidth = np.median(k_distances)
        return bandwidth

    def _select_next_points(self, batch_size, active_models):
        candidate_points = self._sample_points(50 * batch_size)

        # Add points around the best solution (local search)
        if self.best_x is not None:
            local_points = np.random.normal(loc=self.best_x, scale=self.local_search_radius, size=(50 * batch_size, self.dim))
            local_points = np.clip(local_points, self.bounds[0], self.bounds[1])
            candidate_points = np.vstack((candidate_points, local_points))

        acquisition_values = self._acquisition_function(candidate_points, active_models)
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]


        return next_points

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

        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]
            self.improvement_rate = (self.previous_best_y - self.best_y) / self.previous_best_y if self.previous_best_y != 0 else 0
        else:
            self.improvement_rate = 0

        self.previous_best_y = self.best_y

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[
        np.float64, np.array
    ]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        active_models = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            remaining_evals = self.budget - self.n_evals

            # Adjust batch size based on uncertainty
            sigmas = []
            for model in active_models:
                _, sigma = model.predict(self.X, return_std=True)
                sigmas.append(np.mean(sigma))
            avg_sigma = np.mean(sigmas)

            if avg_sigma > self.uncertainty_threshold:
                batch_size = self.max_batch_size
            else:
                batch_size = self.min_batch_size

            batch_size = min(batch_size, remaining_evals)  # Adjust batch size to budget

            next_X = self._select_next_points(batch_size, active_models)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            active_models = self._fit_model(self.X, self.y)

            # Update exploration weight
            self.exploration_weight = max(self.exploration_weight_min, self.exploration_weight * (1 - self.n_evals / self.budget) * (1 + avg_sigma) * (1 + (0.5 - self.improvement_rate)))

            # Calculate gradient norm for local search radius adjustment
            if self.best_x is not None:
                # Use a mean model for gradient estimation
                def mean_prediction(x):
                    X_reshaped = x.reshape(1, -1)
                    return np.mean([model.predict(X_reshaped) for model in active_models])

                gradient = approx_fprime(self.best_x, mean_prediction, epsilon=1e-6)
                gradient_norm = np.linalg.norm(gradient)
            else:
                gradient_norm = 0

            # Update local search radius
            self.local_search_radius = max(self.local_search_radius_min, self.initial_local_search_radius * (1 - self.n_evals / self.budget) * (1 + avg_sigma) * (1 + gradient_norm))


        return self.best_y, self.best_x
```
## Feedback
 The algorithm AETSALSARBO_V2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1684 with standard deviation 0.1050.

took 366.31 seconds to run.