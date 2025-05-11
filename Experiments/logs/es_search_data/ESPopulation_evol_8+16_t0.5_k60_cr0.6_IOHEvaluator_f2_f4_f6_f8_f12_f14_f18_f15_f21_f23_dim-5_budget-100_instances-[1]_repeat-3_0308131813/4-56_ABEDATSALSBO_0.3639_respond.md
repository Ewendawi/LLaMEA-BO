# Description
**Adaptive Batch Ensemble Density-Aware Thompson Sampling with Uncertainty-Aware Local Search Bayesian Optimization (ABEDA-TSALSBO):** This algorithm synergistically integrates adaptive batch size, ensemble modeling, density-aware exploration, Thompson Sampling, and uncertainty-aware local search. It dynamically manages an ensemble of Gaussian Process Regression (GPR) models, adjusting the ensemble size based on optimization progress. Thompson Sampling is employed for efficient acquisition within the ensemble. The batch size is adaptively adjusted based on the average uncertainty of the GPR predictions. A Kernel Density Estimation (KDE) focuses the search on high-density regions of promising solutions. Furthermore, it incorporates an uncertainty-aware local search, using the variance predictions from the GPR models to guide the local search iterations and step size. A hybrid acquisition function combines Thompson Sampling, distance-based exploration, and KDE-based exploration.

# Justification
This algorithm combines the strengths of ABETSALSBO and ABDAHBO while attempting to address their individual limitations. ABETSALSBO has a good AOCC score but a slow runtime, while ABDAHBO has a faster runtime but a slightly worse AOCC score.

1.  **Ensemble Modeling with Adaptive Size:** Using an ensemble of GPR models (as in ABETSALSBO) improves the robustness of the surrogate model by capturing different aspects of the function landscape. Adapting the ensemble size based on the optimization progress allows for a better balance between exploration and exploitation.
2.  **Thompson Sampling:** Thompson Sampling provides an efficient way to balance exploration and exploitation within the ensemble (as in ABETSALSBO), reducing the computational overhead compared to other acquisition functions like EI.
3.  **Adaptive Batch Size:** Dynamically adjusting the batch size based on model uncertainty (as in ABETSALSBO and ABDAHBO) improves sampling efficiency. Larger batch sizes are used when uncertainty is high to promote exploration, and smaller batch sizes are used when uncertainty is low to focus on exploitation.
4.  **Density-Aware Exploration:** Incorporating a Kernel Density Estimation (KDE) (as in ABDAHBO) allows the algorithm to focus the search on high-density regions of promising solutions, improving the efficiency of the search.
5.  **Uncertainty-Aware Local Search:** Refining selected points using local search, guided by the uncertainty estimates from the GPR models (as in ABETSALSBO), improves the accuracy of the solutions. The step size and number of iterations of the local search are adapted based on the uncertainty.
6.  **Hybrid Acquisition Function:** Combining Thompson Sampling, distance-based exploration, and KDE-based exploration in the acquisition function allows the algorithm to balance exploration and exploitation effectively.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity, NearestNeighbors


class ABEDATSALSBO:
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

        self.max_models = 5  # Maximum number of surrogate models in the ensemble
        self.min_models = 1  # Minimum number of surrogate models in the ensemble
        self.models = []
        for i in range(self.max_models):
            length_scale = 1.0 * (i + 1) / self.max_models
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
            model = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=5, alpha=1e-5
            )
            self.models.append(model)

        self.max_batch_size = min(10, dim)
        self.min_batch_size = 1
        self.local_search_step_size_factor = 0.1
        self.uncertainty_threshold = 0.5
        self.exploration_weight = 0.1
        self.exploration_decay = 0.995
        self.min_exploration = 0.01
        self.kde_bandwidth = 0.5

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

        acquisition = np.mean(sampled_values, axis=1, keepdims=True)
        acquisition = acquisition.reshape(-1, 1)

        # Distance-based exploration term
        if self.X is not None:
            min_dist = np.min(
                np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2),
                axis=1,
                keepdims=True,
            )
            exploration = min_dist / np.max(min_dist)
        else:
            exploration = np.ones_like(acquisition)

        # KDE-based exploration term
        if self.X is not None and len(self.X) > self.dim + 1:
            bandwidth = self._adaptive_bandwidth()
            kde = KernelDensity(bandwidth=bandwidth).fit(self.X)
            kde_scores = kde.score_samples(X)
            kde_scores = np.exp(kde_scores).reshape(-1, 1)  # Convert to density
            kde_exploration = kde_scores / np.max(kde_scores)  # Normalize
        else:
            kde_exploration = np.zeros(X.shape[0]).reshape(-1, 1)

        # Dynamic weighting
        exploration_weight = np.clip(1.0 - self.n_evals / self.budget, 0.1, 1.0)
        kde_weight = 1.0 - exploration_weight

        acquisition = acquisition + self.exploration_weight * exploration + kde_weight * kde_exploration

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
        candidate_points = self._sample_points(100 * batch_size)
        acquisition_values = self._acquisition_function(candidate_points, active_models)
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]

        # Adaptive Local search
        for i in range(batch_size):
            def obj_func(x):
                x = x.reshape(1, -1)
                return np.mean([model.predict(x)[0] for model in active_models])

            x0 = next_points[i]
            bounds = [(self.bounds[0][j], self.bounds[1][j]) for j in range(self.dim)]

            # Uncertainty-aware local search
            uncertainty = np.mean(
                [model.predict(x0.reshape(1, -1), return_std=True)[1] for model in active_models]
            )
            maxiter = int(5 + 10 * uncertainty)
            maxiter = min(maxiter, 20)

            # Adaptive step size
            step_size = self.local_search_step_size_factor * uncertainty
            options = {"maxiter": maxiter, "ftol": 1e-4}  # Reduced ftol
            res = minimize(obj_func, x0, method="L-BFGS-B", bounds=bounds, options=options)
            next_points[i] = res.x

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
            sigmas = np.array(
                [model.predict(self.X, return_std=True)[1] for model in active_models]
            )
            avg_sigma = np.mean(sigmas)

            if avg_sigma > self.uncertainty_threshold:
                batch_size = self.max_batch_size
                exploration_decay = 0.99
            else:
                batch_size = self.min_batch_size
                exploration_decay = 0.999

            batch_size = min(batch_size, remaining_evals)
            next_X = self._select_next_points(batch_size, active_models)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            active_models = self._fit_model(self.X, self.y)

            # Update exploration weight
            self.exploration_weight = max(
                self.exploration_weight * exploration_decay, self.min_exploration
            )

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<ABEDATSALSBO>", line 189, in __call__
 189->             next_X = self._select_next_points(batch_size, active_models)
  File "<ABEDATSALSBO>", line 117, in _select_next_points
 117->         acquisition_values = self._acquisition_function(candidate_points, active_models)
  File "<ABEDATSALSBO>", line 88, in _acquisition_function
  86 |         if self.X is not None and len(self.X) > self.dim + 1:
  87 |             bandwidth = self._adaptive_bandwidth()
  88->             kde = KernelDensity(bandwidth=bandwidth).fit(self.X)
  89 |             kde_scores = kde.score_samples(X)
  90 |             kde_scores = np.exp(kde_scores).reshape(-1, 1)  # Convert to density
sklearn.utils._param_validation.InvalidParameterError: The 'bandwidth' parameter of KernelDensity must be a float in the range (0.0, inf) or a str among {'silverman', 'scott'}. Got 0.0 instead.
