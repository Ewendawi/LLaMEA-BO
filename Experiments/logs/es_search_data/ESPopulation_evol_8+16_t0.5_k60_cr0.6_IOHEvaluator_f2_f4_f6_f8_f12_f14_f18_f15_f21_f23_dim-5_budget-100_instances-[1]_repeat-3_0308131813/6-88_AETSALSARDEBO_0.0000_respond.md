# Description
**Adaptive Ensemble with Thompson Sampling, Dynamic Exploration, Adaptive Local Search with Radius and Derivative Exploitation Bayesian Optimization (AETSALSARDEBO):** This algorithm synergistically integrates the strengths of AETSALSARBO and ABETSALSDEBO, incorporating an adaptive ensemble of Gaussian Process Regression (GPR) models with Thompson Sampling for efficient acquisition. It features dynamic exploration with both distance-based and KDE-based components, an adaptive local search with radius adjustment, and a novel derivative exploitation strategy to accelerate convergence. The algorithm dynamically adjusts the batch size based on GPR uncertainty and adapts the exploration weight and local search radius based on optimization progress and uncertainty. The derivative exploitation refines the local search by estimating the gradient around promising solutions and moving along the estimated steepest descent direction, enhancing the algorithm's ability to escape local optima and converge rapidly.

# Justification
The AETSALSARDEBO algorithm is designed to improve upon AETSALSARBO and ABETSALSDEBO by combining their key strengths and addressing their potential weaknesses.

*   **Ensemble of GPR models with Thompson Sampling:** This approach, inherited from both parent algorithms, leverages multiple GPR models with varying length scales to capture different aspects of the function landscape and uses Thompson Sampling for efficient exploration-exploitation trade-off.

*   **Dynamic Exploration:** The algorithm incorporates both distance-based and KDE-based exploration strategies, similar to ABETSALSDEBO. This combination allows for a more comprehensive exploration of the search space, balancing global exploration with focused search in promising regions.

*   **Adaptive Local Search with Radius Adjustment:** AETSALSARBO's adaptive local search radius is retained, allowing the algorithm to dynamically adjust the search intensity based on optimization progress and model uncertainty.

*   **Derivative Exploitation:** This is the key novel component. After selecting points using the acquisition function and performing the initial local search with radius adjustment, the algorithm estimates the gradient of the objective function around these points using finite differences. Then, it moves the points along the estimated steepest descent direction. This derivative exploitation strategy refines the local search, helping the algorithm to converge more quickly to local optima and escape saddle points.

*   **Adaptive Batch Size:** The algorithm dynamically adjusts the batch size based on the GPR model's uncertainty, as in ABETSALSDEBO. This adaptive batch size helps to balance exploration and exploitation, allocating more evaluations to promising regions when uncertainty is low and increasing exploration when uncertainty is high.

*   **Dynamic Adjustment of Exploration Weight:** The exploration weight is dynamically adjusted based on the optimization progress and the GPR model's uncertainty, as in AETSALSARBO, allowing for a robust and efficient exploration-exploitation trade-off.

The derivative exploitation component is designed to address a potential weakness of both AETSALSARBO and ABETSALSDEBO: slow convergence in the later stages of optimization. By explicitly exploiting the gradient information, the algorithm can more efficiently refine solutions and accelerate convergence.

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

class AETSALSARDEBO:
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
        self.derivative_exploitation_step_size = 0.01

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

        # Hybrid acquisition function (EI + exploration + KDE)
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

        # Hybrid acquisition function
        acquisition = (
            ei
            + self.exploration_weight * exploration
            + kde_weight * kde_exploration
        )
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

    def _estimate_gradient(self, func, x, active_models):
        """Estimates the gradient of the objective function using finite differences."""
        gradient = np.zeros(self.dim)
        h = self.derivative_exploitation_step_size  # Step size for finite differences
        for i in range(self.dim):
            x_plus_h = x.copy()
            x_plus_h[i] += h
            x_plus_h = np.clip(x_plus_h, self.bounds[0][i], self.bounds[1][i])

            x_minus_h = x.copy()
            x_minus_h[i] -= h
            x_minus_h = np.clip(x_minus_h, self.bounds[0][i], self.bounds[1][i])

            # Estimate the derivative using central difference
            grad_i = (func(x_plus_h) - func(x_minus_h)) / (2 * h)
            gradient[i] = grad_i
        return gradient

    def _select_next_points(self, batch_size, active_models, func):
        candidate_points = self._sample_points(50 * batch_size)

        # Add points around the best solution (local search)
        if self.best_x is not None:
            local_points = np.random.normal(loc=self.best_x, scale=self.local_search_radius, size=(50 * batch_size, self.dim))
            local_points = np.clip(local_points, self.bounds[0], self.bounds[1])
            candidate_points = np.vstack((candidate_points, local_points))

        acquisition_values = self._acquisition_function(candidate_points, active_models)
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]

        # Derivative exploitation
        for i in range(batch_size):
            gradient = self._estimate_gradient(func, next_points[i], active_models)
            next_points[i] = next_points[i] - self.derivative_exploitation_step_size * gradient
            next_points[i] = np.clip(next_points[i], self.bounds[0], self.bounds[1])

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

            next_X = self._select_next_points(batch_size, active_models, func)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            active_models = self._fit_model(self.X, self.y)

            # Update exploration weight
            self.exploration_weight = max(self.exploration_weight_min, self.exploration_weight * (1 - self.n_evals / self.budget) * (1 + avg_sigma))

            # Update local search radius
            self.local_search_radius = max(self.local_search_radius_min, self.initial_local_search_radius * (1 - self.n_evals / self.budget) * (1 + avg_sigma))

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<AETSALSARDEBO>", line 213, in __call__
 213->             next_X = self._select_next_points(batch_size, active_models, func)
  File "<AETSALSARDEBO>", line 163, in _select_next_points
 163->             gradient = self._estimate_gradient(func, next_points[i], active_models)
  File "<AETSALSARDEBO>", line 144, in _estimate_gradient
 142 | 
 143 |             # Estimate the derivative using central difference
 144->             grad_i = (func(x_plus_h) - func(x_minus_h)) / (2 * h)
 145 |             gradient[i] = grad_i
 146 |         return gradient
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
