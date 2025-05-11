# Description
**Adaptive Batch Ensemble with Thompson Sampling, Kernel Density Estimation, and Gradient-Enhanced Local Search (ABETSALS-GELSBO):** This algorithm builds upon ABETSALSDEBO by incorporating a more robust gradient-enhanced local search and a refined strategy for balancing exploration and exploitation. It uses an adaptive ensemble of Gaussian Process Regression (GPR) models and Thompson Sampling for efficient acquisition. It also includes Kernel Density Estimation (KDE) to focus the search on promising regions. The key improvement is the gradient-enhanced local search, which uses gradient information from the GPR models to guide the local search more effectively. Furthermore, the exploration-exploitation balance is dynamically adjusted based on both the GPR model uncertainty and the KDE density, allowing for a more efficient search.

# Justification
The algorithm combines several successful strategies from previous approaches:
1.  **Adaptive Batch Ensemble:** Using an ensemble of GPR models with varying length scales allows the algorithm to capture different aspects of the function landscape and provides more robust uncertainty estimates. The ensemble size is dynamically adjusted based on the optimization progress to reduce computational cost in later stages.
2.  **Thompson Sampling:** Thompson Sampling is an efficient acquisition strategy that balances exploration and exploitation.
3.  **Kernel Density Estimation (KDE):** KDE helps to focus the search on high-density regions of promising solutions, improving exploitation. The bandwidth is dynamically adjusted based on the local density of evaluated points.
4.  **Gradient-Enhanced Local Search (GELS):** The local search is enhanced by incorporating gradient information from the GPR models. This allows the local search to converge more quickly and accurately to local optima. The number of local search iterations is dynamically adjusted based on the model uncertainty.
5.  **Adaptive Exploration-Exploitation Balance:** The exploration weight is dynamically adjusted based on both the GPR model uncertainty and the KDE density. This allows the algorithm to adapt its search strategy based on the characteristics of the function landscape. When the uncertainty is high, the algorithm increases exploration. When the KDE density is high, the algorithm increases exploitation.

The gradient-enhanced local search is a key improvement over previous approaches. By using gradient information, the local search can converge more quickly and accurately to local optima. This is particularly important for high-dimensional problems, where the function landscape can be complex and difficult to navigate.

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

class ABETSALS_GELSBO:
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
        self.exploration_weight = 0.1  # Initial exploration weight
        self.exploration_decay = 0.995  # Decay factor for exploration weight
        self.min_exploration = 0.01  # Minimum exploration weight
        self.uncertainty_threshold = 0.5  # Threshold for adjusting batch size
        self.local_search_step_size_factor = 0.1

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

    def _gradient(self, x, active_models):
        grads = []
        for model in active_models:
            # numerical gradient
            delta = 1e-5
            grad = np.zeros_like(x)
            for j in range(self.dim):
                x_plus = x.copy()
                x_plus[j] += delta
                x_minus = x.copy()
                x_minus[j] -= delta
                grad[j] = (model.predict(x_plus.reshape(1, -1))[0] - model.predict(x_minus.reshape(1, -1))[0]) / (2 * delta)
            grads.append(grad)
        return np.mean(grads, axis=0)

    def _select_next_points(self, batch_size, active_models):
        candidate_points = self._sample_points(100 * batch_size)
        acquisition_values = self._acquisition_function(candidate_points, active_models)
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]

        # Adaptive Gradient-Enhanced Local search
        for i in range(batch_size):
            def obj_func(x):
                x = x.reshape(1, -1)
                return np.mean([model.predict(x)[0] for model in active_models])

            x0 = next_points[i]
            bounds = [(self.bounds[0][j], self.bounds[1][j]) for j in range(self.dim)]

            # Uncertainty-aware local search
            uncertainty = np.mean([model.predict(x0.reshape(1, -1), return_std=True)[1] for model in active_models])

            # KDE density
            if self.X is not None and len(self.X) > self.dim + 1:
                bandwidth = self._adaptive_bandwidth()
                kde = KernelDensity(bandwidth=bandwidth).fit(self.X)
                kde_score = kde.score_samples(x0.reshape(1, -1))[0]
                kde_density = np.exp(kde_score)
            else:
                kde_density = 0.0

            maxiter = int(5 + 10 * uncertainty * (1 - kde_density))  # More iterations for higher uncertainty and lower density
            maxiter = min(maxiter, 20)

            # Gradient-enhanced optimization
            def grad_func(x):
                return self._gradient(x, active_models)

            options = {'maxiter': maxiter, 'ftol': 1e-4}  # Reduced ftol
            res = minimize(obj_func, x0, method='L-BFGS-B', jac=grad_func, bounds=bounds, options=options)
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
            sigmas = []
            for model in active_models:
                _, sigma = model.predict(self.X, return_std=True)
                sigmas.append(np.mean(sigma))
            avg_sigma = np.mean(sigmas)

            if avg_sigma > self.uncertainty_threshold:
                batch_size = self.max_batch_size
                exploration_decay = 0.99
            else:
                batch_size = self.min_batch_size
                exploration_decay = 0.999

            batch_size = min(batch_size, remaining_evals)  # Adjust batch size to budget

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
## Feedback
 The algorithm ABETSALS_GELSBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1711 with standard deviation 0.1072.

took 655.39 seconds to run.