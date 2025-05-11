# Description
**Adaptive Ensemble Batch-size Density-Aware Hybrid Bayesian Optimization with Thompson Sampling and Local Search (AEBDAHTSLSBO):** This algorithm integrates the strengths of AETSALS_KDEBO and ABDAHLSBO, featuring an adaptive ensemble of Gaussian Process Regression (GPR) models with Thompson Sampling for efficient acquisition, adaptive batch size based on GPR uncertainty, density-aware exploration using Kernel Density Estimation (KDE), and uncertainty-aware local search. It dynamically adjusts the ensemble size, KDE bandwidth, exploration weight, and local search radius. A key improvement is the use of a more robust local search strategy that incorporates both the uncertainty from the GPR models and the density information from the KDE to guide the search direction and step size. Additionally, the acquisition function is enhanced with a dynamic weighting scheme that balances Expected Improvement (EI), distance-based exploration, and KDE-based exploration.

# Justification
This algorithm combines the advantages of ensemble modeling (AETSALS_KDEBO) and adaptive batch size with density-aware exploration (ABDAHLSBO) to achieve a more robust and efficient optimization process.

*   **Adaptive Ensemble with Thompson Sampling:** Employs an ensemble of GPR models to better capture the function landscape and mitigate the risk of relying on a single model. Thompson Sampling provides an efficient way to balance exploration and exploitation within the ensemble.
*   **Adaptive Batch Size:** Adjusts the batch size based on the uncertainty of the GPR predictions, allowing for more exploration when uncertainty is high and more exploitation when uncertainty is low.
*   **Density-Aware Exploration with KDE:** Uses KDE to focus the search on high-density regions of promising solutions, improving the efficiency of exploration.
*   **Uncertainty-Aware Local Search:** Refines the solutions found by the global search by performing local search around the best solution found so far. The local search is guided by the uncertainty estimates from the GPR models, allowing for more aggressive search in regions with high uncertainty. The local search radius is also adapted based on the optimization progress and uncertainty.
*   **Dynamic Weighting of Acquisition Function Components:** Dynamically adjusts the weights of EI, distance-based exploration, and KDE-based exploration in the acquisition function to balance exploration and exploitation.
*   **Improved Local Search Strategy:** The local search is enhanced by incorporating both the uncertainty from the GPR models and the density information from the KDE. This allows the local search to more effectively navigate the function landscape and find better solutions.

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

class AEBDAHTSLSBO:
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
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
            self.models.append(model)

        self.max_batch_size = min(10, dim)
        self.min_batch_size = 1
        self.local_search_step_size_factor = 0.1
        self.uncertainty_threshold = 0.5
        self.exploration_weight = 0.1
        self.exploration_decay = 0.995
        self.min_exploration = 0.01
        self.kde_bandwidth = 1.0  # Initial KDE bandwidth
        self.local_search_radius = 0.1

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Adapt ensemble size
        n_models = max(self.min_models, int(self.max_models * (1 - self.n_evals / self.budget)))
        active_models = self.models[:n_models]
        for model in active_models:
            model.fit(X, y)
        return active_models

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
        mu = np.mean([model.predict(X) for model in active_models], axis=0).reshape(-1, 1)
        sigma = np.mean(sigmas, axis=1).reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Distance-based exploration term
        if self.X is not None:
            min_dist = np.min(np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2), axis=1, keepdims=True)
            exploration = min_dist / np.max(min_dist)
        else:
            exploration = np.ones_like(ei)

        # KDE-based exploration term
        if self.X is not None and len(self.X) > self.dim + 1:
            bandwidth = self._adaptive_bandwidth()
            kde = KernelDensity(bandwidth=bandwidth).fit(self.X)
            log_dens = kde.score_samples(X)
            kde_exploration = np.exp(log_dens).reshape(-1, 1)
            kde_exploration = kde_exploration / np.max(kde_exploration)
        else:
            kde_exploration = np.zeros_like(ei)

        # Dynamic weighting
        exploration_weight = np.clip(1.0 - self.n_evals / self.budget, 0.1, 1.0)
        kde_weight = 1.0 - exploration_weight

        acquisition = ei + self.exploration_weight * exploration + kde_weight * kde_exploration
        return acquisition

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

        # Adaptive Local search
        for i in range(batch_size):
            def obj_func(x):
                x = x.reshape(1, -1)
                return np.mean([model.predict(x)[0] for model in active_models])

            x0 = next_points[i]
            bounds = [(self.bounds[0][j], self.bounds[1][j]) for j in range(self.dim)]

            # Uncertainty-aware local search
            uncertainty = np.mean([model.predict(x0.reshape(1, -1), return_std=True)[1] for model in active_models])

            # KDE score for local search direction
            if self.X is not None and len(self.X) > self.dim + 1:
                bandwidth = self._adaptive_bandwidth()
                kde = KernelDensity(bandwidth=bandwidth).fit(self.X)
                kde_score = kde.score_samples(x0.reshape(1, -1))[0]
                kde_gradient = self._approximate_kde_gradient(kde, x0)
                # Normalize KDE gradient
                if np.linalg.norm(kde_gradient) > 0:
                    kde_gradient = kde_gradient / np.linalg.norm(kde_gradient)
            else:
                kde_gradient = np.zeros(self.dim)

            # Combine uncertainty and KDE gradient for local search direction
            search_direction = uncertainty * np.random.randn(self.dim) + 0.1 * kde_gradient

            # Adaptive step size
            step_size = self.local_search_step_size_factor * uncertainty
            x0 = x0 + step_size * search_direction  # Move along the search direction
            x0 = np.clip(x0, self.bounds[0], self.bounds[1]) # Clip to bounds
            next_points[i] = x0

        return next_points

    def _approximate_kde_gradient(self, kde, x, epsilon=1e-5):
        """Approximate the gradient of the KDE using finite differences."""
        gradient = np.zeros_like(x)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            log_density_plus = kde.score_samples(x_plus.reshape(1, -1))[0]
            log_density_minus = kde.score_samples(x_minus.reshape(1, -1))[0]
            gradient[i] = (log_density_plus - log_density_minus) / (2 * epsilon)
        return gradient

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
            sigmas = np.array([model.predict(self.X, return_std=True)[1] for model in active_models])
            avg_sigma = np.mean(sigmas)

            if avg_sigma > self.uncertainty_threshold:
                batch_size = self.max_batch_size
                exploration_decay = 0.99
            else:
                batch_size = self.min_batch_size
                exploration_decay = 0.999

            batch_size = min(batch_size, remaining_evals)

            # Adjust local search radius
            self.local_search_radius = 0.1 * (1 - self.n_evals / self.budget)

            next_X = self._select_next_points(batch_size, active_models)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            active_models = self._fit_model(self.X, self.y)

            # Update exploration weight
            self.exploration_weight = max(self.exploration_weight * exploration_decay, self.min_exploration)

            # Update KDE bandwidth
            self.kde_bandwidth = np.std(self.X) / (self.n_evals**0.2)  # Adjust bandwidth based on data spread

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AEBDAHTSLSBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1521 with standard deviation 0.0954.

took 458.64 seconds to run.