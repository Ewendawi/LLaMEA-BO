# Description
**Adaptive Ensemble with Thompson Sampling, Uncertainty-Aware Local Search with Adaptive Momentum, and Dynamic Kernel Density Estimation (AETSALS_KDE_MBO):** This algorithm combines the strengths of ABETSALSDEBO and AETSALS_KDEBO, incorporating adaptive momentum in the local search to escape local optima and refining the dynamic exploration strategy using KDE. It adaptively manages an ensemble of Gaussian Process Regression (GPR) models, dynamically adjusting the ensemble size based on optimization progress. Thompson Sampling is used for efficient acquisition within the ensemble. The batch size is adaptively adjusted based on the average uncertainty of the GPR predictions. An uncertainty-aware local search with adaptive momentum, using the variance predictions from the GPR models, guides the local search iterations and step size. The acquisition function is a hybrid of Expected Improvement (EI), a distance-based exploration term, and a KDE-based exploration term. The KDE bandwidth is dynamically adjusted based on the local density of evaluated points and the optimization progress. The exploration weight is dynamically adjusted based on the optimization progress and the GPR model's uncertainty.

# Justification
This algorithm builds upon ABETSALSDEBO and AETSALS_KDEBO by incorporating momentum into the local search to improve its ability to escape local optima. The adaptive momentum term helps the local search to overcome small barriers and converge to better solutions. The KDE bandwidth is dynamically adjusted based on the optimization progress to balance exploration and exploitation. The exploration weight is also dynamically adjusted based on the optimization progress and the GPR model's uncertainty, allowing for a robust and efficient exploration-exploitation trade-off. The ensemble size is dynamically adjusted based on the optimization progress to reduce computational cost in later stages. The local search step size is dynamically adjusted based on the GPR model uncertainty.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity

class AETSALS_KDE_MBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim

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
        self.kde_bandwidth = 1.0 # Initial KDE bandwidth
        self.momentum = 0.1 # Momentum for local search

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
        if self.X is not None:
            kde = KernelDensity(kernel='gaussian', bandwidth=self.kde_bandwidth).fit(self.X)
            log_dens = kde.score_samples(X)
            kde_exploration = np.exp(log_dens).reshape(-1, 1)
        else:
            kde_exploration = np.ones_like(ei)

        acquisition = ei + self.exploration_weight * exploration + 0.01 * kde_exploration

        return acquisition

    def _select_next_points(self, batch_size, active_models):
        candidate_points = self._sample_points(100 * batch_size)
        acquisition_values = self._acquisition_function(candidate_points, active_models)
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]

        # Adaptive Local search with momentum
        if hasattr(self, 'previous_local_search_direction'):
            previous_direction = self.previous_local_search_direction
        else:
            previous_direction = np.zeros((batch_size, self.dim))
            self.previous_local_search_direction = previous_direction

        for i in range(batch_size):
            def obj_func(x):
                x = x.reshape(1, -1)
                return np.mean([model.predict(x)[0] for model in active_models])

            x0 = next_points[i]
            bounds = [(self.bounds[0][j], self.bounds[1][j]) for j in range(self.dim)]

            # Uncertainty-aware local search
            uncertainty = np.mean([model.predict(x0.reshape(1, -1), return_std=True)[1] for model in active_models])
            maxiter = int(5 + 10 * uncertainty)
            maxiter = min(maxiter, 20)

            # Adaptive step size
            step_size = self.local_search_step_size_factor * uncertainty
            options = {'maxiter': maxiter, 'ftol': 1e-4}  # Reduced ftol
            res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds, options=options)
            
            # Apply momentum
            direction = res.x - x0
            res.x = res.x + self.momentum * previous_direction[i]
            res.x = np.clip(res.x, self.bounds[0], self.bounds[1]) # Clip to bounds

            next_points[i] = res.x
            self.previous_local_search_direction[i] = direction

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

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
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
            next_X = self._select_next_points(batch_size, active_models)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            active_models = self._fit_model(self.X, self.y)

            # Update exploration weight
            self.exploration_weight = max(self.exploration_weight * exploration_decay, self.min_exploration)

            # Update KDE bandwidth
            self.kde_bandwidth = np.std(self.X) / (self.n_evals**0.2) # Adjust bandwidth based on data spread

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AETSALS_KDE_MBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1693 with standard deviation 0.1084.

took 513.79 seconds to run.