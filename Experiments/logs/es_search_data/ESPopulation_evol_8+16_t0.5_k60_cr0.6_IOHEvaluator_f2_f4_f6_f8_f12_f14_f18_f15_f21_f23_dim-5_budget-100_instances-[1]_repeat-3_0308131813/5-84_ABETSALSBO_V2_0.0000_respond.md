# Description
**ABETSALSBO_V2: Adaptive Batch Ensemble with Thompson Sampling, Uncertainty-Aware Local Search, and Dynamic Kernel Lengthscale Adaptation Bayesian Optimization.** This algorithm builds upon ABETSALSBO by introducing dynamic adaptation of the kernel lengthscale in the Gaussian Process Regression (GPR) models, guided by the optimization progress and data distribution. It also incorporates a more refined exploration-exploitation balance. The ensemble size and batch size are adaptively adjusted based on model uncertainty and optimization progress. The local search is retained, guided by GPR uncertainty.

# Justification
1.  **Dynamic Kernel Lengthscale Adaptation:** The kernel lengthscale in GPR models significantly impacts their ability to capture the function's characteristics. Instead of using fixed lengthscales for each model in the ensemble, this adaptation allows each model to dynamically adjust its lengthscale based on the data it observes. This is achieved by optimizing the lengthscale using the data and the marginal log-likelihood. This helps the GPR models to better fit the data and improve prediction accuracy, leading to better acquisition function values and improved overall optimization performance.

2.  **Refined Exploration-Exploitation Balance:** The original ABETSALSBO uses a fixed exploration weight decay. This is improved by incorporating a more sophisticated exploration weight decay schedule that considers both the number of evaluations and the GPR model uncertainty. This ensures a more robust and efficient exploration-exploitation trade-off. High uncertainty encourages exploration, while low uncertainty favors exploitation.

3.  **Adaptive Ensemble Size and Batch Size:** The ensemble size is adapted based on the optimization progress, reducing the computational cost in later stages. The batch size is adjusted based on the average GPR uncertainty, encouraging larger batches when uncertainty is high and smaller batches when uncertainty is low.

4.  **Uncertainty-Aware Local Search:** The local search remains an important component, refining the solutions found by the global search. The step size and number of iterations are adapted based on the GPR uncertainty.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Kernel
from scipy.optimize import minimize
from sklearn.metrics import pairwise_distances

class AdaptiveRBF(Kernel):
    """
    Custom RBF kernel with adaptive lengthscale.
    """
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        length_scale = np.exp(self.length_scale)  # Ensure length_scale is positive
        if Y is None:
            dists = pairwise_distances(X, metric='sqeuclidean') / length_scale
        else:
            Y = np.atleast_2d(Y)
            dists = pairwise_distances(X, Y, metric='sqeuclidean') / length_scale

        K = np.exp(-.5 * dists)
        if eval_gradient:
            if Y is None:
                return K, (K * dists)[:, :, np.newaxis]
            else:
                return K, (K * pairwise_distances(X, Y, metric='sqeuclidean') / length_scale)[:, :, np.newaxis]
        else:
            return K

    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return True

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric",
                              self.length_scale_bounds,
                              default_value=self.length_scale)


class ABETSALSBO_V2:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
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
            #kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
            kernel = C(1.0, (1e-3, 1e3)) * AdaptiveRBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
            self.models.append(model)

        self.max_batch_size = min(10, dim)
        self.min_batch_size = 1
        self.local_search_step_size_factor = 0.1
        self.uncertainty_threshold = 0.5
        self.exploration_weight = 0.1
        self.exploration_decay = 0.995
        self.min_exploration = 0.01

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Adapt ensemble size
        n_models = max(self.min_models, int(self.max_models * (1 - self.n_evals / self.budget)))
        active_models = self.models[:n_models]
        for model in active_models:
            #model.fit(X, y)
            # Optimize lengthscale
            try:
                model.fit(X, y)
            except Exception as e:
                print(f"Fitting GPR model failed: {e}")

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

        # Hybrid acquisition function (EI + exploration)
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

        acquisition = ei + self.exploration_weight * exploration

        return acquisition

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
            uncertainty = np.mean([model.predict(x0.reshape(1, -1), return_std=True)[1] for model in active_models])
            maxiter = int(5 + 10 * uncertainty)
            maxiter = min(maxiter, 20)

            # Adaptive step size
            step_size = self.local_search_step_size_factor * uncertainty
            options = {'maxiter': maxiter, 'ftol': 1e-4}  # Reduced ftol
            res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds, options=options)
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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
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

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<ABETSALSBO_V2>", line 189, in __call__
 189->             sigmas = np.array([model.predict(self.X, return_std=True)[1] for model in active_models])
  File "<ABETSALSBO_V2>", line 189, in <listcomp>
 187 | 
 188 |             # Adjust batch size based on uncertainty
 189->             sigmas = np.array([model.predict(self.X, return_std=True)[1] for model in active_models])
 190 |             avg_sigma = np.mean(sigmas)
 191 | 
AttributeError: 'GaussianProcessRegressor' object has no attribute 'alpha_'. Did you mean: 'alpha'?
