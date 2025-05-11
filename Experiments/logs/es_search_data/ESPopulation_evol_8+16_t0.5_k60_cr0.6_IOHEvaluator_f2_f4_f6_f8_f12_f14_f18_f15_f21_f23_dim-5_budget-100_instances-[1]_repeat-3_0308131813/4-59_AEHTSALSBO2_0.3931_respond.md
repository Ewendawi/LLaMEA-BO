# Description
**Adaptive Ensemble Hybrid Bayesian Optimization with Thompson Sampling, Uncertainty-Aware Local Search, and Exploration-Exploitation Balancing (AEHTSALSBO2):** This algorithm combines the strengths of ABETSALSBO and AHBBO_ABSLS. It employs an adaptive ensemble of Gaussian Process Regression (GPR) models, Thompson Sampling for efficient acquisition, and uncertainty-aware local search. It also incorporates an adaptive exploration-exploitation strategy based on both the optimization progress and the GPR model uncertainty, similar to ABETSALSBO, but with a simplified ensemble management and local search to improve computational efficiency. The batch size is dynamically adjusted based on the uncertainty estimates from the GPR model.

# Justification
This algorithm aims to improve upon ABETSALSBO by simplifying the ensemble management and refining the local search strategy to reduce computational cost while retaining its key advantages. It also integrates the adaptive exploration-exploitation strategy from AHBBO_ABSLS.
- **Ensemble of GPR Models:** Using an ensemble of GPR models improves the robustness of the surrogate model, capturing different aspects of the function landscape. The ensemble size is fixed to reduce computational overhead.
- **Thompson Sampling:** Thompson Sampling is used for efficient acquisition, balancing exploration and exploitation within the ensemble.
- **Uncertainty-Aware Local Search:** The local search strategy refines selected points based on the uncertainty estimates from the GPR models. The number of local search iterations is reduced to improve computational efficiency.
- **Adaptive Batch Size:** The batch size is dynamically adjusted based on the uncertainty estimates from the GPR model, increasing when the model uncertainty is high and decreasing when the model is confident.
- **Adaptive Exploration-Exploitation:** The exploration weight in the acquisition function is dynamically adjusted based on the optimization progress, shifting the focus from exploration to exploitation as the number of evaluations increases.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class AEHTSALSBO2:
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

        self.n_models = 3  # Fixed number of surrogate models in the ensemble
        self.models = []
        for i in range(self.n_models):
            length_scale = 1.0 * (i + 1) / self.n_models
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
            self.models.append(model)

        self.max_batch_size = min(10, dim)
        self.min_batch_size = 1
        self.local_search_step_size_factor = 0.1
        self.uncertainty_threshold = 0.5
        self.exploration_weight = 0.2
        self.exploration_weight_min = 0.01

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def _acquisition_function(self, X):
        # Thompson Sampling
        sampled_values = np.zeros((X.shape[0], len(self.models)))
        sigmas = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            mu, sigma = model.predict(X, return_std=True)
            mu = mu.reshape(-1, 1)
            sigma = sigma.reshape(-1, 1)
            sampled_values[:, i] = np.random.normal(mu.flatten(), sigma.flatten())
            sigmas[:, i] = sigma.flatten()

        acquisition = np.mean(sampled_values, axis=1, keepdims=True)
        acquisition = acquisition.reshape(-1, 1)

        # Hybrid acquisition function (EI + exploration)
        mu = np.mean([model.predict(X) for model in self.models], axis=0).reshape(-1, 1)
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

    def _select_next_points(self, batch_size):
        candidate_points = self._sample_points(100 * batch_size)
        acquisition_values = self._acquisition_function(candidate_points)
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]

        # Adaptive Local search
        for i in range(batch_size):
            def obj_func(x):
                x = x.reshape(1, -1)
                return np.mean([model.predict(x)[0] for model in self.models])

            x0 = next_points[i]
            bounds = [(self.bounds[0][j], self.bounds[1][j]) for j in range(self.dim)]

            # Uncertainty-aware local search
            uncertainty = np.mean([model.predict(x0.reshape(1, -1), return_std=True)[1] for model in self.models])
            maxiter = 3  # Reduced number of local search iterations

            # Adaptive step size
            step_size = self.local_search_step_size_factor * uncertainty
            options = {'maxiter': maxiter, 'ftol': 1e-4}
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

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            remaining_evals = self.budget - self.n_evals

            # Adjust batch size based on uncertainty
            sigmas = np.array([model.predict(self.X, return_std=True)[1] for model in self.models])
            avg_sigma = np.mean(sigmas)

            if avg_sigma > self.uncertainty_threshold:
                batch_size = self.max_batch_size
            else:
                batch_size = self.min_batch_size

            batch_size = min(batch_size, remaining_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self._fit_model(self.X, self.y)

            # Update exploration weight
            self.exploration_weight = max(self.exploration_weight_min, self.exploration_weight * (1 - self.n_evals / self.budget))

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AEHTSALSBO2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1683 with standard deviation 0.0993.

took 721.43 seconds to run.