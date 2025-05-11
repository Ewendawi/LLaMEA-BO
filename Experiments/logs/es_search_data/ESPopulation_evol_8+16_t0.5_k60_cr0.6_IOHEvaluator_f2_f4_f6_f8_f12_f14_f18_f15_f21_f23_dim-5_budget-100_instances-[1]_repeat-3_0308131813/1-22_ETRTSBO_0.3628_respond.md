# Description
**Ensemble Trust Region with Thompson Sampling Bayesian Optimization (ETRTSBO):** This algorithm combines the strengths of SETSBO and ATRBO. It uses an ensemble of Gaussian Process Regression models with Thompson Sampling for acquisition, similar to SETSBO, to provide a more robust estimate of the function landscape. It also incorporates an adaptive trust region, like ATRBO, to manage the exploration-exploitation trade-off dynamically. The key enhancement is that the trust region is adapted based on the average performance of the ensemble, and the local search within the trust region is guided by the ensemble's predictions. This aims to improve both the accuracy and the adaptability of the optimization process.

# Justification
*   **Ensemble of Surrogates:** Using an ensemble of GPR models, as in SETSBO, improves the robustness of the surrogate model, especially when dealing with complex or multi-modal functions. Different kernels in the ensemble capture different aspects of the function landscape.
*   **Thompson Sampling:** This acquisition function provides a computationally efficient way to balance exploration and exploitation. It naturally samples from the posterior distribution, favoring regions with high uncertainty and/or high predicted values.
*   **Adaptive Trust Region:** The trust region framework, adapted from ATRBO, dynamically adjusts the search space based on the agreement between the surrogate model's predictions and the actual function evaluations. This helps to focus the search on promising regions while still allowing for exploration.
*   **Ensemble-Guided Trust Region Adaptation:** The trust region adaptation is based on the average prediction of the ensemble, making it more robust to individual model inaccuracies.
*   **Local Search within Trust Region:** Local search refines the search within the trust region, improving exploitation. The ensemble's average prediction guides the local search.
*   **Computational Efficiency:** The algorithm aims to maintain computational efficiency by using Thompson Sampling and limiting the iterations of local search.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ETRTSBO:
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

        self.n_models = 3  # Number of surrogate models in the ensemble
        self.models = []
        for i in range(self.n_models):
            length_scale = 1.0 * (i + 1) / self.n_models  # Varying length scales
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
            self.models.append(model)

        self.trust_region_center = np.zeros(dim)
        self.trust_region_radius = 2.5  # Initial trust region radius
        self.radius_min = 0.1
        self.radius_max = 5.0
        self.gamma_inc = 2.0
        self.gamma_dec = 0.5
        self.eta_good = 0.9
        self.eta_bad = 0.1

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        scaled_sample = qmc.scale(sample, self.bounds[0], self.bounds[1])

        # Clip to trust region
        for i in range(n_points):
            if np.linalg.norm(scaled_sample[i] - self.trust_region_center) > self.trust_region_radius:
                direction = scaled_sample[i] - self.trust_region_center
                direction = direction / np.linalg.norm(direction)
                scaled_sample[i] = self.trust_region_center + direction * self.trust_region_radius

        return scaled_sample

    def _fit_model(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def _acquisition_function(self, X):
        # Thompson Sampling with Ensemble
        sampled_values = np.zeros((X.shape[0], self.n_models))
        for i, model in enumerate(self.models):
            mu, sigma = model.predict(X, return_std=True)
            mu = mu.reshape(-1, 1)
            sigma = sigma.reshape(-1, 1)
            sampled_values[:, i] = np.random.normal(mu.flatten(), sigma.flatten())

        acquisition = np.mean(sampled_values, axis=1, keepdims=True)
        return acquisition

    def _select_next_point(self):
        def obj_func(x):
            x = x.reshape(1, -1)
            return -self._acquisition_function(x)[0][0]

        x0 = self.trust_region_center
        bounds = [(max(self.bounds[0][i], self.trust_region_center[i] - self.trust_region_radius),
                   min(self.bounds[1][i], self.trust_region_center[i] + self.trust_region_radius)) for i in range(self.dim)]

        res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5})
        next_point = res.x.reshape(1, -1)
        return next_point

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
        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Optimization
            next_X = self._select_next_point()
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Update the trust region
            ensemble_predictions = np.array([model.predict(next_X)[0] for model in self.models])
            predicted_y = np.mean(ensemble_predictions)
            actual_y = next_y[0][0]
            rho = (self.y[-1][0] - actual_y) / (self.y[-1][0] - predicted_y) if (self.y[-1][0] - predicted_y) != 0 else 0

            if rho < self.eta_bad:
                self.trust_region_radius = max(self.radius_min, self.gamma_dec * self.trust_region_radius)
            else:
                self.trust_region_center = next_X[0]
                if rho > self.eta_good:
                    self.trust_region_radius = min(self.radius_max, self.gamma_inc * self.trust_region_radius)

            self._fit_model(self.X, self.y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ETRTSBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1364 with standard deviation 0.0998.

took 1461.42 seconds to run.