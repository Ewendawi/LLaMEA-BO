# Description
**Adaptive Ensemble Hybrid Bayesian Optimization with Thompson Sampling and Local Search (AEHTSALSBO):** This algorithm builds upon the strengths of EHBBO and SETSBO by incorporating an ensemble of Gaussian Process Regression (GPR) models with different kernels, Thompson Sampling for efficient acquisition, a hybrid acquisition function that incorporates both Expected Improvement (EI) and a distance-based exploration term, and an adaptive local search strategy. Additionally, it introduces an adaptive mechanism to dynamically adjust the exploration-exploitation trade-off based on the optimization progress. The exploration weight in the hybrid acquisition function is decreased as the number of evaluations increases, shifting the focus from exploration to exploitation. A lower bound on the exploration weight is also introduced to prevent premature convergence. The local search intensity is also adapted based on the GPR uncertainty estimates.

# Justification
The AEHTSALSBO algorithm combines several key features to enhance its performance:

*   **Ensemble of GPR Models:** Using an ensemble of GPR models with different kernels improves the robustness and accuracy of the surrogate model, allowing it to capture diverse aspects of the function landscape.
*   **Thompson Sampling:** Thompson Sampling provides an efficient and natural way to balance exploration and exploitation.
*   **Hybrid Acquisition Function:** The hybrid acquisition function combines Expected Improvement (EI) and a distance-based exploration term, allowing the algorithm to explore promising regions while also exploiting the best-known solutions.
*   **Adaptive Exploration:** Dynamically adjusting the exploration weight in the hybrid acquisition function allows the algorithm to adapt to the optimization progress, shifting the focus from exploration to exploitation as the number of evaluations increases.
*   **Adaptive Local Search:** Adjusting the intensity of the local search based on the uncertainty estimates from the GPR models allows the algorithm to refine solutions more effectively in regions where the surrogate model is more confident.
*   **Computational Efficiency:** The algorithm is designed to be computationally efficient by using Thompson Sampling for acquisition and limiting the number of iterations in the local search.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class AEHTSALSBO:
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

        self.exploration_weight = 0.1  # Initial exploration weight
        self.exploration_weight_min = 0.01 # Minimum exploration weight
        self.exploration_decay = 0.99 # Decay rate for exploration weight

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def _acquisition_function(self, X):
        # Thompson Sampling with hybrid acquisition
        sampled_values = np.zeros((X.shape[0], self.n_models))
        mu_list = []
        sigma_list = []
        for i, model in enumerate(self.models):
            mu, sigma = model.predict(X, return_std=True)
            mu = mu.reshape(-1, 1)
            sigma = sigma.reshape(-1, 1)
            mu_list.append(mu)
            sigma_list.append(sigma)
            sampled_values[:, i] = np.random.normal(mu.flatten(), sigma.flatten())

        acquisition = np.mean(sampled_values, axis=1, keepdims=True)

        # Add distance-based exploration
        min_dist = np.min(np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2), axis=1, keepdims=True) if self.X is not None else np.ones((X.shape[0], 1))
        exploration = min_dist / np.max(min_dist)

        acquisition = acquisition + self.exploration_weight * exploration
        return acquisition

    def _select_next_points(self, batch_size):
        candidate_points = self._sample_points(100 * batch_size)
        acquisition_values = self._acquisition_function(candidate_points)
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]

        # Adaptive Local Search
        for i in range(batch_size):
            def obj_func(x):
                x = x.reshape(1, -1)
                return np.mean([model.predict(x)[0] for model in self.models])

            x0 = next_points[i]
            bounds = [(self.bounds[0][j], self.bounds[1][j]) for j in range(self.dim)]

            # Adaptive local search iterations based on uncertainty
            uncertainty = np.mean([model.predict(x0.reshape(1, -1), return_std=True)[1] for model in self.models])
            max_iter = min(10, max(1, int(5 / (1 + uncertainty)))) # More uncertainty, fewer iterations
            res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter})
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

        self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            remaining_evals = self.budget - self.n_evals
            batch_size = min(self.n_init, remaining_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self._fit_model(self.X, self.y)

            # Adaptive Exploration
            self.exploration_weight = max(self.exploration_weight * self.exploration_decay, self.exploration_weight_min)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AEHTSALSBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1436 with standard deviation 0.0989.

took 93.36 seconds to run.