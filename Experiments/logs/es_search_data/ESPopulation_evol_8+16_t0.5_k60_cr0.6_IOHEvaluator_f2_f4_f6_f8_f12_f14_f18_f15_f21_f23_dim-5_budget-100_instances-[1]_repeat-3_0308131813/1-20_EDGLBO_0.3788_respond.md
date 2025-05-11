# Description
**Ensemble Density-Guided Bayesian Optimization with Local Refinement (EDGLBO):** This algorithm synergistically integrates the strengths of ensemble modeling, density-based exploration, and local search for efficient black-box optimization. It employs an ensemble of Gaussian Process Regression (GPR) models with varying kernels to capture diverse aspects of the function landscape. A Kernel Density Estimation (KDE) guides exploration towards promising high-density regions, while Thompson Sampling, applied to the ensemble, balances exploration and exploitation. Finally, a gradient-based local search refines the solutions within these regions. The algorithm dynamically adjusts the KDE bandwidth to adapt to the data distribution.

# Justification
This algorithm builds upon SETSBO and DensiTreeBO to create a more robust and efficient optimizer.
1.  **Ensemble of Surrogates (SETSBO):** Using an ensemble of GPR models with different kernels improves the robustness of the surrogate model, reducing the risk of overfitting to local features. This is particularly important in complex or multimodal landscapes.
2.  **Density-Guided Exploration (DensiTreeBO):** Integrating KDE helps to identify promising regions in the search space, especially in the early stages of optimization. This allows the algorithm to focus its search on areas where good solutions are likely to be found, improving sample efficiency.
3.  **Thompson Sampling:** Thompson Sampling provides an efficient way to balance exploration and exploitation within the ensemble framework.
4.  **Local Search:** Local search refines the solutions obtained from Thompson Sampling, helping to converge to local optima within the identified promising regions.
5.  **Adaptive KDE Bandwidth:** The KDE bandwidth is dynamically adjusted based on the variance of the data, allowing the algorithm to adapt to different data distributions. This improves the accuracy of the density estimation and the effectiveness of the density-guided exploration.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize

class EDGLBO:
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

        self.n_models = 3
        self.models = []
        for i in range(self.n_models):
            length_scale = 1.0 * (i + 1) / self.n_models
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
            self.models.append(model)

        self.kde_bandwidth = 0.5

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def _acquisition_function(self, X, kde_scores):
        # Thompson Sampling with KDE Prior
        sampled_values = np.zeros((X.shape[0], self.n_models))
        for i, model in enumerate(self.models):
            mu, sigma = model.predict(X, return_std=True)
            mu = mu.reshape(-1, 1)
            sigma = sigma.reshape(-1, 1)
            sampled_values[:, i] = np.random.normal(mu.flatten(), sigma.flatten())

        acquisition = np.mean(sampled_values, axis=1, keepdims=True) + 0.1 * kde_scores.reshape(-1, 1) #Adding KDE scores as a prior
        return acquisition

    def _select_next_points(self, batch_size):
        if self.X is None or len(self.X) < self.dim + 1:
            return self._sample_points(batch_size)

        #Adaptive Bandwidth
        self.kde_bandwidth = np.std(self.X) / 2 if np.std(self.X) > 0 else 0.5

        kde = KernelDensity(bandwidth=self.kde_bandwidth).fit(self.X)
        candidate_points = self._sample_points(100 * batch_size)
        kde_scores = kde.score_samples(candidate_points)

        acquisition_values = self._acquisition_function(candidate_points, kde_scores)
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]

        # Local search
        for i in range(batch_size):
            def obj_func(x):
                x = x.reshape(1, -1)
                return np.mean([model.predict(x)[0] for model in self.models])

            x0 = next_points[i]
            bounds = [(self.bounds[0][j], self.bounds[1][j]) for j in range(self.dim)]
            res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5})
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

        return self.best_y, self.best_x
```
## Feedback
 The algorithm EDGLBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1555 with standard deviation 0.1019.

took 179.15 seconds to run.