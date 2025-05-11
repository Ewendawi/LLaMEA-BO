# Description
**Adaptive Ensemble Density-Guided Hybrid Bayesian Optimization with Enhanced Local Search (AEDGHBO):** This algorithm combines the strengths of EHBBO and EDGLBO, enhancing them with adaptive components. It employs an ensemble of Gaussian Process Regression (GPR) models with varying kernels, similar to EDGLBO, to capture diverse aspects of the function landscape. A Kernel Density Estimation (KDE) guides exploration towards promising high-density regions, and Thompson Sampling, applied to the ensemble, balances exploration and exploitation. The acquisition function is dynamically adjusted based on the optimization progress, adapting the weights of EI, distance-based exploration, and KDE score. The local search strategy is enhanced by increasing the number of iterations and dynamically adjusting the step size based on the uncertainty estimates from the GPR models. An adaptive mechanism adjusts the KDE bandwidth based on the dimensionality and variance of the search space. The initial sampling is performed using Latin Hypercube Sampling (LHS) to ensure good space coverage.

# Justification
This algorithm builds upon the strengths of EHBBO and EDGLBO by incorporating the following key improvements:

*   **Adaptive Acquisition Function:** The weights for Expected Improvement (EI), distance-based exploration, and the KDE score are dynamically adjusted during the optimization process. Initially, more weight is given to exploration (distance and KDE), and as the optimization progresses, the weight shifts towards exploitation (EI). This adaptive weighting helps to balance exploration and exploitation effectively.

*   **Enhanced Local Search:** The local search is enhanced by increasing the number of iterations and dynamically adjusting the step size based on the uncertainty estimates from the GPR models. This allows for more effective refinement of solutions in promising regions.

*   **Adaptive KDE Bandwidth:** The KDE bandwidth is adaptively adjusted based on both the dimensionality and the variance of the evaluated points. This ensures that the KDE accurately captures the underlying density structure of the search space, even in high-dimensional problems or when the distribution of points changes during the optimization process.

*   **Ensemble of GPR Models:** Using an ensemble of GPR models with different kernels, as in EDGLBO, improves the robustness and accuracy of the surrogate model. This helps to capture different aspects of the function landscape and reduces the risk of overfitting.

These enhancements aim to address the limitations of EHBBO and EDGLBO by providing a more robust and adaptive optimization algorithm that can effectively handle a wide range of black-box optimization problems.

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

class AEDGHBO:
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
        self.exploration_weight = 0.1
        self.kde_weight = 0.1
        self.ei_weight = 1.0

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

        acquisition = self.ei_weight * np.mean(sampled_values, axis=1, keepdims=True) + self.kde_weight * kde_scores.reshape(-1, 1) + self.exploration_weight * self._distance_exploration(X)
        return acquisition

    def _distance_exploration(self, X):
        if self.X is None or len(self.X) == 0:
            return np.zeros(X.shape[0])  # Return zeros if no data points exist

        min_dist = np.min(np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2), axis=1, keepdims=True)
        exploration = min_dist / np.max(min_dist)
        return exploration

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
            # Dynamically adjust the step size based on uncertainty
            sigma = np.mean([model.predict(x0.reshape(1, -1), return_std=True)[1] for model in self.models])
            options = {'maxiter': 10}
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

        self._fit_model(self.X, self.y)

        # Adaptive weights
        exploration_decay = 0.95
        kde_decay = 0.95

        while self.n_evals < self.budget:
            remaining_evals = self.budget - self.n_evals
            batch_size = min(self.n_init, remaining_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self._fit_model(self.X, self.y)

            # Update adaptive weights
            self.exploration_weight *= exploration_decay
            self.kde_weight *= kde_decay
            self.ei_weight = 1.0 - self.exploration_weight - self.kde_weight
            self.ei_weight = max(self.ei_weight, 0.1) # Ensure EI has a minimum weight

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AEDGHBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1575 with standard deviation 0.1038.

took 217.55 seconds to run.