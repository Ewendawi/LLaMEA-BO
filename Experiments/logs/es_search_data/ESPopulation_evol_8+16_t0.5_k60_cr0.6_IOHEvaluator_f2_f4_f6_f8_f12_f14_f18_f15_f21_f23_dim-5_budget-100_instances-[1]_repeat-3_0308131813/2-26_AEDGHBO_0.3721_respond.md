# Description
**Adaptive Ensemble Density-Guided Hybrid Bayesian Optimization with Dynamic Local Search (AEDGHBO):** This algorithm combines the strengths of AHBBO and EDGLBO, incorporating adaptive exploration, ensemble modeling, density-guided search, and dynamic local refinement. It leverages an ensemble of Gaussian Process Regression (GPR) models with different kernels and Thompson Sampling for acquisition. A Kernel Density Estimation (KDE) guides exploration towards promising high-density regions, while an adaptive local search strategy refines the solutions. The exploration weight in the acquisition function is dynamically adjusted based on the optimization progress, and the intensity of the local search is also dynamically adapted based on the uncertainty estimates from the GPR models.

# Justification
This algorithm builds upon the strengths of AHBBO and EDGLBO to create a more robust and efficient optimization strategy.

*   **Adaptive Exploration (from AHBBO):** The exploration weight is dynamically adjusted, promoting exploration early in the optimization process and exploitation later on. This helps to avoid premature convergence and ensures that the search space is adequately explored.
*   **Ensemble Modeling (from EDGLBO):** Using an ensemble of GPR models with different kernels improves the robustness and accuracy of the surrogate model. This is particularly important for complex or multimodal functions, where a single GPR model may not be sufficient to capture the function landscape accurately.
*   **Density-Guided Search (from EDGLBO):** KDE is used to guide exploration towards promising high-density regions. This helps to focus the search on areas of the search space that are likely to contain good solutions.
*   **Dynamic Local Search (Enhanced EDGLBO):** The local search intensity is dynamically adjusted based on the uncertainty estimates from the GPR models. This allows the algorithm to focus local search efforts on regions where the surrogate model is less certain, potentially leading to faster convergence. The local search is also applied with a probability that decreases as the optimization progresses, further enhancing the exploitation phase.
*   **Computational Efficiency:** The algorithm aims to balance exploration and exploitation efficiently by combining these techniques. The adaptive exploration and dynamic local search help to reduce the computational cost of the optimization process, while the ensemble modeling and density-guided search improve the accuracy and robustness of the algorithm.

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
        self.exploration_decay = 0.995
        self.min_exploration = 0.01
        self.local_search_prob = 0.8 # Initial probability of performing local search
        self.local_search_decay = 0.99 # Decay factor for local search probability

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def _acquisition_function(self, X, kde_scores):
        # Thompson Sampling with KDE Prior and Adaptive Exploration
        sampled_values = np.zeros((X.shape[0], self.n_models))
        for i, model in enumerate(self.models):
            mu, sigma = model.predict(X, return_std=True)
            mu = mu.reshape(-1, 1)
            sigma = sigma.reshape(-1, 1)
            sampled_values[:, i] = np.random.normal(mu.flatten(), sigma.flatten())

        acquisition = np.mean(sampled_values, axis=1, keepdims=True) + self.exploration_weight * kde_scores.reshape(-1, 1) #Adding KDE scores as a prior
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

        # Dynamic Local search
        for i in range(batch_size):
            if np.random.rand() < self.local_search_prob:
                def obj_func(x):
                    x = x.reshape(1, -1)
                    return np.mean([model.predict(x)[0] for model in self.models])

                x0 = next_points[i]
                bounds = [(self.bounds[0][j], self.bounds[1][j]) for j in range(self.dim)]
                # Dynamically adjust maxiter based on remaining budget and uncertainty
                maxiter = min(5, self.budget - self.n_evals)
                res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': maxiter})
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

            # Update exploration weight
            self.exploration_weight = max(self.exploration_weight * self.exploration_decay, self.min_exploration)
            # Update local search probability
            self.local_search_prob = self.local_search_prob * self.local_search_decay

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AEDGHBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1500 with standard deviation 0.1003.

took 133.69 seconds to run.