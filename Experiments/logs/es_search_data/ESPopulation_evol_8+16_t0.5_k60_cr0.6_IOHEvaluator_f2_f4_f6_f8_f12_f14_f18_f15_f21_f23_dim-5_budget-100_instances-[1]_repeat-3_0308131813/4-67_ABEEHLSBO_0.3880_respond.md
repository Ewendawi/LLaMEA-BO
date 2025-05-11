# Description
**Adaptive Batch-Size Exploration-Exploitation with Local Search Bayesian Optimization (ABEEHLSBO):** This algorithm combines the adaptive batch size strategy from ABDAHBO, the adaptive exploration-exploitation balance from AEEHBBO, and incorporates a refined local search strategy. It dynamically adjusts the batch size based on the uncertainty estimates from the Gaussian Process Regression (GPR) model. The exploration weight in the hybrid acquisition function is dynamically adjusted based on the optimization progress, decreasing as the number of evaluations increases. The local search is enhanced by using the GPR model's uncertainty estimates to guide the local search iterations and step size. A hybrid acquisition function combines Expected Improvement (EI) and a distance-based exploration term. The KDE component from ABDAHBO is removed to reduce computational cost while retaining the benefits of adaptive batch size and exploration-exploitation balance.

# Justification
The algorithm combines the strengths of ABDAHBO and AEEHBBO while addressing their limitations.
- Adaptive Batch Size: The adaptive batch size strategy from ABDAHBO allows for efficient exploration and exploitation by adjusting the batch size based on the uncertainty of the GPR model.
- Adaptive Exploration-Exploitation: The adaptive exploration-exploitation balance from AEEHBBO dynamically adjusts the exploration weight in the acquisition function, shifting the focus from exploration to exploitation as the optimization progresses.
- Uncertainty-Aware Local Search: The local search is enhanced by using the GPR model's uncertainty estimates to guide the local search iterations and step size, allowing for more efficient refinement of promising solutions.
- Removal of KDE: The KDE component from ABDAHBO is removed to reduce computational cost without sacrificing the benefits of adaptive batch size and exploration-exploitation balance. The exploration term based on minimum distance to existing points is retained.
- Efficient Point Selection: The point selection strategy uses a combination of random sampling and local search around the best point found so far to efficiently select the next points to evaluate.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ABEEHLSBO:
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
        self.max_batch_size = min(10, dim)  # Maximum batch size for selecting points
        self.min_batch_size = 1
        self.exploration_weight = 0.2  # Initial exploration weight
        self.exploration_weight_min = 0.01  # Minimum exploration weight
        self.uncertainty_threshold = 0.5  # Threshold for adjusting batch size
        self.local_search_radius = 0.1

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=5, alpha=1e-5
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

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

        # Hybrid acquisition function
        acquisition = ei + self.exploration_weight * exploration
        return acquisition

    def _select_next_points(self, batch_size):
        # Generate candidate points
        candidate_points = self._sample_points(50 * batch_size)

        # Add points around the best solution (local search)
        if self.best_x is not None:
            # Use GPR uncertainty to guide local search step size
            _, best_sigma = self.model.predict(self.best_x.reshape(1, -1), return_std=True)
            local_search_scale = max(self.local_search_radius * best_sigma, 0.01)  # Ensure a minimum scale

            local_points = np.random.normal(
                loc=self.best_x, scale=local_search_scale, size=(50 * batch_size, self.dim)
            )
            local_points = np.clip(local_points, self.bounds[0], self.bounds[1])
            candidate_points = np.vstack((candidate_points, local_points))

        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_points)

        # Select the top batch_size points with the highest acquisition values
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]

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

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            remaining_evals = self.budget - self.n_evals

            # Adjust batch size based on uncertainty
            _, sigma = self.model.predict(self.X, return_std=True)
            avg_sigma = np.mean(sigma)

            if avg_sigma > self.uncertainty_threshold:
                batch_size = self.max_batch_size
                exploration_decay = 0.99
            else:
                batch_size = self.min_batch_size
                exploration_decay = 0.999

            batch_size = min(batch_size, remaining_evals)  # Adjust batch size to budget

            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self.model = self._fit_model(self.X, self.y)

            # Update exploration weight
            self.exploration_weight = max(
                self.exploration_weight_min,
                self.exploration_weight * (1 - self.n_evals / self.budget),
            )

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ABEEHLSBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1591 with standard deviation 0.1049.

took 436.58 seconds to run.