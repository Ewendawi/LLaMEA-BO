# Description
**ParetoActiveBO (PABO):** This algorithm uses a Pareto-based approach to manage multiple acquisition functions simultaneously, promoting a diverse set of candidate solutions. It employs a Gaussian Process (GP) surrogate model and considers two acquisition functions: Expected Improvement (EI) for exploitation and a distance-based metric to encourage exploration of less-sampled regions. The Pareto front of non-dominated solutions is maintained, and new points are selected from this front. Active learning is incorporated by querying the function value at the point that maximizes the variance predicted by the GP, thus reducing uncertainty.

# Justification
The ParetoActiveBO (PABO) algorithm is designed to be diverse from previous algorithms in several ways:

1.  **Multi-objective Acquisition:** Instead of using a single acquisition function or a weighted combination, it explicitly uses a Pareto front to balance Expected Improvement (EI) and exploration (diversity). This allows for a more nuanced trade-off between exploitation and exploration.
2.  **Active Learning:** The algorithm incorporates an active learning component by selecting points that maximize the GP's prediction variance. This actively reduces the uncertainty of the surrogate model, which is different from the other algorithms.
3.  **Diversity Metric:** A distance-based diversity metric is used to encourage exploration of less-sampled regions.
4.  **Computational Efficiency:** The algorithm uses efficient implementations of GP prediction and Pareto front computation to maintain computational efficiency.

This approach aims to address the limitations of single acquisition function methods, which can get stuck in local optima or fail to adequately explore the search space. By maintaining a Pareto front and actively reducing uncertainty, PABO seeks to find a better balance between exploration and exploitation, leading to improved performance on a wider range of black-box optimization problems. The error in `GradientEnhancedBO` related to exceeding the budget is avoided by not having a separate local search step.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.spatial.distance import cdist

class ParetoActiveBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(5*dim, self.budget//10)

        # Do not add any other arguments without a default value
        self.gp = None
        self.best_x = None
        self.best_y = float('inf')

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        self.gp.fit(X, y)
        return self.gp

    def _expected_improvement(self, X):
        # Implement Expected Improvement acquisition function
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            mu, sigma = self.gp.predict(X, return_std=True)
            sigma = np.clip(sigma, 1e-9, np.inf)
            imp = self.best_y - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            return ei.reshape(-1, 1)

    def _diversity_metric(self, X):
        # Implement a distance-based diversity metric
        if self.X is None:
            return np.ones((len(X), 1))  # No diversity if no points yet
        else:
            distances = np.min(cdist(X, self.X), axis=1)
            return distances.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using Pareto front
        n_candidates = min(100 * self.dim, self.budget - self.n_evals)
        candidates = self._sample_points(n_candidates)

        # Calculate acquisition function values
        ei = self._expected_improvement(candidates)
        diversity = self._diversity_metric(candidates)

        # Normalize acquisition functions
        ei_normalized = (ei - np.min(ei)) / (np.max(ei) - np.min(ei)) if np.max(ei) != np.min(ei) else np.zeros_like(ei)
        diversity_normalized = (diversity - np.min(diversity)) / (np.max(diversity) - np.min(diversity)) if np.max(diversity) != np.min(diversity) else np.zeros_like(diversity)

        # Combine acquisition functions into a multi-objective matrix
        F = np.hstack([ei_normalized, diversity_normalized])

        # Find the Pareto front
        is_efficient = np.ones(F.shape[0], dtype = bool)
        for i, c in enumerate(F):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(F[is_efficient]>=c, axis=1)  # Keep any point with at least one better objective
                is_efficient[i] = True  # And keep this point

        pareto_front = candidates[is_efficient]

        # Active learning: select point with maximum variance
        if self.gp is not None:
            _, sigma = self.gp.predict(pareto_front, return_std=True)
            next_point = pareto_front[np.argmax(sigma)].reshape(1, -1)
        else:
            # If GP is not fitted yet, select a random point from the Pareto front
            next_point = pareto_front[np.random.choice(len(pareto_front))].reshape(1, -1)

        return next_point

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        # Do not change the function signature
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

        # Update best seen value
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Optimization loop
        batch_size = min(1, self.dim)
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Select points by acquisition function
            next_X = self._select_next_points(batch_size)

            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)

            # Update the evaluated points
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ParetoActiveBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1492 with standard deviation 0.0999.

took 10.52 seconds to run.