# Description
**Contextual Evolutionary Trust Region Bayesian Optimization (CETRBO):** This algorithm synergistically combines the strengths of AdaptiveEvolutionaryTrustRegionBO (AETRBO) and AdaptiveContextualTrustRegionBO (ACTRBO). It employs a Gaussian Process (GP) surrogate model within an adaptive trust region. Inside the trust region, it leverages differential evolution (DE) to efficiently search for points that optimize a context-aware acquisition function. This acquisition function not only considers the Lower Confidence Bound (LCB) for balancing exploration and exploitation but also incorporates a context penalty based on the distances to the nearest neighbors in the evaluated points. This penalty discourages sampling points too close to existing ones, promoting diversity and preventing premature convergence. The trust region radius is adaptively adjusted based on the success of previous iterations. To improve computational efficiency, the GP kernel parameters are fixed.

# Justification
The rationale behind CETRBO is to address the limitations of AETRBO and ACTRBO individually.

*   **Combining Strengths:** AETRBO uses DE for efficient search within the trust region, while ACTRBO uses a context-aware acquisition function to promote diversity. Combining these addresses both efficient local search and global exploration.
*   **Context-Awareness:** The context penalty in the acquisition function, inspired by ACTRBO, helps to avoid redundant sampling in crowded regions and encourages exploration of less-explored areas.
*   **Adaptive Trust Region:** The adaptive trust region mechanism, similar to both AETRBO and ACTRBO, allows the algorithm to focus the search on promising regions while dynamically adjusting the search space based on the success of previous iterations.
*   **Fixed Kernel:** To reduce computational overhead, the GP kernel parameters are fixed, as tuning them dynamically (as in AdaptiveBayesianEvolutionaryBO) can be very expensive.
*   **Efficiency:** By using DE within the trust region, the algorithm efficiently explores the local landscape defined by the GP, while the context penalty guides the search toward more diverse and potentially promising regions. The adaptive trust region further enhances efficiency by focusing computational resources on promising areas.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution
from sklearn.neighbors import NearestNeighbors

class ContextualEvolutionaryTrustRegionBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(5*dim, self.budget//10)

        self.gp = None
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0
        self.radius_decay = 0.95
        self.radius_increase = 1.1
        self.min_radius = 0.1
        self.success_ratio = 0.0
        self.random_restart_prob = 0.05
        self.de_pop_size = 10
        self.lcb_kappa = 2.0
        self.knn = NearestNeighbors(n_neighbors=5)
        self.context_penalty = 0.1

    def _sample_points(self, n_points):
        if self.best_x is None or np.random.rand() < self.random_restart_prob:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            points = []
            while len(points) < n_points:
                sample = np.random.uniform(-1, 1, size=(self.dim,))
                sample = self.best_x + self.trust_region_radius * sample
                sample = np.clip(sample, self.bounds[0], self.bounds[1])
                points.append(sample)
            return np.array(points)

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        self.gp.fit(X, y)
        self.knn.fit(X)
        return self.gp

    def _acquisition_function(self, X):
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            mu, sigma = self.gp.predict(X, return_std=True)
            sigma = np.clip(sigma, 1e-9, np.inf)
            LCB = mu - self.lcb_kappa * sigma

            distances, _ = self.knn.kneighbors(X)
            context_penalty = np.mean(distances, axis=1).reshape(-1, 1)
            acquisition = LCB + self.context_penalty * sigma #context_penalty reduces LCB, promoting exploration
            return acquisition.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        if self.best_x is None:
            return self._sample_points(batch_size)

        def de_objective(x):
            return self._acquisition_function(x.reshape(1, -1))[0, 0]

        de_bounds = [(max(self.bounds[0][i], self.best_x[i] - self.trust_region_radius),
                      min(self.bounds[1][i], self.best_x[i] + self.trust_region_radius)) for i in range(self.dim)]

        # Adjust maxiter based on remaining budget
        maxiter = max(1, self.budget // (self.de_pop_size * self.dim * 2) - self.n_evals//(self.de_pop_size * self.dim * 2))
        result = differential_evolution(de_objective, de_bounds, popsize=self.de_pop_size, maxiter=maxiter, tol=0.01, disp=False)

        return result.x.reshape(1, -1)

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]
            self.success_ratio = 1.0
        else:
            self.success_ratio *= 0.75

    def _adjust_trust_region(self):
        if self.success_ratio > 0.5:
            self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        batch_size = min(1, self.dim)
        while self.n_evals < self.budget:
            self._fit_model(self.X, self.y)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self._adjust_trust_region()

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ContextualEvolutionaryTrustRegionBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1769 with standard deviation 0.0943.

took 377.83 seconds to run.