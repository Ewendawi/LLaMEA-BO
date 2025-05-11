# Description
Adaptive Batch Bayesian Optimization with Trust Region (ABTRBO) integrates the adaptive trust region approach from ATRBO into the batch-oriented framework of EHBO. It employs a Gaussian Process (GP) with Expected Improvement (EI) for modeling the objective function. A trust region is maintained around the current best point, and points are sampled within this region using a quasi-Monte Carlo (QMC) sequence (Sobol sequence). The size of the trust region is adapted based on the optimization progress, similar to ATRBO. A batch of points is selected within the trust region. The batch size is dynamically adjusted according to the remaining budget and the dimension of the search space, like EHBO, to balance exploration and exploitation. Additionally, L-BFGS-B is used to refine the sampled points by optimizing the EI acquisition function within the trust region. A novel aspect is the dynamic trust region shrinking and expansion mechanism, which adjusts the trust region size and location based on the success of recent evaluations. If a better point is found, the trust region expands, otherwise, it shrinks. The center of the trust region is also updated towards the best point found within the current trust region.

# Justification
This approach combines the benefits of both ATRBO and EHBO. The trust region focuses exploration around promising areas while the batch evaluation and dynamic batch size provide computational efficiency. The L-BFGS-B refinement helps to locate better points within the trust region. The dynamic adaptation of the trust region size and location allows the algorithm to adapt to the landscape of the objective function, promoting both exploration and exploitation. Using EI promotes both exploration and exploitation.
The batch size starts large to promote initial exploration, and then decreases as the remaining budget dwindles, to enable more focused exploitation near the end of the budget, like EHBO.
Additionally, this algorithm attempts to address the weaknesses of each of the other algorithms:

- ATRBO is slow since it explores one point at a time
- EHBO has a static search space.

ABTRBO avoids both of these errors by using batch processing and trust region.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ABTRBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(10 * dim, self.budget // 5)
        self.best_y = float('inf')
        self.best_x = None
        self.trust_region_radius = 2.5  # Initial trust region radius
        self.rho = 0.95  # Shrinking/expanding factor
        self.trust_region_center = (self.bounds[1] + self.bounds[0]) / 2 # Initial center

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points within the trust region
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        points = sampler.random(n=n_points)
        points = qmc.scale(points, -1, 1)  # Scale to [-1, 1]

        # Project points to a hypersphere
        lengths = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / lengths * np.random.uniform(0, 1, size=lengths.shape) ** (1/self.dim)

        points = points * self.trust_region_radius + self.trust_region_center
        
        # Clip to the bounds
        points = np.clip(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp):
        # Implement Expected Improvement acquisition function
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu - 1e-9  # Adding a small constant to avoid division by zero
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        return ei

    def _select_next_points(self, batch_size, gp):
        # Select the next points to evaluate
        # Optimization of acquisition function using L-BFGS-B
        x_starts = self._sample_points(batch_size)  # Multiple starting points
        x_next = []
        for x_start in x_starts:
            res = minimize(lambda x: -self._acquisition_function(x.reshape(1, -1), gp),
                           x_start,
                           bounds=[(max(self.bounds[0][i], self.trust_region_center[i] - self.trust_region_radius),
                                    min(self.bounds[1][i], self.trust_region_center[i] + self.trust_region_radius)) for i in range(self.dim)],
                           method='L-BFGS-B')
            x_next.append(res.x)

        return np.array(x_next)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y
    
    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))
        
        # Update best seen value
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Initialize best_x to a random initial point
        self.best_x = self.X[np.argmin(self.y)].copy()
        self.trust_region_center = self.best_x.copy()

        while self.n_evals < self.budget:
            # Optimization
            gp = self._fit_model(self.X, self.y)

            # Dynamic batch size
            remaining_evals = self.budget - self.n_evals
            batch_size = min(max(1, int(remaining_evals / (self.dim * 0.1))), 20) # Ensure at least 1 point and limit to 20

            # Select next point(s)
            next_X = self._select_next_points(batch_size, gp)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Adjust trust region radius and center
            if np.min(next_y) < self.best_y:
                self.trust_region_radius /= self.rho  # Expand
                self.trust_region_center = self.best_x.copy()  # Move center towards the best point
            else:
                self.trust_region_radius *= self.rho  # Shrink

            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)

        return self.best_y, self.best_x
```