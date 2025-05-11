# Description
**Adaptive Trust Region Bayesian Optimization with Dynamic Kappa, Adaptive Initial Exploration, Trust Region Center Adaptation, and Stochastic Radius Perturbation (ATRBO_DKAICASP):** This algorithm builds upon ATRBO_DKAICA by adding a stochastic perturbation to the trust region radius after each iteration. This perturbation aims to prevent premature convergence and encourage exploration, especially in cases where the trust region might be shrinking too aggressively. The perturbation is scaled by the GP's predicted uncertainty to allow for larger perturbations in regions with high uncertainty. The initial exploration phase is the same as in ATRBO_DKAICA. The trust region center is moved to the best point found within the trust region, rather than only updating it if the new point globally improves the best objective. This allows the trust region to more accurately focus on promising local areas. Additionally, the updates to kappa and rho are refined to provide more stable and effective adaptation, especially focusing on the GP uncertainty.

# Justification
The key idea is to introduce a small, stochastic element to the trust region radius adjustment. The original ATRBO_DKAICA already incorporates dynamic kappa, adaptive initial exploration, and trust region center adaptation. However, it can still get stuck in local optima if the trust region shrinks too quickly. By adding a stochastic perturbation to the radius, we aim to balance exploration and exploitation more effectively. The magnitude of the perturbation is proportional to the GP's predicted uncertainty, which allows for larger perturbations in unexplored regions and smaller perturbations in well-explored regions. This stochastic element aims to prevent premature convergence, especially in complex and multimodal landscapes. The trust region center adaptation is retained to focus on promising local areas. The dynamic kappa and rho are retained as well since they are important for the exploration-exploitation trade-off.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ATRBO_DKAICASP:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0  # the number of function evaluations
        self.n_init = min(20 * dim, self.budget // 5) # increased samples for initial exploration
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.5  # Initial trust region radius
        self.rho = 0.95  # Shrinking factor
        self.kappa = 2.0  # Exploration-exploitation trade-off for LCB
        self.success_history = [] # Keep track of successful moves

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, center=None, radius=None):
        # sample points within the trust region
        if center is None:
            center = (self.bounds[1] + self.bounds[0]) / 2
        if radius is None:
            radius = np.max(self.bounds[1] - self.bounds[0]) / 2

        sampler = qmc.Sobol(d=self.dim, scramble=True)
        points = sampler.random(n=n_points)
        points = qmc.scale(points, -1, 1)  # Scale to [-1, 1]

        # Project points to a hypersphere
        lengths = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / lengths * np.random.uniform(0, 1, size=lengths.shape) ** (1 / self.dim)

        points = points * radius + center

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
        # Implement Lower Confidence Bound acquisition function
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        return mu - self.kappa * sigma

    def _select_next_point(self, gp):
        # Select the next point to evaluate within the trust region
        n_samples = 100 * self.dim
        samples = self._sample_points(n_samples, center=self.best_x, radius=self.trust_region_radius)
        acq_values = self._acquisition_function(samples, gp)
        best_index = np.argmin(acq_values)
        return samples[best_index]

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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop

        # Initial exploration: Combine uniform sampling with sampling around the best seen point
        n_uniform = self.n_init // 2
        n_around_best = self.n_init - n_uniform

        initial_X_uniform = self._sample_points(n_uniform)
        initial_X_best = self._sample_points(n_around_best, center=self.best_x if self.best_x is not None else (self.bounds[1] + self.bounds[0]) / 2, radius=np.max(self.bounds[1] - self.bounds[0]) / 4)

        initial_X = np.vstack((initial_X_uniform, initial_X_best))
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Initialize best_x to a random initial point
        self.best_x = self.X[np.argmin(self.y)].copy()

        while self.n_evals < self.budget:
            # Optimization
            gp = self._fit_model(self.X, self.y)

            # Select next point
            next_x = self._select_next_point(gp)
            next_y = self._evaluate_points(func, next_x.reshape(1, -1))
            self._update_eval_points(next_x.reshape(1, -1), next_y)

            # Adjust trust region radius and kappa
            mu, sigma = gp.predict(next_x.reshape(1, -1), return_std=True)
            sigma = sigma[0]

            if next_y < self.best_y:
                self.trust_region_radius /= self.rho  # Expand
                self.kappa *= np.clip(self.rho * 0.9 * (1 - sigma), 0.1, 1.0) # Reduced kappa decrease, also consider GP's uncertainty
                self.success_history.append(True)
                self.best_x = next_x.copy() # global best, keep it.
            else:
                self.trust_region_radius *= self.rho  # Shrink
                self.kappa /= np.clip(self.rho * 0.9 * (1 - sigma), 0.1, 1.0) # increase kappa more when unsuccessful, also consider GP's uncertainty
                self.success_history.append(False)
                
            # Trust region center adaptation: adapt the trust region to the best point within the region
            samples_in_tr = self._sample_points(n_points=100, center=self.best_x, radius=self.trust_region_radius)
            mu_tr, _ = gp.predict(samples_in_tr, return_std=True)
            best_index_tr = np.argmin(mu_tr)
            self.best_x = samples_in_tr[best_index_tr].copy()

            # Stochastic radius perturbation
            self.trust_region_radius *= np.exp(np.random.normal(0, 0.05 * sigma)) # Perturbation scaled by GP uncertainty

            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)
            self.kappa = np.clip(self.kappa, 0.1, 10.0)
            
            # Adjust rho based on success history
            if len(self.success_history) > 10:
                success_rate = np.mean(self.success_history[-10:])
                self.rho = 0.9 + 0.09 * success_rate #adaptive rho. Higher success rate leads to higher rho, and thus slower shrinking.

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ATRBO_DKAICASP got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1695 with standard deviation 0.1130.

took 426.64 seconds to run.