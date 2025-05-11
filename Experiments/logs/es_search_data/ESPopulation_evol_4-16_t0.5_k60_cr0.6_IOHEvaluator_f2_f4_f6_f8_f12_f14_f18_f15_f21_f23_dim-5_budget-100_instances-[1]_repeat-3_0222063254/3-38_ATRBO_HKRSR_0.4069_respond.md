# Description
**Adaptive Trust Region Bayesian Optimization with Hybrid Kappa-Rho and Stochastic Radius (ATRBO_HKRSR):** This algorithm synergistically combines the dynamic adaptation of kappa from ATRBO_HKRA, the success-history based rho adjustment from ATRBO_DKAIS, and the stochastic trust region radius expansion from ATRBO_DKAIS. Further, it incorporates an adaptive initial exploration strategy using a combination of Latin Hypercube sampling and Sobol sequences. The initial kappa is also made adaptive to the problem dimensionality. This combination aims to provide a robust and efficient exploration-exploitation trade-off while preventing premature convergence.

# Justification
This design combines the strengths of both ATRBO_HKRA and ATRBO_DKAIS, while attempting to address their weaknesses.
*   **Hybrid Kappa Adjustment (from ATRBO_HKRA):** Offers a balance between exploitation based on the GP model's variance and exploration based on the success/failure history. This is crucial for adapting to different problem landscapes.
*   **Success-History based Rho Adaptation (from ATRBO_DKAIS):** Allows for a more informed adjustment of the trust region radius shrinking factor, preventing over-shrinking in areas of consistent improvement, which can lead to premature convergence.
*   **Stochastic Radius Expansion (from ATRBO_DKAIS):** Introduces a controlled degree of randomness in the trust region radius adjustment, allowing the algorithm to escape local optima.
*   **Adaptive Initial Exploration:** Combines Latin Hypercube sampling (good space-filling properties) and Sobol sequences (good low-discrepancy properties) to ensure a diverse and well-distributed initial sample set. This helps in quickly identifying promising regions in the search space.
*   **Adaptive Initial Kappa:** Setting the initial kappa based on the dimensionality of the problem helps to scale the exploration-exploitation balance appropriately.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ATRBO_HKRSR:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0  # the number of function evaluations
        self.n_init = min(20 * dim, self.budget // 5)
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.5  # Initial trust region radius
        self.kappa = 2.0 + np.log(dim)  # Exploration-exploitation trade-off for LCB, adaptive to dimension
        self.success_count = 0
        self.failure_count = 0
        self.min_evals_for_adjust = 5 * dim
        self.rho = 0.95  # Shrinking factor for trust region
        self.success_history = []

        self.kappa_success_failure_weight = 0.5  # Weight for combining success/failure and variance based kappa adjustment

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
            self.best_x = self.X[idx].copy()

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop

        # Initial exploration
        n_uniform = self.n_init // 2
        n_sobol = self.n_init - n_uniform
        sampler_lhs = qmc.LatinHypercube(d=self.dim, seed=42)
        initial_X_uniform = sampler_lhs.random(n=n_uniform)
        initial_X_uniform = qmc.scale(initial_X_uniform, self.bounds[0][0], self.bounds[1][0])

        sampler_sobol = qmc.Sobol(d=self.dim, scramble=True)
        initial_X_sobol = sampler_sobol.random(n=n_sobol)
        initial_X_sobol = qmc.scale(initial_X_sobol, self.bounds[0][0], self.bounds[1][0])

        initial_X = np.vstack((initial_X_uniform, initial_X_sobol))
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
            if self.n_evals > self.n_init + self.min_evals_for_adjust:
                # Radius Adjustment (DKRA component)
                if next_y < self.best_y:
                    self.success_count += 1
                    self.failure_count = 0
                    self.success_history.append(True)
                else:
                    self.failure_count += 1
                    self.success_count = 0
                    self.success_history.append(False)

                # Hybrid Kappa Adjustment (DKRA + GP variance)
                mu, sigma = gp.predict(self.X, return_std=True)
                avg_sigma = np.mean(sigma)
                kappa_gp_component = 1.0 + np.log(1 + avg_sigma)  # Example GP variance component, can be tuned

                success_ratio = self.success_count / (self.success_count + self.failure_count + 1e-9)
                kappa_success_failure_component = (0.9 + 0.09 * success_ratio)

                self.kappa = (self.kappa_success_failure_weight * kappa_success_failure_component +
                               (1 - self.kappa_success_failure_weight) * kappa_gp_component)

                self.kappa = np.clip(self.kappa, 0.1, 10.0)

                # Radius and Rho Adjustment (Based on both success/failure and history)
                if len(self.success_history) > 10:
                    success_rate = np.mean(self.success_history[-10:])
                    self.rho = 0.9 + 0.09 * success_rate  # adaptive rho. Higher success rate leads to higher rho, and thus slower shrinking.

                if next_y < self.best_y:
                    self.trust_region_radius /= self.rho  # Expand
                else:
                    self.trust_region_radius *= self.rho  # Shrink
                    self.trust_region_radius *= (1 + np.random.normal(0, 0.05))  # Stochastic expansion

                self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)

            # Trust Region Center Update
            if next_y < self.best_y:
                self.best_x = next_x.copy()  # Adapt trust region center

        return self.best_y, self.best_x
```