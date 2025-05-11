# Description
**AdaTrustRegionBO**: Adaptive Trust Region Bayesian Optimization with Dynamic Kappa, Adaptive Rho, and Center Adaptation. This algorithm combines the strengths of ATRBO_DKAI and ATRBO_DKAR. It adaptively adjusts the exploration-exploitation trade-off (kappa), the trust region radius shrinking factor (rho), and the trust region center based on optimization progress. Stochasticity is also introduced to avoid premature convergence of trust region radius. Furthermore, a minimum number of evaluations are introduced before adjusting the trust region radius and kappa.

# Justification
This algorithm builds upon the success of ATRBO_DKAI and ATRBO_DKAR by integrating their best features and addressing their shortcomings. ATRBO_DKAI has good initial exploration and adaptive kappa but lacks trust region center adaptation and the stochasticity. ATRBO_DKAR includes dynamic Kappa based on GP variance, adaptive rho and trust region center adaptation. By combining these strengths, AdaTrustRegionBO provides robust and efficient optimization by:
1.  **Dynamic Kappa:** Adaptively adjusts kappa based on the GP's predicted variance within the trust region. High variance leads to increased exploration, while low variance promotes exploitation.
2.  **Adaptive Rho:** Dynamically adjusts the shrinking factor rho based on a "success rate" within the trust region. If recent samples have led to improvements, rho decreases (slower shrinking). If not, rho increases to quickly shrink the trust region and explore elsewhere.
3.  **Trust Region Center Adaptation:** The trust region center is adapted after each iteration if the new evaluation point improves the best objective, focusing the search in promising regions.
4.  **Stochastic Trust Region Radius:** Introduce stochasticity when updating the trust region radius to avoid premature convergence.
5.  **Initial Exploration:** Combine uniform sampling with sampling around the middle of search space for efficient initial exploration.
6. **Minimum number of evaluation before adjusting trust region radius and kappa:** Avoid adjusting trust region radius and kappa before certain number of function evaluations to prevent premature convergence, this also improves the speed of the code.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class AdaTrustRegionBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(20 * dim, self.budget // 5)
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.5
        self.rho = 0.95
        self.kappa = 2.0
        self.success_history = []
        self.success_window = 5
        self.rng = np.random.RandomState(42)
        self.min_evals_before_adjust = 5  # Minimum evaluations before TR adjustment

    def _sample_points(self, n_points, center=None, radius=None):
        if center is None:
            center = (self.bounds[1] + self.bounds[0]) / 2
        if radius is None:
            radius = np.max(self.bounds[1] - self.bounds[0]) / 2

        sampler = qmc.Sobol(d=self.dim, scramble=True)
        points = sampler.random(n=n_points)
        points = qmc.scale(points, -1, 1)

        lengths = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / lengths * self.rng.uniform(0, 1, size=lengths.shape) ** (1 / self.dim)

        points = points * radius + center
        points = np.clip(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp):
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        return mu - self.kappa * sigma

    def _select_next_point(self, gp):
        n_samples = 100 * self.dim
        samples = self._sample_points(n_samples, center=self.best_x, radius=self.trust_region_radius)
        acq_values = self._acquisition_function(samples, gp)
        best_index = np.argmin(acq_values)
        return samples[best_index]

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

        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx].copy()

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:

        # Initial exploration: Combine uniform sampling with sampling around the best seen point
        n_uniform = self.n_init // 2
        n_around_best = self.n_init - n_uniform

        initial_X_uniform = self._sample_points(n_uniform)
        initial_X_best = self._sample_points(n_around_best, center=self.bounds[1]/2, radius=np.max(self.bounds[1] - self.bounds[0]) / 4) # Sampling around the middle of the search space as initial guess

        initial_X = np.vstack((initial_X_uniform, initial_X_best))

        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Initialize best_x to a random initial point
        self.best_x = self.X[np.argmin(self.y)].copy()

        while self.n_evals < self.budget:
            # Optimization
            gp = self._fit_model(self.X, self.y)
            next_x = self._select_next_point(gp)
            next_y = self._evaluate_points(func, next_x.reshape(1, -1))
            self._update_eval_points(next_x.reshape(1, -1), next_y)

            # Update success history
            success = next_y < self.best_y
            self.success_history.append(success[0])
            if len(self.success_history) > self.success_window:
                self.success_history = self.success_history[-self.success_window:]

            # Adapt kappa based on GP variance within trust region
            mu, sigma = gp.predict(self._sample_points(100, center=self.best_x, radius=self.trust_region_radius), return_std=True)
            avg_sigma = np.mean(sigma)
            self.kappa = np.clip(self.kappa * (1 + avg_sigma), 0.1, 10.0)

            # Adapt rho based on success rate
            success_rate = np.mean(self.success_history) if self.success_history else 0.5
            self.rho = np.clip(0.9 + (0.5 - success_rate) / 5, 0.7, 0.99)

            # Adjust trust region radius with stochasticity
            if self.n_evals > self.min_evals_before_adjust:
                if success:
                    self.trust_region_radius /= (self.rho + self.rng.normal(0, 0.01))
                    self.best_x = next_x.copy()
                else:
                    self.trust_region_radius *= (self.rho + self.rng.normal(0, 0.01))

            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)

        return self.best_y, self.best_x
```