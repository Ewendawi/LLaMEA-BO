# Description
Adaptive Trust Region Bayesian Optimization with Dynamic Kappa and Adaptive Rho (ATRBO-DKAR) enhances the original ATRBO by introducing a more responsive and adaptable trust region strategy. Key improvements include:
1.  Dynamic Kappa: Instead of fixed bounds, `kappa` (exploration-exploitation trade-off) now adapts based on the GP's uncertainty within the trust region. If the GP's predicted variance is high, `kappa` increases to encourage exploration. Conversely, low variance leads to exploitation.
2.  Adaptive Rho: The shrinking factor `rho` is also made dynamic. Its adjustment is based on a "success rate" within the trust region. If recent samples have led to improvements, `rho` decreases (slower shrinking) to allow for more focused exploitation. If not, `rho` increases to quickly shrink the trust region and explore elsewhere.
3.  Trust Region Center Adaptation: The trust region center is now adapted after each iteration. If the new evaluation point improves the best objective, the trust region center is set as the evaluation point. This is beneficial when the next evaluation point is far from the current best location and has better performance.
4.  Stochastic Trust Region Radius: Add a stochasticity when updating the trust region radius to avoid premature convergence.

# Justification
The primary goal is to improve the balance between exploration and exploitation within the trust region framework. The original ATRBO's fixed shrinking factor and kappa can lead to premature convergence or inefficient exploration. The dynamic adjustment of these parameters, informed by the GP's uncertainty and the success rate of recent samples, allows the algorithm to react more intelligently to the landscape of the objective function. The adaptively updated trust region center guides the search more effectively toward promising regions, while the stochasticity prevents getting stuck in local optima.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ATRBO_DKAR:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(10 * dim, self.budget // 5)
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.5
        self.rho = 0.95
        self.kappa = 2.0
        self.success_history = []  # Track recent success
        self.success_window = 5  # Window size for success rate calculation
        self.rng = np.random.RandomState(42)  # Consistent random state

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
            self.best_x = self.X[idx]

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
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
            if success:
                self.trust_region_radius /= (self.rho + self.rng.normal(0, 0.01))
                self.best_x = next_x.copy()
            else:
                self.trust_region_radius *= (self.rho + self.rng.normal(0, 0.01))

            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)

        return self.best_y, self.best_x
```