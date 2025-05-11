# Description
**ATRBO_DKAIS_GAE:** Adaptive Trust Region Bayesian Optimization with Dynamic Kappa, Adaptive Initial Exploration, Stochastic Radius, and Gradient-Aware Exploration (ATRBO_DKAIS_GAE). This algorithm integrates the strengths of ATRBO_DKAIS (stochastic radius expansion for escaping local optima) and ATRBO_DKARE (gradient-aware kappa adaptation and refined rho adaptation). The core idea is to combine the stochastic trust region adjustments with gradient information to guide exploration, while also adaptively managing the trust region size and exploration-exploitation trade-off. It also incorporates a more robust initial exploration strategy.

# Justification
This algorithm aims to improve upon ATRBO_DKAIS and ATRBO_DKARE by combining their strengths:
*   **Stochastic Radius Expansion (from ATRBO_DKAIS):** Helps to escape local optima by occasionally expanding the trust region, even when no improvement is observed.
*   **Gradient-Aware Kappa Adaptation (from ATRBO_DKARE):** Uses gradient information to refine the exploration-exploitation trade-off. Higher predicted gradients near the current best indicate a potentially steeper slope to a better optimum, encouraging exploration in that region.
*   **Refined Rho Adaptation (from ATRBO_DKARE):** Adjusts `rho` based on both the recent success rate and the relative change in the objective function value. If the objective function is improving slowly, increase `rho` to shrink the trust region faster and focus on potentially better areas.
*   **Improved Initial Exploration:** The initial exploration phase is enhanced by combining uniform sampling with sampling around multiple promising locations.
*   **Adaptive Noise Handling:** Includes the noise handling from ATRBO_DKARE to prevent overfitting.

The combination of these features should lead to a more robust and efficient optimization algorithm. The stochastic radius expansion helps to prevent premature convergence, while the gradient-aware kappa adaptation and refined rho adaptation help to guide the search towards promising regions of the search space.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import approx_fprime


class ATRBO_DKAIS_GAE:
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
        self.global_sampling_prob = 0.05
        self.prev_best_y = float('inf')
        self.noise_level = 1e-6

    def _sample_points(self, n_points, center=None, radius=None, global_sample=False):
        if global_sample:
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            points = sampler.random(n=n_points)
            points = qmc.scale(points, self.bounds[0], self.bounds[1])
            return points
        else:
            if center is None:
                center = (self.bounds[1] + self.bounds[0]) / 2
            if radius is None:
                radius = np.max(self.bounds[1] - self.bounds[0]) / 2

            sampler = qmc.LatinHypercube(d=self.dim, seed=42)
            points = sampler.random(n=n_points)
            points = qmc.scale(points, -1, 1)

            lengths = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / lengths * self.rng.uniform(0, 1, size=lengths.shape) ** (1 / self.dim)

            points = points * radius + center
            points = np.clip(points, self.bounds[0], self.bounds[1])
            return points

    def _fit_model(self, X, y):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=self.noise_level, random_state=42)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp):
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        return mu - self.kappa * sigma

    def _select_next_point(self, gp, func):
        if self.rng.rand() < self.global_sampling_prob:
            # Global sampling
            n_samples = 100 * self.dim
            samples = self._sample_points(n_samples, global_sample=True)
            acq_values = self._acquisition_function(samples, gp)
            best_index = np.argmin(acq_values)
            return samples[best_index]
        else:
            # Trust region sampling
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
        # Initial exploration: Combine uniform sampling with sampling around the best seen point
        n_uniform = self.n_init // 3
        n_around_best1 = self.n_init // 3
        n_around_best2 = self.n_init - n_uniform - n_around_best1

        initial_X_uniform = self._sample_points(n_uniform)
        initial_X_best1 = self._sample_points(n_around_best1, center=self.bounds[1]/4, radius=np.max(self.bounds[1] - self.bounds[0]) / 8) # Sampling around the middle of the search space as initial guess
        initial_X_best2 = self._sample_points(n_around_best2, center=-self.bounds[1]/4, radius=np.max(self.bounds[1] - self.bounds[0]) / 8) # Sampling around the middle of the search space as initial guess


        initial_X = np.vstack((initial_X_uniform, initial_X_best1, initial_X_best2))
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Initialize best_x to a random initial point
        self.best_x = self.X[np.argmin(self.y)].copy()
        self.prev_best_y = self.best_y


        while self.n_evals < self.budget:
            # Optimization
            gp = self._fit_model(self.X, self.y)
            next_x = self._select_next_point(gp, func)
            next_y = self._evaluate_points(func, next_x.reshape(1, -1))
            self._update_eval_points(next_x.reshape(1, -1), next_y)

            # Update success history
            success = next_y < self.best_y
            self.success_history.append(success[0])
            if len(self.success_history) > self.success_window:
                self.success_history = self.success_history[-self.success_window:]

            # Adapt kappa based on GP variance and gradient
            mu, sigma = gp.predict(self._sample_points(100, center=self.best_x, radius=self.trust_region_radius), return_std=True)
            avg_sigma = np.mean(sigma)
            #approximate gradient using finite differences
            gradient = approx_fprime(self.best_x, lambda x: gp.predict(x.reshape(1,-1))[0], epsilon=1e-6)
            gradient_norm = np.linalg.norm(gradient)

            self.kappa = np.clip(self.kappa * (1 + avg_sigma + gradient_norm), 0.1, 10.0)

            # Adapt rho based on success rate and relative improvement
            success_rate = np.mean(self.success_history) if self.success_history else 0.5
            relative_improvement = abs(self.best_y - self.prev_best_y) / (abs(self.prev_best_y) + 1e-9)
            self.rho = np.clip(0.9 + (0.5 - success_rate) / 5 + (0.1 - relative_improvement), 0.7, 0.99)

            # Adjust trust region radius with stochasticity
            if success:
                self.trust_region_radius /= (self.rho + self.rng.normal(0, 0.01))
                self.best_x = next_x.copy()
            else:
                self.trust_region_radius *= (self.rho + self.rng.normal(0, 0.05)) #Stochastic expansion

            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)

            # Estimate noise level (simple estimate based on recent evaluations)
            if len(self.y) > 5:
                self.noise_level = np.std(self.y[-5:])

            self.prev_best_y = self.best_y

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ATRBO_DKAIS_GAE got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1802 with standard deviation 0.1012.

took 247.54 seconds to run.