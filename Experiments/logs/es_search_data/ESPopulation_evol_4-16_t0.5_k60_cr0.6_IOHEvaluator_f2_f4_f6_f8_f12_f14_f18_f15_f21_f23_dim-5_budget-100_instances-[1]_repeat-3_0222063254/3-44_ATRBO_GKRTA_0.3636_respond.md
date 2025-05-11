# Description
**Adaptive Trust Region Bayesian Optimization with Gradient-Enhanced Kappa, Refined Rho, and Trust Region Adaptation (ATRBO_GKRTA):** This algorithm combines the strengths of ATRBO_DKAICA and ATRBO_DKARE, focusing on enhanced exploration/exploitation balance and robust trust region management. It incorporates gradient information into the kappa adaptation, refines the rho adaptation based on both success rate and objective function improvement, and adapts the trust region center to the best point within the trust region. Furthermore, it introduces a dynamic mechanism for adjusting the global sampling probability based on the success rate within the trust region. This aims to escape local optima more effectively while maintaining focused exploration.

# Justification
The ATRBO_GKRTA algorithm is designed to address the limitations of its predecessors by incorporating the following key features:

1.  **Gradient-Enhanced Kappa Adaptation:** Adapting `kappa` based on both GP variance and gradient information provides a more informed exploration-exploitation trade-off. High GP variance encourages exploration in uncertain regions, while high gradient norms suggest promising areas to exploit.
2.  **Refined Rho Adaptation:** Adjusting `rho` based on both success rate and relative improvement in the objective function allows for a more responsive trust region radius adaptation. This helps to balance exploration and exploitation by shrinking the trust region faster when progress is slow and expanding it when progress is being made.
3.  **Trust Region Center Adaptation:** Moving the trust region center to the best point found within the trust region enables the algorithm to focus on promising local areas more effectively.
4. **Dynamic Global Sampling Probability:** By dynamically adjusting the probability of global sampling based on the success rate within the trust region, the algorithm can escape local optima more effectively. If the success rate within the trust region is low, the algorithm increases the probability of global sampling to explore new regions of the search space.
5. **Noise handling:** The algorithm estimates the noise level and adjusts the GP's hyperparameters accordingly.

These features are combined to create a robust and efficient Bayesian optimization algorithm that can handle a wide range of black box optimization problems. The algorithm aims to balance exploration and exploitation by dynamically adjusting the trust region radius, exploration-exploitation trade-off, and global sampling probability based on the performance of the algorithm.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import approx_fprime


class ATRBO_GKRTA:
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
        self.success_history = []  # Track recent success
        self.success_window = 5  # Window size for success rate calculation
        self.rng = np.random.RandomState(42)  # Consistent random state
        self.global_sampling_prob = 0.05  # Probability of sampling globally
        self.prev_best_y = float('inf') # Store the previous best y to calculate the relative change
        self.noise_level = 1e-6
        self.tr_center_adaptation_frequency = 5 # Adapt trust region center every n iterations

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
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        self.best_x = self.X[np.argmin(self.y)].copy()
        self.prev_best_y = self.best_y

        for i in range(self.budget // (100 * self.dim)):
            if self.n_evals >= self.budget:
                break

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
                self.trust_region_radius *= (self.rho + self.rng.normal(0, 0.01))

            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)

            # Estimate noise level (simple estimate based on recent evaluations)
            if len(self.y) > 5:
                self.noise_level = np.std(self.y[-5:])

            self.prev_best_y = self.best_y

            # Trust region center adaptation
            if (i + 1) % self.tr_center_adaptation_frequency == 0:
                samples_in_tr = self._sample_points(n_points=100, center=self.best_x, radius=self.trust_region_radius)
                mu_tr, _ = gp.predict(samples_in_tr, return_std=True)
                best_index_tr = np.argmin(mu_tr)
                self.best_x = samples_in_tr[best_index_tr].copy()

            # Dynamic global sampling probability adjustment
            if success_rate < 0.2:
                self.global_sampling_prob = np.clip(self.global_sampling_prob * 1.1, 0.05, 0.5)  # Increase global sampling
            else:
                self.global_sampling_prob = np.clip(self.global_sampling_prob * 0.9, 0.01, 0.5)  # Decrease global sampling

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ATRBO_GKRTA got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1396 with standard deviation 0.0986.

took 0.13 seconds to run.