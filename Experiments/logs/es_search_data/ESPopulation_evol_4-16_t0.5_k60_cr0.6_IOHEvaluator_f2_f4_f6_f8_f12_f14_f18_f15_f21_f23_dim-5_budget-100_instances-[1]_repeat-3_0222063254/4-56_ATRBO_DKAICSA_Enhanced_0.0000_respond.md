# Description
**ATRBO_DKAICSA_Enhanced**: Adaptive Trust Region Bayesian Optimization with Dynamic Kappa, Adaptive Initial Exploration, Trust Region Center Adaptation, Stochastic Radius Adjustment, and Enhanced Radius Control with Adaptive Noise Handling. This algorithm combines the strengths of ATRBO_DKAICSA and ATRBO_DKAIS_TRCA, focusing on robust radius control and noise handling. It incorporates trust region center adaptation, dynamic kappa based on GP uncertainty, adaptive initial exploration, stochastic radius adjustment, and success-history based radius adaptation. Furthermore, it introduces a noise-aware mechanism to prevent premature convergence in noisy environments by adaptively adjusting the trust region radius based on the estimated noise level.

# Justification
The new algorithm, ATRBO_DKAICSA_Enhanced, builds upon the strengths of ATRBO_DKAICSA and ATRBO_DKAIS_TRCA, focusing on improving the trust region radius adaptation and handling potential noise in the function evaluations.

*   **Combination of Strengths:** It integrates the trust region center adaptation, dynamic kappa, adaptive initial exploration, and stochastic radius adjustment from ATRBO_DKAICSA. It also incorporates the success-history based radius adaptation from ATRBO_DKAIS_TRCA.
*   **Enhanced Radius Control:** The algorithm refines the trust region radius adjustment mechanism. It incorporates a success-history based adaptation of the shrinking factor `rho`, allowing for more aggressive expansion when the search is consistently improving.
*   **Noise-Aware Adaptation:** A key addition is a mechanism to estimate the noise level in the function evaluations. The trust region radius is then adaptively adjusted based on this noise level. This prevents premature convergence in noisy environments by maintaining a larger trust region radius when noise is high.  The noise level is estimated by looking at the variance of function evaluations within a small neighborhood of the current best point.
*   **Latin Hypercube Sampling:** Uses Latin Hypercube Sampling for the initial exploration to improve the coverage of the search space.
*   **Safety Mechanism:** Includes a safety mechanism to prevent the trust region radius from shrinking too fast, which helps maintain exploration capabilities, especially in later stages of the optimization.
*   **Computational Efficiency:** The core components are designed to be computationally efficient, leveraging numpy and scikit-learn for fast calculations. The GP fitting is limited to a small number of restarts to reduce the computational overhead.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import NearestNeighbors

class ATRBO_DKAICSA_Enhanced:
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
        self.tr_center_adaptation_frequency = 5 # Adapt TR center every 5 iterations
        self.iteration = 0
        self.noise_level = 0.0  # Estimated noise level

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, center=None, radius=None):
        # sample points within the trust region
        if center is None:
            center = (self.bounds[1] + self.bounds[0]) / 2
        if radius is None:
            radius = np.max(self.bounds[1] - self.bounds[0]) / 2

        sampler = qmc.LatinHypercube(d=self.dim, seed=42)
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
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42, alpha=self.noise_level**2)
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

    def _estimate_noise_level(self):
        # Estimate the noise level based on the variance of function evaluations near the best point
        if self.X is None or len(self.X) < 5:
            return 0.0  # Not enough data to estimate noise

        knn = NearestNeighbors(n_neighbors=min(5, len(self.X)), algorithm='ball_tree')
        knn.fit(self.X)
        distances, indices = knn.kneighbors(self.best_x.reshape(1, -1))
        neighbor_values = self.y[indices].flatten()
        self.noise_level = np.std(neighbor_values)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop

        # Initial exploration: Combine uniform sampling with sampling around the best seen point
        n_uniform = self.n_init // 2
        n_around_best = self.n_init - n_uniform

        initial_X_uniform = self._sample_points(n_uniform)
        initial_X_best = self._sample_points(n_around_best, center=(self.bounds[1] + self.bounds[0]) / 2, radius=np.max(self.bounds[1] - self.bounds[0]) / 4)

        initial_X = np.vstack((initial_X_uniform, initial_X_best))
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Initialize best_x to the best initial point
        self.best_x = self.X[np.argmin(self.y)].copy()

        while self.n_evals < self.budget:
            # Optimization
            self._estimate_noise_level()
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
                self.trust_region_radius *= (1 + np.random.normal(0, 0.05)) #Stochastic expansion
                self.kappa /= np.clip(self.rho * 0.9 * (1 - sigma), 0.1, 1.0) # increase kappa more when unsuccessful, also consider GP's uncertainty
                self.success_history.append(False)

            # Trust region center adaptation: adapt the trust region to the best point within the region
            if self.iteration % self.tr_center_adaptation_frequency == 0:
                samples_in_tr = self._sample_points(n_points=100, center=self.best_x, radius=self.trust_region_radius)
                mu_tr, _ = gp.predict(samples_in_tr, return_std=True)
                best_index_tr = np.argmin(mu_tr)
                self.best_x = samples_in_tr[best_index_tr].copy()

            #Adaptive lower bound of trust region:
            distances = np.linalg.norm(self.X - self.best_x, axis=1)
            avg_distance = np.mean(distances) if len(self.X) > 1 else np.max(self.bounds[1] - self.bounds[0]) / 10
            self.trust_region_radius = np.clip(self.trust_region_radius, min(1e-2, avg_distance), np.max(self.bounds[1] - self.bounds[0]) / 2)
            self.kappa = np.clip(self.kappa, 0.1, 10.0)

            # Adjust rho based on success history
            if len(self.success_history) > 10:
                success_rate = np.mean(self.success_history[-10:])
                self.rho = 0.9 + 0.09 * success_rate #adaptive rho. Higher success rate leads to higher rho, and thus slower shrinking.

            # Noise aware radius adjustment
            self.trust_region_radius = max(self.trust_region_radius, 0.1 * self.noise_level)
            self.iteration += 1

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<ATRBO_DKAICSA_Enhanced>", line 127, in __call__
 127->             gp = self._fit_model(self.X, self.y)
  File "<ATRBO_DKAICSA_Enhanced>", line 57, in _fit_model
  55 |         kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
  56 |         gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42, alpha=self.noise_level**2)
  57->         gp.fit(X, y)
  58 |         return gp
  59 | 
numpy.linalg.LinAlgError: ("The kernel, 1**2 * RBF(length_scale=1), is not returning a positive definite matrix. Try gradually increasing the 'alpha' parameter of your GaussianProcessRegressor estimator.", '35-th leading minor of the array is not positive definite')
