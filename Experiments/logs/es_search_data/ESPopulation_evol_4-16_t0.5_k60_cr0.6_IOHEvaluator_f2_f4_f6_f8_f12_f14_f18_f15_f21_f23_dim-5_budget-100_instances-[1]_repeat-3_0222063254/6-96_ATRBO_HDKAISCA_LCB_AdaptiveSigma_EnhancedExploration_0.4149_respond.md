# Description
**ATRBO-HDKAISCA-LCB-AdaptiveSigma-EnhancedExploration**: Adaptive Trust Region Bayesian Optimization combining Hybrid Dynamic Kappa, Radius Adjustment based on Exploration Success, Adaptive Initial Exploration, Stochastic Radius, Center Adaptation with Success-rate-based Radius Modification, Lower Confidence Bound Enhancement, Adaptive Sigma Scaling, and Enhanced Exploration with dynamic probability. This algorithm integrates successful features from ATRBO_DKAISCA_LCB_AdaptiveSigma and ATRBO_HDKRAES_LCB. It uses hybrid dynamic kappa adaptation, adjusts the radius based on exploration success, employs adaptive initial exploration, adapts the trust region center, scales sigma adaptively, and uses a lower confidence bound for acquisition. Additionally, it incorporates enhanced exploration by sampling from a wider region with a dynamically adjusted probability based on the success rate. The dynamic probability of enhanced exploration allows for more frequent global exploration when the success rate is low, and less frequent exploration when the success rate is high, focusing on exploitation within the trust region.

# Justification
This algorithm combines the strengths of ATRBO_DKAISCA_LCB_AdaptiveSigma and ATRBO_HDKRAES_LCB.

*   **Hybrid Dynamic Kappa:** Combines success/failure ratio and GP variance for a more robust kappa adaptation.
*   **Radius Adjustment based on Exploration Success:** Expands the trust region more aggressively when the search is consistently improving.
*   **Adaptive Initial Exploration:** Uses a combination of uniform sampling, sampling around the center, and Latin Hypercube sampling for a more diverse initial exploration.
*   **Stochastic Radius:** Introduces randomness in the trust region radius adjustment.
*   **Center Adaptation:** Adapts the trust region center to the best point within the region.
*   **Lower Confidence Bound:** Balances exploration and exploitation using the LCB acquisition function.
*   **Adaptive Sigma Scaling:** Dynamically adjusts the GP's sigma (uncertainty) in the acquisition function based on the success rate.
*   **Enhanced Exploration with Dynamic Probability:** Samples from a wider region with a probability that adapts based on the success rate. This allows the algorithm to dynamically adjust the balance between local exploitation and global exploration. If the success rate is low, the algorithm increases the probability of sampling from a wider region to escape local optima. If the success rate is high, the algorithm decreases the probability to focus on exploiting the current promising region.

The dynamic probability of enhanced exploration is a key addition. It allows the algorithm to automatically adjust its exploration-exploitation balance based on the search progress. This is particularly useful for complex and multimodal functions where the optimal balance may change during the optimization process.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ATRBO_HDKAISCA_LCB_AdaptiveSigma_EnhancedExploration:
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
        self.success_rate_threshold = 0.7 # Threshold for aggressive expansion
        self.kappa_min = 0.1
        self.kappa_max = 10.0
        self.success_rate_window = 10
        self.tr_center_adaptation_frequency = 5
        self.iteration = 0
        self.sigma_scaling = 1.0
        self.exploration_probability = 0.1 # Initial probability of sampling from a wider region
        self.exploration_probability_scaling = 0.5 # Scaling factor for adjusting exploration probability

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
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=min(10, self.dim), random_state=42)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp):
        # Implement Lower Confidence Bound acquisition function
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Adjust kappa based on success rate
        if len(self.success_history) > self.success_rate_window:
            success_rate = np.mean(self.success_history[-self.success_rate_window:])
        else:
            success_rate = 0.5 # Initial guess

        kappa_adjusted = self.kappa * (1.0 - 0.5 * success_rate) # Reduce kappa if success rate is high
        sigma_scaled = sigma * (1.0 + (0.5 - success_rate)) # Scale sigma based on success rate

        return mu - kappa_adjusted * sigma_scaled

    def _select_next_point(self, gp):
        # Select the next point to evaluate within the trust region
        n_samples = 100 * self.dim

        # Enhanced exploration: sample from a wider region with a dynamically adjusted probability
        if np.random.rand() < self.exploration_probability:
            radius = np.max(self.bounds[1] - self.bounds[0]) / 2
        else:
            radius = self.trust_region_radius

        samples = self._sample_points(n_samples, center=self.best_x, radius=radius)
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
        n_uniform = self.n_init // 3
        n_around_best = self.n_init // 3
        n_lhs = self.n_init - n_uniform - n_around_best

        initial_X_uniform = self._sample_points(n_uniform)
        initial_X_best = self._sample_points(n_around_best, center=self.bounds[1]/2, radius=np.max(self.bounds[1] - self.bounds[0]) / 4) # Sampling around the middle of the search space as initial guess

        sampler = qmc.LatinHypercube(d=self.dim, seed=42)
        initial_X_lhs = sampler.random(n=n_lhs)
        initial_X_lhs = qmc.scale(initial_X_lhs, self.bounds[0][0], self.bounds[1][0])

        initial_X = np.vstack((initial_X_uniform, initial_X_best, initial_X_lhs))
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

            if len(self.success_history) > 0:
                last_success = self.success_history[-1]
            else:
                last_success = False

            if next_y < self.best_y:
                self.trust_region_radius /= self.rho  # Expand
                self.kappa *= np.clip(self.rho * 0.9 * (1 - sigma), 0.1, 1.0) # Reduced kappa decrease, also consider GP's uncertainty
                self.success_history.append(True)
                self.best_x = next_x.copy() # global best, keep it.
            else:
                self.trust_region_radius *= self.rho  # Shrink
                # Stochastic expansion
                self.trust_region_radius *= (1 + np.random.normal(0, 0.05) * np.sqrt(self.dim))
                self.kappa /= np.clip(self.rho * 0.9 * (1 - sigma), 0.1, 1.0) # increase kappa more when unsuccessful
                self.success_history.append(False)

            # Trust region center adaptation: adapt the trust region to the best point within the region
            if self.iteration % self.tr_center_adaptation_frequency == 0:
                samples_in_tr = self._sample_points(n_points=100, center=self.best_x, radius=self.trust_region_radius)
                mu_tr, _ = gp.predict(samples_in_tr, return_std=True)
                best_index_tr = np.argmin(mu_tr)
                self.best_x = samples_in_tr[best_index_tr].copy()

            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)
            self.kappa = np.clip(self.kappa, self.kappa_min, self.kappa_max)

            # Adjust rho based on success history
            if len(self.success_history) > self.success_rate_window:
                success_rate = np.mean(self.success_history[-self.success_rate_window:])
                self.rho = 0.9 + 0.09 * success_rate #adaptive rho. Higher success rate leads to higher rho, and thus slower shrinking.

                # Aggressive expansion if success rate is high
                if success_rate > self.success_rate_threshold:
                    self.trust_region_radius /= (self.rho * (1 + (success_rate - self.success_rate_threshold)))

                # Adapt TR center adaptation frequency
                self.tr_center_adaptation_frequency = max(1, int(5 * (1 - success_rate)))

                # Adjust exploration probability based on success rate
                self.exploration_probability = np.clip(self.exploration_probability_scaling * (1 - success_rate), 0.01, 0.5) # Ensure a minimum exploration probability

            self.iteration += 1

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ATRBO_HDKAISCA_LCB_AdaptiveSigma_EnhancedExploration got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1839 with standard deviation 0.1012.

took 432.72 seconds to run.