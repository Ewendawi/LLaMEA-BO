# Description
**ATRBO-HDKEICA**: Hybrid Dynamic Kappa, Enhanced Initial Exploration, Center Adaptation, EI-based Acquisition with Confidence Bound Regularization, and Radius Adjustment based on Exploration Success. This algorithm combines the strengths of ATRBO_DKEICA and ATRBO_HDKRAES_LCB, incorporating hybrid dynamic kappa adaptation (success/failure ratio and GP variance), enhanced initial exploration with Latin Hypercube sampling and center-focused sampling, trust region center adaptation, stochastic radius adjustment, and an Expected Improvement (EI) acquisition function regularized by a Lower Confidence Bound (LCB) to balance exploration and exploitation. It further enhances the initial exploration by using Latin Hypercube sampling and focuses on the center of the search space. It also includes radius adjustment based on exploration success and success-rate-based radius scaling.

# Justification
This algorithm aims to improve upon ATRBO_DKEICA and ATRBO_HDKRAES_LCB by combining their best features.
- **Hybrid Dynamic Kappa:** Combines the success/failure ratio and GP variance for a more robust kappa adaptation.
- **EI with LCB Regularization:** Uses EI as the primary acquisition function but regularizes it with LCB to balance exploration and exploitation.
- **Enhanced Exploration:** Uses a combination of Latin Hypercube sampling, uniform sampling, and center-focused sampling for initial exploration.
- **Adaptive Radius Adjustment:** Adjusts the trust region radius based on exploration success and success rate.
- **Trust Region Center Adaptation:** Adapts the trust region center based on the best point within the region.
- **Stochastic Radius Adjustment**: Adds randomness to the radius adjustment to avoid premature convergence.

The combination of these features aims to create a robust and efficient Bayesian optimization algorithm that can handle a wide range of black box optimization problems.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ATRBO_HDKEICA:
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
        self.kappa = 2.0  # Exploration-exploitation trade-off for EI/LCB
        self.success_history = [] # Keep track of successful moves
        self.success_rate_threshold = 0.7 # Threshold for aggressive expansion
        self.kappa_min = 0.1
        self.kappa_max = 10.0
        self.success_rate_window = 10
        self.tr_center_adaptation_frequency = 5
        self.iteration = 0
        self.ei_weight = 0.7 # Weight for EI in the combined acquisition function
        self.exploration_probability = 0.1 # Probability of sampling from a wider region
        self.min_trust_region_radius = 0.1

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
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp):
        # Implement Expected Improvement acquisition function
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Ensure sigma is non-zero to avoid division by zero
        sigma = np.maximum(sigma, 1e-6)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Implement Lower Confidence Bound acquisition function
        lcb = mu - self.kappa * sigma

        # Adjust exploration-exploitation based on GP uncertainty
        exploration_factor = 1.0 + np.mean(sigma)
        ei = ei * exploration_factor

        # Combine EI and LCB
        # Dynamically adjust the weight of LCB based on GP uncertainty
        lcb_weight = 1.0 - self.ei_weight #was based on sigma
        acq_func = self.ei_weight * (-ei) + lcb_weight * lcb # We want to maximize EI, but minimize the acquisition function

        return acq_func

    def _select_next_point(self, gp):
        # Select the next point to evaluate within the trust region
        n_samples = 100 * self.dim

        # Enhanced exploration: sample from a wider region with a certain probability
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
        initial_X_best = self._sample_points(n_around_best, center=(self.bounds[1] + self.bounds[0]) / 2, radius=np.max(self.bounds[1] - self.bounds[0]) / 4) # Sampling around the middle of the search space as initial guess
        
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

            self.trust_region_radius = np.clip(self.trust_region_radius, self.min_trust_region_radius, np.max(self.bounds[1] - self.bounds[0]) / 2)
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

            self.iteration += 1

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ATRBO_HDKEICA got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1945 with standard deviation 0.1092.

took 907.68 seconds to run.