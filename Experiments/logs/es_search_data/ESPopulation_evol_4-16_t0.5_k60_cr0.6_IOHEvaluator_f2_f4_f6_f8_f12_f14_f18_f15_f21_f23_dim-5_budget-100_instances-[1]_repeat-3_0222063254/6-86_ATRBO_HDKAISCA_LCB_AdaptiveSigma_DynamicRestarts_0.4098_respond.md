# Description
**ATRBO-HDKAISCA-LCB-AdaptiveSigma-DynamicRestarts**: Adaptive Trust Region Bayesian Optimization with Hybrid Dynamic Kappa, Adaptive Initial Exploration, Stochastic Radius, Center Adaptation with Success-rate-based Radius Modification, Lower Confidence Bound Enhancement, Adaptive Sigma Scaling, and Dynamic GP Optimizer Restarts. This algorithm combines the strengths of ATRBO_DKAISCA_LCB_AdaptiveSigma and ATRBO_HDKRAES_LCB, incorporating hybrid dynamic kappa adaptation (success/failure ratio and GP variance), dynamic radius adjustment based on exploration success, enhanced exploration by sampling from a wider region with a certain probability, success-rate-based radius scaling for more aggressive expansion when the search is consistently improving, Lower Confidence Bound for acquisition, adaptive scaling of the GP's sigma (uncertainty) in the acquisition function, and dynamic adjustment of GP optimizer restarts based on problem dimensionality and success rate.

# Justification
This algorithm aims to improve upon the previous ATRBO variants by integrating several successful strategies.

1.  **Hybrid Dynamic Kappa:** Combines success/failure ratio and GP variance for a more robust kappa adaptation.
2.  **Adaptive Sigma Scaling:** Dynamically adjusts the GP's sigma based on the success rate, allowing for a more nuanced exploration-exploitation balance.
3.  **Enhanced Exploration:** Samples from a wider region with a certain probability to escape local optima.
4.  **Dynamic GP Optimizer Restarts:** Adjusts the number of restarts for the GP optimizer based on the problem dimensionality and success rate within the trust region, aiming to improve GP fitting, especially for higher-dimensional problems where the GP fitting can be challenging. The number of restarts is increased if the success rate is low, suggesting that the GP might be trapped in a local optimum.
5.  **Success-rate-based Radius Scaling:** Allows for aggressive expansion of the trust region when the search is consistently improving.
6.  **Trust Region Center Adaptation:** Adapts the trust region center to the best point within the region.
7.  **Adaptive Rho:** Adjusts the shrinking factor based on success history. Higher success rate leads to slower shrinking.

The combination of these strategies should provide a more robust and efficient optimization algorithm. The dynamic GP optimizer restarts are particularly important for higher-dimensional problems, where the GP fitting can be a bottleneck.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ATRBO_HDKAISCA_LCB_AdaptiveSigma_DynamicRestarts:
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
        self.exploration_probability = 0.1 # Probability of sampling from a wider region
        self.sigma_scaling = 1.0
        self.gp_n_restarts_optimizer = min(10, self.dim)

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
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=self.gp_n_restarts_optimizer, random_state=42)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp):
        # Implement Lower Confidence Bound acquisition function
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Adjust kappa based on success rate and GP variance (Hybrid Dynamic Kappa)
        if len(self.success_history) > self.success_rate_window:
            success_rate = np.mean(self.success_history[-self.success_rate_window:])
        else:
            success_rate = 0.5 # Initial guess

        kappa_adjusted = self.kappa * (1.0 - 0.5 * success_rate) # Reduce kappa if success rate is high
        kappa_adjusted += 0.1 * np.var(mu) # Increase kappa if GP variance is high

        sigma_scaled = sigma * (1.0 + (0.5 - success_rate)) # Scale sigma based on success rate

        return mu - kappa_adjusted * sigma_scaled

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

                # Adapt GP optimizer restarts
                if success_rate < 0.3:
                    self.gp_n_restarts_optimizer = min(20, 2 * self.dim)
                else:
                    self.gp_n_restarts_optimizer = min(10, self.dim)

            self.iteration += 1

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ATRBO_HDKAISCA_LCB_AdaptiveSigma_DynamicRestarts got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1835 with standard deviation 0.1091.

took 1147.28 seconds to run.