# Description
**ATRBO_DKAISCA_EICR**: Adaptive Trust Region Bayesian Optimization with Dynamic Kappa, Adaptive Initial Exploration, Stochastic Radius, Center Adaptation with Success-rate-based Radius Modification, Expected Improvement, and Cumulative Regret based Kappa Refinement. This algorithm builds upon ATRBO_DKAISCA_EI by incorporating a Cumulative Regret based mechanism to dynamically refine the kappa parameter. This mechanism aims to balance exploration and exploitation more effectively by considering the overall performance of the optimization process. The kappa parameter is adjusted based on the cumulative regret, which is the difference between the best-observed function value and the function values obtained at each iteration. A higher cumulative regret indicates that the algorithm is not exploring effectively, and the kappa parameter is increased to promote more exploration. Conversely, a low cumulative regret suggests that the algorithm is exploiting well, and the kappa parameter is decreased to focus on exploitation.

# Justification
The key improvement lies in the introduction of cumulative regret to refine the kappa parameter.
1.  **Cumulative Regret-based Kappa Refinement:** The cumulative regret provides a more holistic view of the optimization process compared to just the immediate success or failure. By considering the cumulative difference between the best-observed value and the evaluated points, the algorithm can better assess whether it is adequately exploring the search space.
2.  **Adaptive Exploration-Exploitation Balance:** The dynamic adjustment of kappa based on cumulative regret allows the algorithm to adapt its exploration-exploitation balance throughout the optimization process. This is particularly useful in complex search spaces where the optimal balance may change over time.
3.  **Robustness:** The cumulative regret is less sensitive to noise and local fluctuations compared to immediate success or failure, making the algorithm more robust to noisy environments.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ATRBO_DKAISCA_EICR:
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
        self.kappa = 2.0  # Exploration-exploitation trade-off for EI
        self.success_history = [] # Keep track of successful moves
        self.success_rate_threshold = 0.7 # Threshold for aggressive expansion
        self.min_trust_region_radius = 0.1 # Minimum trust region radius
        self.cumulative_regret = 0.0 # Cumulative regret


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

        # Adjust exploration-exploitation based on GP uncertainty
        exploration_factor = 1.0 + np.mean(sigma)
        ei = ei * exploration_factor
        ei = ei * self.kappa

        return -ei  # We want to maximize EI, but minimize the acquisition function

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
        initial_X_best = self._sample_points(n_around_best, center=self.bounds[1]/2, radius=np.max(self.bounds[1] - self.bounds[0]) / 4) # Sampling around the middle of the search space as initial guess

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
            if next_y < self.best_y:
                self.trust_region_radius /= self.rho  # Expand
                #self.kappa *= self.rho * 0.9 # Reduced kappa decrease.
                self.success_history.append(True)
                self.best_x = next_x.copy() # global best, keep it.
            else:
                self.trust_region_radius *= self.rho  # Shrink
                # Stochastic expansion
                self.trust_region_radius *= (1 + np.random.normal(0, 0.05) * np.sqrt(self.dim))
                #self.kappa /= (self.rho*0.9) # increase kappa more when unsuccessful
                self.success_history.append(False)
            
            # Trust region center adaptation: adapt the trust region to the best point within the region
            samples_in_tr = self._sample_points(n_points=100, center=self.best_x, radius=self.trust_region_radius)
            mu_tr, _ = gp.predict(samples_in_tr, return_std=True)
            best_index_tr = np.argmin(mu_tr)
            self.best_x = samples_in_tr[best_index_tr].copy()

            self.trust_region_radius = np.clip(self.trust_region_radius, self.min_trust_region_radius, np.max(self.bounds[1] - self.bounds[0]) / 2)
            #self.kappa = np.clip(self.kappa, 0.1, 10.0)
            
            # Adjust rho based on success history
            if len(self.success_history) > 10:
                success_rate = np.mean(self.success_history[-10:])
                self.rho = 0.9 + 0.09 * success_rate #adaptive rho. Higher success rate leads to higher rho, and thus slower shrinking.

                # Aggressive expansion if success rate is high
                if success_rate > self.success_rate_threshold:
                    self.trust_region_radius /= (self.rho * (1 + (success_rate - self.success_rate_threshold)))
            
            # Update cumulative regret
            self.cumulative_regret += (next_y[0][0] - self.best_y)

            # Adjust kappa based on cumulative regret
            if self.cumulative_regret > 0:
                self.kappa *= 1.1  # Increase exploration
            else:
                self.kappa *= 0.9  # Increase exploitation

            self.kappa = np.clip(self.kappa, 0.1, 10.0)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ATRBO_DKAISCA_EICR got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1878 with standard deviation 0.1033.

took 233.27 seconds to run.