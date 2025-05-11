# Description
**ATRBO_DKAISCA_EI**: Adaptive Trust Region Bayesian Optimization with Dynamic Kappa, Adaptive Initial Exploration, Stochastic Radius, Center Adaptation with Success-rate-based Radius Modification, and Exploration-Exploitation Balancing using Expected Improvement. This algorithm builds upon ATRBO_DKAISCA by incorporating the Expected Improvement (EI) acquisition function alongside the Lower Confidence Bound (LCB). The algorithm adaptively switches between EI and LCB based on the success rate, promoting exploration when the success rate is low and exploitation when it's high. This hybrid approach aims to balance exploration and exploitation more effectively. Furthermore, a small probability of global search is introduced to avoid getting stuck in local optima.

# Justification
The key idea is to leverage the strengths of both LCB and EI acquisition functions. LCB is good at exploitation and avoiding regions with high uncertainty, while EI excels at exploration by focusing on areas where improvement is likely. By adaptively switching between them based on the success rate, the algorithm can dynamically adjust its exploration-exploitation trade-off. The addition of a small probability of global search helps to escape local optima.

*   **Adaptive Acquisition Function:** Using a combination of LCB and EI, weighted by the success rate, allows the algorithm to adaptively balance exploration and exploitation. When the success rate is low, EI is favored to explore new regions. When the success rate is high, LCB is favored to exploit the current best region.
*   **Global Search Probability:** Introducing a small probability of sampling from the entire search space helps to escape local optima and maintain diversity in the search.
*   **Success Rate Based Adaptation:** The success rate is used to dynamically adjust the trust region radius, kappa, and the weighting between LCB and EI. This allows the algorithm to adapt to the characteristics of the objective function.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm

class ATRBO_DKAISCA_EI:
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
        self.global_search_prob = 0.05 # Probability of global search

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

    def _expected_improvement(self, X, gp):
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
        return ei

    def _acquisition_function(self, X, gp, success_rate):
        # Implement Lower Confidence Bound acquisition function
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        lcb = mu - self.kappa * sigma
        ei = self._expected_improvement(X, gp)

        # Adaptive weighting between LCB and EI
        return (1 - success_rate) * ei + success_rate * lcb

    def _select_next_point(self, gp):
        # Select the next point to evaluate within the trust region
        if np.random.rand() < self.global_search_prob:
            # Global search
            n_samples = 100 * self.dim
            samples = self._sample_points(n_samples)
        else:
            # Local search within trust region
            n_samples = 100 * self.dim
            samples = self._sample_points(n_samples, center=self.best_x, radius=self.trust_region_radius)

        if len(self.success_history) > 10:
            success_rate = np.mean(self.success_history[-10:])
        else:
            success_rate = 0.5 # Initial guess

        acq_values = self._acquisition_function(samples, gp, success_rate)
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
                self.kappa *= self.rho * 0.9 # Reduced kappa decrease.
                self.success_history.append(True)
                self.best_x = next_x.copy() # global best, keep it.
            else:
                self.trust_region_radius *= self.rho  # Shrink
                # Stochastic expansion
                self.trust_region_radius *= (1 + np.random.normal(0, 0.05) * np.sqrt(self.dim))
                self.kappa /= (self.rho*0.9) # increase kappa more when unsuccessful
                self.success_history.append(False)
                
            # Trust region center adaptation: adapt the trust region to the best point within the region
            samples_in_tr = self._sample_points(n_points=100, center=self.best_x, radius=self.trust_region_radius)
            mu_tr, _ = gp.predict(samples_in_tr, return_std=True)
            best_index_tr = np.argmin(mu_tr)
            self.best_x = samples_in_tr[best_index_tr].copy()

            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)
            self.kappa = np.clip(self.kappa, 0.1, 10.0)
            
            # Adjust rho based on success history
            if len(self.success_history) > 10:
                success_rate = np.mean(self.success_history[-10:])
                self.rho = 0.9 + 0.09 * success_rate #adaptive rho. Higher success rate leads to higher rho, and thus slower shrinking.

                # Aggressive expansion if success rate is high
                if success_rate > self.success_rate_threshold:
                    self.trust_region_radius /= (self.rho * (1 + (success_rate - self.success_rate_threshold)))

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ATRBO_DKAISCA_EI got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1734 with standard deviation 0.0985.

took 1198.81 seconds to run.