# Description
**Adaptive Trust Region Bayesian Optimization with Stochastic Hybrid Radius and Gradient-Informed Kappa (ATRBO-SHRGK):** This algorithm combines the strengths of ATRBO-DKAIS and ATRBO-HKRA by incorporating stochastic radius adjustments for escaping local optima, a hybrid kappa strategy that balances success/failure rates and GP variance, and gradient information to guide exploration. The algorithm adaptively adjusts the trust region radius based on success/failure, with a stochastic component for exploration. It uses a hybrid kappa that considers both success/failure counts and GP variance, and further refines kappa by incorporating gradient information to encourage exploration in promising regions.

# Justification
This algorithm builds upon ATRBO-DKAIS and ATRBO-HKRA to create a more robust and efficient optimization strategy.

1.  **Stochastic Radius Adjustment (from ATRBO\_DKAIS):** The stochastic expansion of the trust region radius helps to prevent premature convergence by occasionally exploring areas outside the immediate vicinity of the current best solution.
2.  **Hybrid Kappa Adjustment (from ATRBO\_HKRA):** The hybrid kappa strategy, which combines success/failure rates and GP variance, provides a more adaptive exploration-exploitation trade-off. This allows the algorithm to dynamically adjust its exploration behavior based on the characteristics of the objective function and the progress of the optimization.
3. **Gradient-Informed Kappa:** Incorporating gradient information into the kappa adjustment allows the algorithm to focus exploration on regions with potentially steeper slopes to better optima. This can lead to faster convergence and improved performance.
4. **Adaptive Rho:** Adjusting `rho` based on the recent success rate allows for finer control over the trust region size.
5. **Trust Region Center Adaptation:** Moving the trust region center to the best point found within the trust region allows the algorithm to more accurately focus on promising local areas.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ATRBO_SHRGK:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0  # the number of function evaluations
        self.n_init = min(20 * dim, self.budget // 5)  # Increased samples for initial exploration
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.5  # Initial trust region radius
        self.rho = 0.95  # Shrinking factor
        self.kappa = 2.0 + np.log(dim)  # Exploration-exploitation trade-off for LCB, adaptive to dimension
        self.success_count = 0
        self.failure_count = 0
        self.min_evals_for_adjust = 5 * dim
        self.kappa_success_failure_weight = 0.5  # Weight for combining success/failure and variance based kappa adjustment
        self.success_history = []  # Keep track of successful moves

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

    def _acquisition_function(self, X, gp, return_grad=False):
        # Implement Lower Confidence Bound acquisition function
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        
        if return_grad:
            dmu_dx = np.zeros_like(X)
            for i in range(X.shape[0]):
                def obj(x):
                    return gp.predict(x.reshape(1, -1), return_std=False)
                res = minimize(obj, X[i], method='BFGS', jac=None,
                               options={'disp': False})
                dmu_dx[i] = res.jac
            return mu - self.kappa * sigma, dmu_dx
        else:
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
            self.best_x = self.X[idx].copy()

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop

        # Initial exploration: Combine uniform sampling with sampling around the best seen point
        n_uniform = self.n_init // 2
        n_around_best = self.n_init - n_uniform

        initial_X_uniform = self._sample_points(n_uniform)
        initial_X_best = self._sample_points(n_around_best, center=self.bounds[1] / 2,
                                              radius=np.max(self.bounds[1] - self.bounds[0]) / 4)  # Sampling around the middle of the search space as initial guess

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
            if self.n_evals > self.n_init + self.min_evals_for_adjust:
                # Radius Adjustment (DKRA component)
                if next_y < self.best_y:
                    self.success_count += 1
                    self.failure_count = 0
                    success_ratio = self.success_count / (self.success_count + self.failure_count + 1e-9)
                    self.trust_region_radius /= (0.9 + 0.09 * success_ratio)  # Expand faster with higher success
                    self.success_history.append(True)
                else:
                    self.failure_count += 1
                    self.success_count = 0
                    failure_ratio = self.failure_count / (self.success_count + self.failure_count + 1e-9)
                    self.trust_region_radius *= (0.9 + 0.09 * failure_ratio)  # Shrink faster with higher failure
                    self.trust_region_radius *= (1 + np.random.normal(0, 0.05))  # Stochastic expansion
                    self.success_history.append(False)

                self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2,
                                                    np.max(self.bounds[1] - self.bounds[0]) / 2)

                # Kappa Adjustment (Hybrid: DKRA + GP variance + Gradient)
                mu, sigma = gp.predict(self.X, return_std=True)
                avg_sigma = np.mean(sigma)
                kappa_gp_component = 1.0 + np.log(1 + avg_sigma)  # Example GP variance component, can be tuned

                success_ratio = self.success_count / (self.success_count + self.failure_count + 1e-9)
                kappa_success_failure_component = (0.9 + 0.09 * success_ratio)

                # Gradient Information
                _, dmu_dx = self._acquisition_function(self.X, gp, return_grad=True)
                avg_gradient_magnitude = np.mean(np.linalg.norm(dmu_dx, axis=1))
                kappa_gradient_component = 1.0 + np.log(1 + avg_gradient_magnitude)

                self.kappa = (self.kappa_success_failure_weight * kappa_success_failure_component +
                               (1 - self.kappa_success_failure_weight) * (0.5 * kappa_gp_component + 0.5 * kappa_gradient_component))

                self.kappa = np.clip(self.kappa, 0.1, 10.0)

                # Adjust rho based on success history
                if len(self.success_history) > 10:
                    success_rate = np.mean(self.success_history[-10:])
                    self.rho = 0.9 + 0.09 * success_rate  # adaptive rho. Higher success rate leads to higher rho, and thus slower shrinking.

            # Trust Region Center Update
            if next_y < self.best_y:
                self.best_x = next_x.copy()  # Adapt trust region center

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ATRBO_SHRGK got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1840 with standard deviation 0.1029.

took 7109.76 seconds to run.