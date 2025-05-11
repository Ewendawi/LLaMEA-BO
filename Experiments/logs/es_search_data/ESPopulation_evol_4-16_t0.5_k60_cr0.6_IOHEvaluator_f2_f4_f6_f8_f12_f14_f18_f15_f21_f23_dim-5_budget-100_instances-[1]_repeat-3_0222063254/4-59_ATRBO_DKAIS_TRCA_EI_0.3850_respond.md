# Description
**ATRBO_DKAIS_TRCA_EI**: Adaptive Trust Region Bayesian Optimization with Dynamic Kappa, Adaptive Initial Exploration, Stochastic Radius, Trust Region Center Adaptation, and Expected Improvement. This algorithm builds upon ATRBO_DKAIS_TRCA by replacing the Lower Confidence Bound (LCB) acquisition function with Expected Improvement (EI). EI is less sensitive to the kappa parameter and better balances exploration and exploitation, especially in later stages of optimization. Furthermore, the initial exploration phase is enhanced with Latin Hypercube Sampling (LHS) over the entire search space, providing a more diverse initial sample.

# Justification
The key change is the use of Expected Improvement (EI) as the acquisition function instead of Lower Confidence Bound (LCB).

*   **Expected Improvement:** EI directly estimates the expected amount of improvement over the current best solution. This makes the algorithm less sensitive to the kappa parameter (exploration-exploitation trade-off) compared to LCB. EI also encourages exploration in areas where the GP model has high uncertainty and potential for improvement, even if the predicted mean is not very low.
*   **Latin Hypercube Sampling for Initial Exploration:** Using LHS for the initial exploration phase ensures a more uniform coverage of the search space compared to the previous combination of uniform and best-point-centered sampling. This helps the algorithm to better understand the landscape of the objective function and avoid getting stuck in local optima early on.
*   **Simplified Kappa Update:** The kappa update is simplified to depend only on the success history. The GP uncertainty is removed from the update rule, as EI already implicitly handles uncertainty.
*   **Stochastic Radius Adjustment with Gaussian Noise:** The stochastic radius adjustment is retained to escape local optima.
*   **Trust Region Center Adaptation:** The trust region center adaptation is retained to focus the search on promising local areas.
*   **Adaptive Rho:** Adaptive rho is retained to adjust the shrinking factor based on success history.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ATRBO_DKAIS_TRCA_EI:
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
        self.tr_center_adaptation_frequency = 5 # Adapt TR center every 5 iterations
        self.iteration = 0

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
        
        improvement = mu - self.best_y
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # avoid division by zero

        return ei

    def _select_next_point(self, gp):
        # Select the next point to evaluate within the trust region
        n_samples = 100 * self.dim
        samples = self._sample_points(n_samples, center=self.best_x, radius=self.trust_region_radius)
        acq_values = self._acquisition_function(samples, gp)
        best_index = np.argmax(acq_values) #EI is to be maximized
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

        # Initial exploration: Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=42)
        initial_X = sampler.random(n=self.n_init)
        initial_X = qmc.scale(initial_X, self.bounds[0], self.bounds[1])

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
                self.success_history.append(True)
                self.best_x = next_x.copy() # global best, keep it.
            else:
                self.trust_region_radius *= self.rho  # Shrink
                self.trust_region_radius *= (1 + np.random.normal(0, 0.05)) #Stochastic expansion
                self.success_history.append(False)
            
            # Trust region center adaptation: adapt the trust region to the best point within the region
            if self.iteration % self.tr_center_adaptation_frequency == 0:
                samples_in_tr = self._sample_points(n_points=100, center=self.best_x, radius=self.trust_region_radius)
                mu_tr, _ = gp.predict(samples_in_tr, return_std=True)
                best_index_tr = np.argmin(mu_tr)
                self.best_x = samples_in_tr[best_index_tr].copy()

            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)
            
            # Adjust rho based on success history
            if len(self.success_history) > 10:
                success_rate = np.mean(self.success_history[-10:])
                self.rho = 0.9 + 0.09 * success_rate #adaptive rho. Higher success rate leads to higher rho, and thus slower shrinking.

            # Adjust kappa based on success history
            if len(self.success_history) > 10:
                success_rate = np.mean(self.success_history[-10:])
                self.kappa = 1.0 + 9.0 * (1 - success_rate) #adaptive kappa. Lower success rate leads to higher kappa, and thus more exploration.
            self.kappa = np.clip(self.kappa, 0.1, 10.0)

            self.iteration += 1

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ATRBO_DKAIS_TRCA_EI got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1659 with standard deviation 0.1072.

took 1249.09 seconds to run.