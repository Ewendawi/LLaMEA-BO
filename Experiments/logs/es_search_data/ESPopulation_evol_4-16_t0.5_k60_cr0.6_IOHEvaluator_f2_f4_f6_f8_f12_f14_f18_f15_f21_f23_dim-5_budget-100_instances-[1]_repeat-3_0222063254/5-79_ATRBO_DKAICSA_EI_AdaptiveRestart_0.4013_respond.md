# Description
**ATRBO_DKAICSA_EI_AdaptiveRestart**: Adaptive Trust Region Bayesian Optimization with Dynamic Kappa, Adaptive Initial Exploration (Latin Hypercube), Trust Region Center Adaptation, Stochastic Radius Adjustment, Enhanced GP Fitting with Expected Improvement and Adaptive Restarts. This algorithm builds upon ATRBO_DKAICSA_Enhanced by replacing the LCB acquisition function with Expected Improvement (EI) and introducing an adaptive restart mechanism. The restart mechanism monitors the success rate and GP uncertainty to trigger a re-initialization of the trust region and GP model when the search stagnates. The EI acquisition function is less sensitive to the kappa parameter and can lead to better exploration-exploitation balance. The adaptive restart helps the algorithm escape local optima and explore different regions of the search space, especially in complex or multimodal problems.

# Justification
*   **Expected Improvement (EI) Acquisition Function:** EI is often more robust than LCB, especially when the GP model is not perfectly calibrated. It balances exploration and exploitation more effectively by considering both the predicted improvement and the uncertainty.
*   **Adaptive Restart Mechanism:** The algorithm monitors the success rate within the trust region and the average GP uncertainty. If the success rate falls below a threshold and the GP uncertainty is low (indicating stagnation), the algorithm restarts by re-initializing the trust region center to a randomly sampled location and refitting the GP model. This helps the algorithm escape local optima and explore new regions. The threshold for success rate and GP uncertainty are dynamically adjusted based on the problem dimensionality.
*   **Dynamic Kappa, Stochastic Radius Adjustment, and Adaptive Rho:** These components from the original ATRBO_DKAICSA_Enhanced are retained to maintain the adaptive nature of the trust region and exploration-exploitation trade-off.
*   **Latin Hypercube Sampling for Initial Exploration:** Latin Hypercube sampling provides a good initial coverage of the search space, helping to build a better initial GP model.
*   **Trust Region Center Adaptation:** Adapting the trust region center to the best point within the current trust region helps to focus the search on promising areas.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import uniform

class ATRBO_DKAICSA_EI_AdaptiveRestart:
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
        self.restart_flag = False # Flag to indicate restart
        self.restart_patience = 5 # Number of iterations to wait before restarting again
        self.restart_counter = 0

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
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42) # Increased restarts
        gp.fit(X, y)
        return gp

    def _expected_improvement(self, X, gp, best_y):
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # avoid division by zero
        return ei

    def _acquisition_function(self, X, gp):
        # Implement Expected Improvement acquisition function
        return self._expected_improvement(X, gp, self.best_y)

    def _select_next_point(self, gp):
        # Select the next point to evaluate within the trust region
        n_samples = 100 * self.dim
        samples = self._sample_points(n_samples, center=self.best_x, radius=self.trust_region_radius)
        acq_values = self._acquisition_function(samples, gp)
        best_index = np.argmax(acq_values)
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

        # Initial exploration: Combine uniform sampling with sampling around the center
        n_uniform = self.n_init // 2
        n_around_center = self.n_init - n_uniform

        initial_X_uniform = self._sample_points(n_uniform)
        initial_X_center = self._sample_points(n_around_center, center=(self.bounds[1] + self.bounds[0]) / 2, radius=np.max(self.bounds[1] - self.bounds[0]) / 4)

        initial_X = np.vstack((initial_X_uniform, initial_X_center))
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Initialize best_x to the best initial point
        self.best_x = self.X[np.argmin(self.y)].copy()

        while self.n_evals < self.budget:
            # Optimization
            if self.restart_flag:
                # Re-initialize trust region center and refit GP model
                self.best_x = uniform.rvs(loc=self.bounds[0], scale=self.bounds[1] - self.bounds[0], size=self.dim)
                self.trust_region_radius = 2.5
                self.restart_flag = False
                self.restart_counter = 0
                self.success_history = []
                gp = self._fit_model(self.X, self.y)
            else:
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

            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)
            self.kappa = np.clip(self.kappa, 0.1, 10.0)

            # Adjust rho based on success history
            if len(self.success_history) > 10:
                success_rate = np.mean(self.success_history[-10:])
                self.rho = 0.9 + 0.09 * success_rate #adaptive rho. Higher success rate leads to higher rho, and thus slower shrinking.

            # Adaptive Restart Mechanism
            if len(self.success_history) > 10:
                success_rate = np.mean(self.success_history[-10:])
                avg_sigma = np.mean(gp.predict(self.X, return_std=True)[1])

                # Dynamically adjust the thresholds based on dimensionality
                success_threshold = 0.1 + 0.01 * self.dim
                sigma_threshold = 0.1 - 0.005 * self.dim

                if success_rate < success_threshold and avg_sigma < sigma_threshold and self.restart_counter >= self.restart_patience:
                    self.restart_flag = True
                else:
                    self.restart_counter += 1

            self.iteration += 1

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ATRBO_DKAICSA_EI_AdaptiveRestart got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1742 with standard deviation 0.1072.

took 1295.59 seconds to run.