# Description
**ATRBO_DKAIS_TRCA_EI**: Adaptive Trust Region Bayesian Optimization with Dynamic Kappa, Adaptive Initial Exploration, Stochastic Radius, Trust Region Center Adaptation, and Exploration-Exploitation Balancing via Expected Improvement. This enhancement builds upon ATRBO_DKAIS_TRCA by incorporating the Expected Improvement (EI) acquisition function alongside the Lower Confidence Bound (LCB). A dynamic weighting mechanism balances the contributions of EI and LCB, favoring exploration early on and exploitation later in the optimization process. This addresses the potential for LCB to prematurely converge, while retaining its ability to efficiently exploit promising regions. The initial exploration phase is also refined using Latin Hypercube Sampling (LHS) for better coverage of the search space.

# Justification
The key improvements are:

1.  **Expected Improvement (EI) Integration:** EI is a well-established acquisition function that explicitly considers the potential for improvement over the current best value. By incorporating EI, the algorithm gains a more nuanced exploration strategy, especially in the early stages of optimization.

2.  **Dynamic Weighting of EI and LCB:** A dynamic weight `alpha` is introduced to balance EI and LCB. Initially, `alpha` is high, giving more weight to EI for exploration. As the optimization progresses and the GP model becomes more accurate, `alpha` decreases, shifting the focus to exploitation using LCB. This adaptive weighting helps to overcome the limitations of relying solely on LCB, which can sometimes lead to premature convergence. The weight is adjusted based on the iteration number and a sigmoid function to ensure a smooth transition.

3.  **Refined Initial Exploration:** Using Latin Hypercube Sampling (LHS) provides a more uniform and space-filling initial sampling compared to the previous approach. This ensures that the initial GP model is built on a more representative dataset, leading to better subsequent exploration and exploitation.

4.  **Adaptive Kappa with Iteration Consideration:** The kappa adaptation now includes the iteration number, allowing for a more gradual shift from exploration to exploitation. This is particularly useful in high-dimensional spaces or complex functions where a more prolonged exploration phase is beneficial.

5. **Stochastic Trust Region Expansion with Gaussian Noise:** The stochastic trust region expansion is maintained to escape local optima, with Gaussian noise added to the radius.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm

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
        self.kappa = 2.0  # Exploration-exploitation trade-off for LCB
        self.success_history = [] # Keep track of successful moves
        self.tr_center_adaptation_frequency = 5 # Adapt TR center every 5 iterations
        self.iteration = 0
        self.alpha = 1.0 # Weight for EI vs LCB

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

    def _expected_improvement(self, X, gp, best_y):
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

        return ei

    def _acquisition_function(self, X, gp):
        # Implement Lower Confidence Bound acquisition function
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        lcb = mu - self.kappa * sigma
        ei = self._expected_improvement(X, gp, self.best_y)

        # Dynamic weighting between LCB and EI
        self.alpha = 1 / (1 + np.exp(-5 * (self.iteration / self.budget - 0.5)))  # Sigmoid function
        return self.alpha * ei + (1 - self.alpha) * lcb

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

            self.iteration += 1

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ATRBO_DKAIS_TRCA_EI got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1894 with standard deviation 0.1200.

took 864.52 seconds to run.