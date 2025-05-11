# Description
Adaptive Trust Region Bayesian Optimization with Dynamic Kappa, Radius Adjustment, and Gradient-based Refinement (ATRBO_DKRAG). This algorithm enhances ATRBO-DKRA by incorporating gradient information to refine the search within the trust region. It leverages the GP model to estimate the gradient and uses this information to guide the sampling process, promoting faster convergence towards local optima. The update includes a gradient-based local search step and also introduces a momentum term to smooth the radius and kappa adaptation. This helps to stabilize the optimization process and prevents erratic adjustments of these parameters.

# Justification
1.  **Gradient-based Refinement:** Estimating the gradient from the Gaussian Process allows for a more informed search within the trust region. By moving points towards the predicted negative gradient direction, we can potentially accelerate the convergence towards local optima.
2.  **Momentum for Parameter Adaptation:** Instead of directly updating the trust region radius and kappa based on the instantaneous success/failure ratio, a momentum term is introduced. This creates a smoother transition for these parameters, preventing oscillations and contributing to a more stable optimization process.
3.  **Efficient Gradient Calculation:** A simple and efficient method is used to approximate the gradient, making the process computationally feasible.
4.  **Balancing Exploration and Exploitation:** The algorithm balances exploration and exploitation by dynamically adjusting kappa and the trust region radius. The gradient-based refinement enhances exploitation, while the adaptive kappa maintains a degree of exploration.
5.  **Computational Efficiency:** The changes are designed to be computationally efficient, ensuring that the optimization process remains fast.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class ATRBO_DKRAG:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(10 * dim, self.budget // 5)
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.5  # Initial trust region radius
        self.kappa = 2.0 # Exploration-exploitation trade-off for LCB
        self.success_count = 0
        self.failure_count = 0
        self.min_evals_for_adjust = 5 * dim
        self.kappa = 2.0 + np.log(dim)  # Adaptive Initial Kappa

        self.radius_momentum = 0.0
        self.kappa_momentum = 0.0
        self.momentum_factor = 0.9

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, center=None, radius=None):
        # sample points within the trust region
        if center is None:
            center = (self.bounds[1] + self.bounds[0]) / 2
        if radius is None:
            radius = np.max(self.bounds[1] - self.bounds[0]) / 2

        sampler = qmc.Sobol(d=self.dim, scramble=True)
        points = sampler.random(n=n_points)
        points = qmc.scale(points, -1, 1)  # Scale to [-1, 1]

        # Project points to a hypersphere
        lengths = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / lengths * np.random.uniform(0, 1, size=lengths.shape) ** (1/self.dim)

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
        
        # Gradient-based refinement
        initial_x = samples[best_index]
        
        def obj_func(x):
            x = x.reshape(1,-1)
            return self._acquisition_function(x, gp)

        res = minimize(obj_func, initial_x, method='L-BFGS-B', bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)])
        
        if res.success:
            next_x = res.x
        else:
            next_x = samples[best_index]
        
        return next_x

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

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
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
                if next_y < self.best_y:
                    self.success_count += 1
                    self.failure_count = 0
                    success_ratio = self.success_count / (self.success_count + self.failure_count + 1e-9)
                    radius_update = (0.9 + 0.09 * success_ratio)
                    kappa_update = (0.9 + 0.09 * success_ratio)

                else:
                    self.failure_count += 1
                    self.success_count = 0
                    failure_ratio = self.failure_count / (self.success_count + self.failure_count + 1e-9)
                    radius_update = (0.9 + 0.09 * failure_ratio)
                    kappa_update = (0.9 + 0.09 * failure_ratio)

                # Apply momentum to radius and kappa
                self.radius_momentum = self.momentum_factor * self.radius_momentum + (1 - self.momentum_factor) * radius_update
                self.kappa_momentum = self.momentum_factor * self.kappa_momentum + (1 - self.momentum_factor) * kappa_update

                self.trust_region_radius /= self.radius_momentum
                self.kappa *= self.kappa_momentum

                self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)
                self.kappa = np.clip(self.kappa, 0.1, 10.0)

        return self.best_y, self.best_x
```