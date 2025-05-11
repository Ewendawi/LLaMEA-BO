# Description
**AdaptiveTrustRegionBO with Dynamic Radius Adjustment and EI Improvement (ATRBO-DR):** This algorithm builds upon the AdaptiveTrustRegionBO framework by introducing a more responsive trust region radius adjustment mechanism and refining the Expected Improvement (EI) acquisition function. The radius adjustment now considers the uncertainty (sigma) predicted by the Gaussian Process, allowing for more aggressive shrinking in regions of high confidence and slower expansion in uncertain areas. We also introduce a lower bound on the trust region radius relative to the GP's length scale to prevent premature convergence. The EI acquisition function is modified to incorporate a temperature parameter that controls the exploration-exploitation trade-off, annealing over time to favor exploitation as the budget is consumed.

# Justification
The original AdaptiveTrustRegionBO exhibited good exploration but could benefit from a more nuanced trust region adaptation strategy and a better balance between exploration and exploitation. The following changes were made:

1.  **Dynamic Radius Adjustment Based on GP Uncertainty:** Instead of a fixed decay/grow rate, the radius adjustment now depends on the average predicted uncertainty (sigma) within the trust region. High uncertainty leads to slower shrinking and faster expansion, while low uncertainty results in the opposite. This allows the algorithm to adapt to the local landscape more effectively.

2.  **Minimum Radius Relative to GP Length Scale:** To prevent the trust region from shrinking too much when the GP has learned a good model, a minimum radius is enforced based on the average length scale of the RBF kernel. This ensures that the algorithm continues to explore the local region even when the GP is confident.

3.  **Temperature-Controlled EI:** A temperature parameter is introduced into the EI calculation. This parameter is annealed over time, starting high to encourage exploration and decreasing to favor exploitation. This helps the algorithm to converge more quickly in the later stages of the optimization.

4. **Batch Size Adjustment**: The batch size is dynamically adjusted based on the trust region radius. Smaller radius means smaller batch size, which allows for more fine-grained local search.

These changes aim to improve the convergence rate and overall performance of the AdaptiveTrustRegionBO algorithm by providing a more adaptive and robust exploration-exploitation strategy.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from scipy.optimize import minimize

class AdaptiveTrustRegionBO_DR:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * (dim + 1)
        self.trust_region_center = np.zeros(dim)  # Initialize trust region center
        self.trust_region_radius = 2.5  # Initial trust region radius (half of the search space)
        self.min_radius = 0.1
        self.radius_decay = 0.95
        self.radius_grow = 1.1
        self.gp = None
        self.temperature = 1.0  # Initial temperature for EI
        self.temperature_decay = 0.99  # Decay rate for temperature

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points within the trust region
        # return array of shape (n_points, n_dims)
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        sample = sampler.random(n=n_points)
        scaled_sample = qmc.scale(sample, -1.0, 1.0) # Scale to [-1, 1]
        
        # Map to trust region
        points = self.trust_region_center + scaled_sample * self.trust_region_radius
        
        # Clip to bounds
        points = np.clip(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * \
                 RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + \
                 WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-7, 1e-3))
        
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function (Expected Improvement)
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None or self.X is None or self.y is None:
            return np.zeros((len(X), 1))  # Return zero if model is not fitted yet

        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)
        y_best = np.min(self.y)
        gamma = (y_best - mu) / sigma
        ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        
        # Apply temperature to EI
        ei = ei * self.temperature

        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Sample candidate points
        candidates = self._sample_points(100 * self.dim)
        
        # Calculate acquisition function values
        ei = self._acquisition_function(candidates)
        
        # Select top batch_size candidates based on EI
        selected_indices = np.argsort(ei.flatten())[-batch_size:]
        selected_points = candidates[selected_indices]
        
        return selected_points

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)
    
    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        # Do not change the function signature
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        
        while self.n_evals < self.budget:
            # Fit the Gaussian Process model
            gp = self._fit_model(self.X, self.y)
            
            # Calculate average sigma within the trust region
            mu, sigma = self.gp.predict(self._sample_points(100), return_std=True)
            avg_sigma = np.mean(sigma)

            # Adjust trust region radius
            if self.gp.kernel_ is not None:
                length_scale = self.gp.kernel_.get_params()['k1__k2__length_scale']
                min_radius = np.mean(length_scale) * 0.1
            else:
                min_radius = 0.1 # Default value if kernel is not properly initialized

            if current_best_y < best_y:
                # Improvement: move trust region center and shrink radius
                self.trust_region_center = current_best_x
                self.trust_region_radius *= (self.radius_decay + (1-self.radius_decay) * avg_sigma/np.max(sigma))
                best_y = current_best_y
                best_x = current_best_x
            else:
                # No improvement: potentially expand trust region
                if self.trust_region_radius < 1.0:
                    self.trust_region_radius *= (self.radius_grow - (self.radius_grow-1) * avg_sigma/np.max(sigma))
                    self.trust_region_radius = min(self.trust_region_radius, 2.5) # Limit to initial radius
            
            self.trust_region_radius = max(self.trust_region_radius, min_radius)
            self.trust_region_radius = max(self.trust_region_radius, self.min_radius)

            # Select the next points to evaluate
            batch_size = min(self.budget - self.n_evals, int(5 * self.trust_region_radius/2.5) + 1) # Adjust batch size based on radius
            next_X = self._select_next_points(batch_size)
            
            # Evaluate the points
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Update the best solution
            best_idx = np.argmin(self.y)
            current_best_y = self.y[best_idx][0]
            current_best_x = self.X[best_idx]

            # Decay temperature
            self.temperature *= self.temperature_decay

        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<AdaptiveTrustRegionBO_DR>", line 142, in __call__
 140 |                 min_radius = 0.1 # Default value if kernel is not properly initialized
 141 | 
 142->             if current_best_y < best_y:
 143 |                 # Improvement: move trust region center and shrink radius
 144 |                 self.trust_region_center = current_best_x
UnboundLocalError: local variable 'current_best_y' referenced before assignment
