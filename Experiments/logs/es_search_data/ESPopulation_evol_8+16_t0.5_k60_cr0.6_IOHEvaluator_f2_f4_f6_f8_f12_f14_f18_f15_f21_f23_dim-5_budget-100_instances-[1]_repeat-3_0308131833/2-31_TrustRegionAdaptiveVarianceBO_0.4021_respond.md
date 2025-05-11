# Description
**TrustRegionAdaptiveVarianceBO with Dynamic Kernel and Exploration Control (TRAVBO-DKEC):** This enhanced algorithm builds upon the TrustRegionAdaptiveVarianceBO framework by incorporating a dynamic kernel adaptation strategy for the Gaussian Process (GP) and a refined exploration control mechanism. The GP kernel's length scale is optimized during each fitting process to better capture the function's landscape. The exploration weight is dynamically adjusted based not only on the predictive variance but also on the success rate of recent evaluations, leading to a more adaptive exploration-exploitation trade-off. A jitter mechanism is introduced to the local search to escape local optima more effectively.

# Justification
The key improvements are:

1.  **Dynamic Kernel Adaptation:** Optimizing the kernel's length scale allows the GP to adapt to varying function landscapes, improving its predictive accuracy and the effectiveness of the acquisition function. This is done by setting `length_scale_bounds` to a non-fixed value.
2.  **Enhanced Exploration Control:** The exploration weight is adjusted based on both the GP's variance and the recent success rate. This allows the algorithm to dynamically shift its focus between exploration and exploitation, potentially improving its convergence. The success rate is calculated based on the improvement of the best objective value.
3.  **Jittered Local Search:** Adding a small amount of random noise (jitter) to the local search samples can help the algorithm escape local optima and find better solutions.

These changes aim to improve the algorithm's adaptability and robustness to different function landscapes, leading to better overall performance.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

class TrustRegionAdaptiveVarianceBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim
        self.gp = None
        self.trust_region_radius = 2.0 # Initial trust region radius
        self.trust_region_shrink = 0.5 # Shrink factor for trust region
        self.trust_region_expand = 1.5 # Expansion factor for trust region
        self.success_threshold = 0.75 # Threshold for trust region expansion
        self.failure_threshold = 0.25 # Threshold for trust region contraction
        self.trust_region_center = np.zeros(dim) # Initial trust region center
        self.best_x = None
        self.best_y = np.inf
        self.local_search_radius = 0.1
        self.exploration_weight = 0.1 # Initial exploration weight
        self.exploration_decay = 0.95 # Decay factor for exploration weight
        self.min_variance_threshold = 0.01 # Minimum variance threshold for exploration
        self.success_history = [] # Keep track of recent success
        self.success_window = 5  # Window size for success rate calculation
        self.jitter = 1e-6 # Jitter for local search

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points within the trust region
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        scaled_sample = qmc.scale(sample, -self.trust_region_radius, self.trust_region_radius)
        return scaled_sample + self.trust_region_center

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None:
            return np.zeros((len(X), 1)) # Return zeros if the model is not fitted yet

        mu, sigma = self.gp.predict(X, return_std=True)

        # Adaptive exploration weight based on variance and success rate
        mean_sigma = np.mean(sigma)
        success_rate = np.mean(self.success_history[-self.success_window:]) if self.success_history else 0.5 # Default to 0.5 initially

        if mean_sigma < self.min_variance_threshold:
            self.exploration_weight = max(self.exploration_weight * self.exploration_decay, 0.01)  # Reduce exploration, but keep a minimum
        
        # Adjust exploration based on success rate
        if success_rate > 0.7:
             self.exploration_weight = max(self.exploration_weight * 0.9, 0.01) # Reduce exploration when doing well
        elif success_rate < 0.3:
            self.exploration_weight = min(self.exploration_weight * 1.1, 0.5) # Increase exploration when not doing well

        # Expected Improvement acquisition function
        improvement = self.best_y - mu
        Z = improvement / (sigma + 1e-9)  # Add a small constant to avoid division by zero
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # Avoid division by zero
        
        # Incorporate adaptive variance and trust region
        acquisition = ei + self.exploration_weight * sigma
        return acquisition.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Optimize the acquisition function within the trust region
        # return array of shape (batch_size, n_dims)

        # Generate candidate points within the trust region
        x_tries = self._sample_points(batch_size * 10)

        # Clip the points to stay within the bounds
        x_tries = np.clip(x_tries, self.bounds[0], self.bounds[1])

        acq_values = self._acquisition_function(x_tries)

        # Select the top batch_size points based on the acquisition function values
        indices = np.argsort(acq_values.flatten())[::-1][:batch_size]
        return x_tries[indices]

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y
    
    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        # Do not change the function signature
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))
        
        # Update best observed solution
        best_index = np.argmin(self.y)
        if self.y[best_index, 0] < self.best_y:
            self.best_x = self.X[best_index]
            self.best_y = self.y[best_index, 0]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_X = np.clip(initial_X, self.bounds[0], self.bounds[1])
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        while self.n_evals < self.budget:
            # Optimization

            # Fit GP model
            self.gp = self._fit_model(self.X, self.y)

            # Select points by acquisition function
            batch_size = min(10, self.budget - self.n_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Calculate the ratio of improvement
            predicted_improvement = self.best_y - self.gp.predict(self.best_x.reshape(1, -1))[0]
            actual_improvement = self.best_y - min(self.y)[0]

            if abs(predicted_improvement) < 1e-9:
                ratio = 0.0
            else:
                ratio = actual_improvement / predicted_improvement

            # Adjust trust region size
            if ratio > self.success_threshold:
                self.trust_region_radius *= self.trust_region_expand
            elif ratio < self.failure_threshold:
                self.trust_region_radius *= self.trust_region_shrink
                self.trust_region_radius = max(self.trust_region_radius, 0.1)

            # Update trust region center
            self.trust_region_center = self.best_x
            
            # Update success history
            self.success_history.append(1 if actual_improvement > 0 else 0)
            if len(self.success_history) > self.success_window:
                self.success_history.pop(0)

            # Local search around the best solution
            if self.best_x is not None and self.n_evals < self.budget:
                local_X = self._sample_points(1) * self.local_search_radius + self.best_x + np.random.normal(0, self.jitter, self.dim)
                local_X = np.clip(local_X, self.bounds[0], self.bounds[1])
                local_y = self._evaluate_points(func, local_X)
                self._update_eval_points(local_X, local_y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm TrustRegionAdaptiveVarianceBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1766 with standard deviation 0.1050.

took 23.60 seconds to run.