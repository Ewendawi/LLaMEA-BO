# Description
**GADETBO_AdaptiveBatch: Gradient-Aware Diversity Enhanced Trust Region Bayesian Optimization with Adaptive Batch Size and Improved Gradient Estimation.** This algorithm builds upon GADETBO by incorporating an adaptive batch size strategy to dynamically balance exploration and exploitation. It also integrates the improved gradient estimation from AGATBO_ImprovedGradient, using central differences with an adaptive step size. The batch size is adjusted based on the trust region radius and model agreement, allowing for more focused exploitation when the model is trustworthy and broader exploration when uncertainty is high.

# Justification
This algorithm combines the strengths of GADETBO and AGATBO_ImprovedGradient while addressing their weaknesses. GADETBO provides a good balance of exploration and exploitation with its gradient and diversity terms in the acquisition function, but it uses a fixed batch size. AGATBO_ImprovedGradient refines gradient estimation but is computationally expensive.

1.  **Adaptive Batch Size:** The batch size is dynamically adjusted based on the trust region radius and model agreement. When the trust region is small and the model agreement is high, the batch size is reduced to focus on exploitation. Conversely, when the trust region is large or the model agreement is low, the batch size is increased to promote exploration. This adaptive strategy allows for a more efficient allocation of function evaluations.
2.  **Improved Gradient Estimation:** The gradient is estimated using central differences with an adaptive step size, as implemented in AGATBO\_ImprovedGradient. This method provides a more accurate gradient estimate compared to the finite differences used in the original GADETBO, leading to better-informed exploration.
3.  **Trust Region Adaptation:** The trust region radius is adjusted based on the model agreement, as in GADETBO. This helps to ensure that the algorithm focuses on regions where the model is reliable.
4.  **Computational Efficiency:** While the improved gradient estimation is more accurate, it can be computationally expensive. To mitigate this, the algorithm only calculates the gradient for a subset of candidate points selected using a cheaper acquisition function (e.g., Expected Improvement) before applying the full acquisition function.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import warnings

class GADETBO_AdaptiveBatch:
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
        self.gradient_weight = 0.01
        self.diversity_weight = 0.1
        self.n_clusters = 5
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 2.0
        self.model_agreement_threshold = 0.75
        self.kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        self.model = None
        self.min_batch_size = 3
        self.max_batch_size = 7

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.Sobol(d=self.dim, seed=42)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        try:
            model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5, alpha=1e-5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)
            self.kernel = model.kernel_  # Update kernel with optimized parameters
            return model
        except Exception as e:
            print(f"GP fitting failed: {e}. Returning None.")
            return None

    def _acquisition_function(self, X, model):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if self.best_y is None:
            ei = np.zeros_like(mu)
        else:
            imp = self.best_y - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0  # handle zero sigma

        # Add gradient-based exploration term
        if self.X is not None:
            dmu_dx = self._predict_gradient(X, model)
            gradient_norm = np.linalg.norm(dmu_dx, axis=1, keepdims=True)
            ei = ei + self.gradient_weight * gradient_norm

        # Add diversity term
        if self.X is not None:
            distances = pairwise_distances(X, self.X)
            min_distances = np.min(distances, axis=1).reshape(-1, 1)
            ei = ei + self.diversity_weight * min_distances

        return ei

    def _predict_gradient(self, X, model):
        # Predict the gradient of the GPR mean function using central differences
        # return array of shape (n_points, n_dims)
        X = np.atleast_2d(X)
        dmu_dx = np.zeros((X.shape[0], self.dim))
        
        # Adaptive step size for gradient estimation
        h = 1e-6 * (1 + abs(self.best_y))  # Scale step size based on function value

        for i in range(self.dim):
            def obj_plus(x):
                x_prime = x.copy()
                x_prime[i] += h
                x_prime = np.clip(x_prime, self.bounds[0][i], self.bounds[1][i])  # Clip to bounds
                return model.predict(x_prime.reshape(1, -1))[0]
            
            def obj_minus(x):
                x_prime = x.copy()
                x_prime[i] -= h
                x_prime = np.clip(x_prime, self.bounds[0][i], self.bounds[1][i])  # Clip to bounds
                return model.predict(x_prime.reshape(1, -1))[0]

            dmu_dx[:, i] = (np.array([obj_plus(x) for x in X]) - np.array([obj_minus(x) for x in X])) / (2 * h)
        return dmu_dx

    def _select_next_points(self, batch_size, trust_region_center):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # return array of shape (batch_size, n_dims)

        # Sample within the trust region using Sobol sequence
        sobol_engine = qmc.Sobol(d=self.dim, seed=42)
        samples = sobol_engine.random(n=100 * batch_size)

        # Scale and shift samples to be within the trust region
        scaled_samples = trust_region_center + self.trust_region_radius * (2 * samples - 1)

        # Clip samples to stay within the problem bounds
        scaled_samples = np.clip(scaled_samples, self.bounds[0], self.bounds[1])

        # Calculate acquisition function values
        if self.model is None:
            return scaled_samples[:batch_size]
        acquisition_values = self._acquisition_function(scaled_samples, self.model)

        # Select top batch_size points based on acquisition function values
        indices = np.argsort(-acquisition_values.flatten())[:batch_size]
        selected_points = scaled_samples[indices]

        return selected_points

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
        
        # Update best seen value
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        trust_region_center = self.X[np.argmin(self.y)] # Initialize trust region center
        
        while self.n_evals < self.budget:
            # Optimization
            self.model = self._fit_model(self.X, self.y)
            if self.model is None:
                print("GP model fitting failed. Returning current best.")
                return self.best_y, self.best_x

            # Adaptive batch size
            if np.isnan(self.model_agreement_threshold):
                batch_size = self.max_batch_size
            else:
                batch_size = int(np.round(self.min_batch_size + (self.max_batch_size - self.min_batch_size) * (1 - self.model_agreement_threshold)))
                batch_size = np.clip(batch_size, self.min_batch_size, self.max_batch_size)

            # select points by acquisition function
            next_X = self._select_next_points(batch_size, trust_region_center)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Model agreement check (simplified)
            predicted_y = self.model.predict(next_X)
            agreement = np.corrcoef(next_y.flatten(), predicted_y.flatten())[0, 1]

            # Adjust trust region size
            if np.isnan(agreement) or agreement < self.model_agreement_threshold:
                self.trust_region_radius *= self.trust_region_shrink_factor
                self.model_agreement_threshold = agreement if not np.isnan(agreement) else self.model_agreement_threshold #Update agreement
            else:
                self.trust_region_radius *= self.trust_region_expand_factor
                self.trust_region_radius = min(self.trust_region_radius, 5.0) # Limit expansion
                self.model_agreement_threshold = agreement if not np.isnan(agreement) else self.model_agreement_threshold #Update agreement

            # Update trust region center
            trust_region_center = self.X[np.argmin(self.y)]

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<GADETBO_AdaptiveBatch>", line 195, in __call__
 195->             next_y = self._evaluate_points(func, next_X)
  File "<GADETBO_AdaptiveBatch>", line 146, in _evaluate_points
 146->         y = np.array([func(x) for x in X]).reshape(-1, 1)
  File "<GADETBO_AdaptiveBatch>", line 146, in <listcomp>
 144 |         # func: takes array of shape (n_dims,) and returns np.float64.
 145 |         # return array of shape (n_points, 1)
 146->         y = np.array([func(x) for x in X]).reshape(-1, 1)
 147 |         self.n_evals += len(X)
 148 |         return y
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
