# Description
**AGATBO_ImprovedGradient_v2: Adaptive Gradient-Aware Trust Region Bayesian Optimization with Improved Gradient Estimation, Adaptive Gradient Weight, and Enhanced Trust Region Management.** This algorithm builds upon AGATBO_ImprovedGradient by incorporating several enhancements to improve its performance and robustness. These include: 1) An improved trust region update mechanism based on the prediction variance, 2) adaptive adjustment of the gradient weight based on the trust region radius and model uncertainty, and 3) a more robust model agreement check using Spearman's rank correlation. The goal is to achieve a better balance between exploration and exploitation, leading to faster convergence and improved optimization results.

# Justification
The key improvements are justified as follows:

1.  **Trust Region Update with Variance:** The original trust region update relies solely on model agreement, which can be noisy. By incorporating the prediction variance (uncertainty) from the GP model, the trust region adaptation becomes more informed. High variance suggests the model is uncertain, prompting shrinkage, while good agreement allows for expansion. This helps to avoid premature convergence and promotes exploration in uncertain regions.

2.  **Adaptive Gradient Weight:** The gradient weight is dynamically adjusted based on both the trust region radius and the model uncertainty (variance). This allows for a more refined balance between exploration (gradient-based) and exploitation (EI-based). A smaller trust region or high uncertainty increases the gradient weight, encouraging exploration.

3.  **Spearman's Rank Correlation:** Using Spearman's rank correlation for model agreement provides a more robust measure compared to Pearson correlation, as it is less sensitive to outliers and non-linear relationships between predicted and observed values.

These modifications aim to make the algorithm more adaptive to the characteristics of the objective function and improve its overall performance.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import minimize
import warnings
from scipy.stats import spearmanr

class AGATBO_ImprovedGradient_v2:
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
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 2.0
        self.model_agreement_threshold = 0.5  # Lower threshold for Spearman correlation
        self.kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        self.model = None

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.Halton(d=self.dim, seed=42)
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
            # Adaptive gradient weight
            adaptive_gradient_weight = self.gradient_weight * (1 + 1/(1 + self.trust_region_radius)) * (1 + np.mean(sigma))
            ei = ei + adaptive_gradient_weight * gradient_norm

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
        batch_size = 5
        while self.n_evals < self.budget:
            # Optimization
            self.model = self._fit_model(self.X, self.y)
            if self.model is None:
                print("GP model fitting failed. Returning current best.")
                return self.best_y, self.best_x

            # select points by acquisition function
            next_X = self._select_next_points(batch_size, trust_region_center)
            next_y = self._evaluate_points(func, next_X)
            
            # Model agreement check using Spearman's rank correlation
            predicted_y, sigma = self.model.predict(next_X, return_std=True)
            agreement, _ = spearmanr(next_y.flatten(), predicted_y.flatten())

            # Adjust trust region size based on model agreement and variance
            mean_sigma = np.mean(sigma)
            if np.isnan(agreement) or agreement < self.model_agreement_threshold:
                self.trust_region_radius *= self.trust_region_shrink_factor * (1 + mean_sigma) # Shrink more with high variance
            else:
                self.trust_region_radius *= self.trust_region_expand_factor
                self.trust_region_radius = min(self.trust_region_radius, 5.0) # Limit expansion
            
            self._update_eval_points(next_X, next_y)

            # Update trust region center
            trust_region_center = self.X[np.argmin(self.y)]

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AGATBO_ImprovedGradient_v2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1494 with standard deviation 0.1025.

took 556.88 seconds to run.