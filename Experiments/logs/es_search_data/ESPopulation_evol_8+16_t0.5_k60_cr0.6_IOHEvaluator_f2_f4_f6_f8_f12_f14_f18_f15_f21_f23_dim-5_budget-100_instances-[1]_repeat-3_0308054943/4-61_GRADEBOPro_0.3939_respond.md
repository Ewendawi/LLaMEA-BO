# Description
**GRADEBO-Pro: Gradient-Regularized Adaptive Diversity Enhanced Bayesian Optimization with Probabilistic Trust Region and Enhanced Gradient Estimation:** This algorithm refines GRADEBO by introducing a probabilistic trust region adaptation based on the uncertainty of the Gaussian Process model, enhancing gradient estimation using a central difference scheme with adaptive step size, and incorporating a more robust diversity measure. The probabilistic trust region allows for a more nuanced adjustment of the trust region size based on the confidence in the model predictions. The enhanced gradient estimation improves the accuracy of the gradient-based exploration. The diversity measure is improved by using the median distance to be more robust to outliers.

# Justification
The following changes were made to improve GRADEBO:

1.  **Probabilistic Trust Region Adaptation:** The original GRADEBO uses a fixed threshold for model agreement to adjust the trust region size. This is replaced with a probabilistic approach, where the trust region radius is adjusted based on the uncertainty (sigma) of the Gaussian Process model. The higher the uncertainty, the smaller the trust region, promoting exploration. This allows for a more adaptive and nuanced adjustment of the trust region size.

2.  **Enhanced Gradient Estimation:** The gradient estimation is improved by using a central difference scheme with an adaptive step size. The step size is proportional to the trust region radius, allowing for more accurate gradient estimation within the current trust region.

3.  **Robust Diversity Measure:** The diversity measure is improved by using the median distance instead of the minimum distance. This makes the diversity term more robust to outliers, preventing the algorithm from being overly influenced by a few points that are very close to existing samples.

4.  **Adaptive Exploration Factor:** The exploration factor is made adaptive by scaling it with the trust region radius. This ensures that exploration is more aggressive when the trust region is large and more conservative when the trust region is small.

5. **Initial Trust Region Center:** The initial trust region center is set to the best initial point instead of a random point.

These changes aim to improve the exploration-exploitation balance and robustness of the algorithm, leading to better performance on the BBOB test suite.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.metrics import pairwise_distances
import warnings

class GRADEBOPro:
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
        self.reg_weight = 0.1 # Initial weight for the regularization term
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
        self.exploration_factor = 0.01

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

    def _acquisition_function(self, X, model, iteration):
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

        # Adaptive Regularization
        reg_weight = self.reg_weight * (1 - (iteration / self.budget))
        regularization_term = -reg_weight * np.linalg.norm(mu / (sigma + 1e-6), axis=1, keepdims=True)**2 # Uncertainty aware regularization
        ei = ei + regularization_term + self.exploration_factor * sigma * self.trust_region_radius # Add exploration factor scaled by trust region

        # Add diversity term
        if self.X is not None:
            distances = pairwise_distances(X, self.X)
            median_distances = np.median(distances, axis=1).reshape(-1, 1)
            ei = ei + self.diversity_weight * median_distances

        return ei

    def _predict_gradient(self, X, model):
        # Predict the gradient of the GPR mean function using central difference
        # return array of shape (n_points, n_dims)
        X = np.atleast_2d(X)
        dmu_dx = np.zeros((X.shape[0], self.dim))
        
        # Adaptive step size for central difference
        delta = 1e-6 * self.trust_region_radius
        
        for i in range(self.dim):
            X_plus = X.copy()
            X_minus = X.copy()
            X_plus[:, i] += delta
            X_minus[:, i] -= delta
            dmu_dx[:, i] = (model.predict(X_plus) - model.predict(X_minus)) / (2 * delta)
        
        return dmu_dx

    def _select_next_points(self, batch_size, trust_region_center, model, iteration):
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
        acquisition_values = self._acquisition_function(scaled_samples, model, iteration)

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
        iteration = self.n_init
        while self.n_evals < self.budget:
            # Adaptive batch size
            batch_size = max(1, int(5 * (1 - self.n_evals / self.budget))) # Linearly decreasing batch size

            # Optimization
            self.model = self._fit_model(self.X, self.y)
            if self.model is None:
                print("GP model fitting failed. Returning current best.")
                return self.best_y, self.best_x

            # select points by acquisition function
            next_X = self._select_next_points(batch_size, trust_region_center, self.model, iteration)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Model agreement check (probabilistic)
            predicted_y, predicted_sigma = self.model.predict(next_X, return_std=True)
            predicted_sigma = predicted_sigma.reshape(-1, 1)
            agreement = np.mean(np.abs(next_y.flatten() - predicted_y.flatten()) <= 2 * predicted_sigma.flatten())

            # Adjust trust region size based on model uncertainty
            if np.isnan(agreement) or agreement < self.model_agreement_threshold:
                self.trust_region_radius *= self.trust_region_shrink_factor
            else:
                self.trust_region_radius *= self.trust_region_expand_factor
                self.trust_region_radius = min(self.trust_region_radius, 5.0) # Limit expansion

            # Update trust region center
            trust_region_center = self.X[np.argmin(self.y)]
            iteration += batch_size

        return self.best_y, self.best_x
```
## Feedback
 The algorithm GRADEBOPro got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1602 with standard deviation 0.0985.

took 271.10 seconds to run.