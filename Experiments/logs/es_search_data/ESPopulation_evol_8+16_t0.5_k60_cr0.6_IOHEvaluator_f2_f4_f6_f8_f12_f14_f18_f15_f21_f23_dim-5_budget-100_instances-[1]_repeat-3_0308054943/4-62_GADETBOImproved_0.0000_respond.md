# Description
**GADETBO-Improved: Gradient-Aware Diversity Enhanced Trust Region Bayesian Optimization with Enhanced Gradient Prediction and Adaptive Diversity Weighting.** This algorithm builds upon GADETBO by incorporating several key improvements. First, it refines the gradient prediction using a more accurate central difference method with adaptive step size. Second, it introduces an adaptive diversity weighting scheme that adjusts the diversity weight based on the trust region radius, promoting exploration when the trust region is small and exploitation when it's large. Third, it implements a more robust model agreement check using Spearman correlation and incorporates a dynamic batch size adjustment based on the trust region size. Finally, it uses Halton sequence for initial sampling to achieve better space coverage.

# Justification
The improvements are justified as follows:

1.  **Enhanced Gradient Prediction:** The original GADETBO uses a simple finite difference method for gradient estimation. The improved version uses a more accurate central difference method with an adaptive step size, which can lead to more reliable gradient information and better exploration of the search space.
2.  **Adaptive Diversity Weighting:** The diversity weight in the original GADETBO is fixed. The improved version adaptively adjusts the diversity weight based on the trust region radius. When the trust region is small, the diversity weight is increased to encourage exploration. When the trust region is large, the diversity weight is decreased to promote exploitation. This adaptive weighting scheme can help to balance exploration and exploitation more effectively.
3.  **Robust Model Agreement Check:** The original GADETBO uses Pearson correlation for the model agreement check, which can be sensitive to outliers. The improved version uses Spearman correlation, which is more robust to outliers. This can lead to more reliable trust region adaptation.
4.  **Dynamic Batch Size Adjustment:** The improved version incorporates a dynamic batch size adjustment based on the trust region size. When the trust region is small, the batch size is increased to explore more points within the limited region. When the trust region is large, the batch size is decreased to focus on exploiting the promising region.
5.  **Halton Sequence for Initial Sampling:** Halton sequence provides better space-filling properties compared to Sobol sequence, especially for smaller sample sizes. This can lead to a better initial exploration of the search space.

These modifications aim to address the limitations of the original GADETBO and improve its performance on a wider range of black-box optimization problems.

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
from scipy.stats import spearmanr
import warnings

class GADETBOImproved:
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
        self.diversity_weight_initial = 0.1
        self.n_clusters = 5
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 2.0
        self.model_agreement_threshold = 0.5 # Reduced threshold for Spearman correlation
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
            ei = ei + self.gradient_weight * gradient_norm

        # Add diversity term
        if self.X is not None:
            distances = pairwise_distances(X, self.X)
            min_distances = np.min(distances, axis=1).reshape(-1, 1)
            diversity_weight = self.diversity_weight_initial * (1 - self.trust_region_radius / 5.0)  # Adaptive diversity weight
            ei = ei + diversity_weight * min_distances

        return ei

    def _predict_gradient(self, X, model):
        # Predict the gradient of the GPR mean function using central difference
        # return array of shape (n_points, n_dims)
        X = np.atleast_2d(X)
        dmu_dx = np.zeros((X.shape[0], self.dim))

        # Adaptive step size for finite differences
        delta = 1e-6 * self.trust_region_radius

        # Efficient gradient calculation using central differences
        for i in range(self.dim):
            X_plus = X.copy()
            X_minus = X.copy()
            X_plus[:, i] += delta
            X_minus[:, i] -= delta
            dmu_dx[:, i] = (model.predict(X_plus) - model.predict(X_minus)) / (2 * delta)

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

            # Dynamic batch size adjustment
            batch_size = int(3 + 7 * (1 - self.trust_region_radius / 5.0))  # batch_size between 3 and 10

            # select points by acquisition function
            next_X = self._select_next_points(batch_size, trust_region_center)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Model agreement check (Spearman correlation)
            predicted_y = self.model.predict(next_X)
            agreement, _ = spearmanr(next_y.flatten(), predicted_y.flatten())

            # Adjust trust region size
            if np.isnan(agreement) or agreement < self.model_agreement_threshold:
                self.trust_region_radius *= self.trust_region_shrink_factor
            else:
                self.trust_region_radius *= self.trust_region_expand_factor
                self.trust_region_radius = min(self.trust_region_radius, 5.0) # Limit expansion

            # Update trust region center
            trust_region_center = self.X[np.argmin(self.y)]

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<GADETBOImproved>", line 185, in __call__
 185->             next_y = self._evaluate_points(func, next_X)
  File "<GADETBOImproved>", line 140, in _evaluate_points
 140->         y = np.array([func(x) for x in X]).reshape(-1, 1)
  File "<GADETBOImproved>", line 140, in <listcomp>
 138 |         # func: takes array of shape (n_dims,) and returns np.float64.
 139 |         # return array of shape (n_points, 1)
 140->         y = np.array([func(x) for x in X]).reshape(-1, 1)
 141 |         self.n_evals += len(X)
 142 |         return y
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
