# Description
**GADETREBO_AdaptiveWeights: Gradient-Aware Diversity Enhanced Trust Region Bayesian Optimization with Adaptive Weights.** This algorithm refines GADETREBO by adaptively adjusting the weights of the gradient, diversity, and exploration terms in the acquisition function. The weights are dynamically adjusted based on the trust region radius and the model's uncertainty (estimated by the GPR's sigma). This adaptive weighting aims to improve the balance between exploration and exploitation throughout the optimization process. Additionally, the trust region update is improved by using Spearman's rank correlation and explicitly considering the uncertainty of the GPR model.

# Justification
The key improvements focus on adaptively weighting the different components of the acquisition function.

*   **Adaptive Weights:** Instead of fixed weights for gradient, diversity, and exploration, the algorithm dynamically adjusts these weights based on the trust region radius and the model's uncertainty. When the trust region is small (exploitation phase), the gradient and diversity weights are increased, emphasizing local refinement. When the trust region is large (exploration phase), the exploration weight is increased, encouraging broader search. Model uncertainty (sigma) is also used to modulate the exploration weight, increasing exploration in regions of high uncertainty.
*   **Improved Trust Region Update:** Uses Spearman's rank correlation for a more robust model agreement check compared to Pearson correlation. It also incorporates the uncertainty (sigma) of the GPR model in the trust region update. If the model is uncertain and agreement is low, the trust region shrinks more aggressively.
*   **Gradient Estimation:** Implements a more stable gradient estimation using a central difference scheme with an adaptive step size. The step size adapts to the length scales of the GPR kernel, providing a more accurate gradient estimate.

These changes aim to provide a more robust and efficient exploration-exploitation trade-off, leading to improved performance across a wider range of black-box optimization problems.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, spearman_correlation
import warnings

class GADETREBO_AdaptiveWeights:
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
        self.exploration_weight = 0.01
        self.n_clusters = 5
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 2.0
        self.model_agreement_threshold = 0.75
        self.kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        self.model = None

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

        # Adaptive weights
        gradient_weight = self.gradient_weight * (1 + np.exp(-self.trust_region_radius))
        diversity_weight = self.diversity_weight * (1 + np.exp(-self.trust_region_radius))
        exploration_weight = self.exploration_weight * (self.trust_region_radius + sigma.mean())

        # Add gradient-based exploration term
        if self.X is not None:
            dmu_dx = self._predict_gradient(X, model)
            gradient_norm = np.linalg.norm(dmu_dx, axis=1, keepdims=True)
            ei = ei + gradient_weight * gradient_norm

        # Add diversity term
        if self.X is not None:
            distances = pairwise_distances(X, self.X)
            min_distances = np.min(distances, axis=1).reshape(-1, 1)
            ei = ei + diversity_weight * min_distances

        # Add exploration regularization term
        exploration_bonus = exploration_weight * sigma * (1 - np.exp(-self.trust_region_radius))
        ei = ei + exploration_bonus

        return ei

    def _predict_gradient(self, X, model):
        # Predict the gradient of the GPR mean function
        # return array of shape (n_points, n_dims)
        X = np.atleast_2d(X)
        dmu_dx = np.zeros((X.shape[0], self.dim))

        # Adaptive step size for gradient estimation
        length_scales = model.kernel_.get_params()['k2__length_scale']
        delta = 1e-6 * length_scales

        # Efficient gradient calculation using central differences
        for i in range(self.dim):
            X_plus = X.copy()
            X_minus = X.copy()
            X_plus[:, i] += delta[i]
            X_minus[:, i] -= delta[i]
            dmu_dx[:, i] = (model.predict(X_plus) - model.predict(X_minus)) / (2 * delta[i])

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

        # Identify diverse regions using clustering
        if self.X is not None:
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.X)
            cluster_centers = kmeans.cluster_centers_
        else:
            cluster_centers = self._sample_points(self.n_clusters)

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
            self._update_eval_points(next_X, next_y)

            # Model agreement check (simplified)
            predicted_y = self.model.predict(next_X)
            agreement, _ = spearman_correlation(next_y.flatten(), predicted_y.flatten())

            # Adjust trust region size
            if np.isnan(agreement) or agreement < self.model_agreement_threshold:
                self.trust_region_radius *= self.trust_region_shrink_factor * (1 + sigma.mean())
            else:
                self.trust_region_radius *= self.trust_region_expand_factor
                self.trust_region_radius = min(self.trust_region_radius, 5.0) # Limit expansion

            # Update trust region center
            trust_region_center = self.X[np.argmin(self.y)]

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<GADETREBO_AdaptiveWeights>", line 8, in <module>
   6 | from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
   7 | from sklearn.cluster import KMeans
   8-> from sklearn.metrics import pairwise_distances, spearman_correlation
   9 | import warnings
  10 | 
ImportError: cannot import name 'spearman_correlation' from 'sklearn.metrics' (/home/hpcl2325/envs/llmbo/lib/python3.10/site-packages/sklearn/metrics/__init__.py)
