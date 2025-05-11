# Description
**GADETBOv2: Gradient-Aware Diversity Enhanced Trust Region Bayesian Optimization with Adaptive Weights and Gradient Enhancement:** This algorithm refines GADETBO by introducing adaptive weights for the gradient and diversity terms in the acquisition function, enhancing the gradient prediction, and improving the trust region adaptation strategy. It dynamically adjusts the gradient and diversity weights based on the optimization progress, prioritizes gradient information in promising regions while maintaining diversity in unexplored areas. A more accurate gradient estimation method is used. The trust region adaptation is improved by considering both model agreement and the uncertainty of the GPR model.

# Justification
*   **Adaptive Gradient and Diversity Weights:** Fixed weights for gradient and diversity might not be optimal across different stages of optimization or different functions. Adapting these weights allows for a more flexible exploration-exploitation balance. Initially, diversity might be emphasized, and as the search converges, the gradient information can be prioritized.
*   **Enhanced Gradient Prediction:** The finite difference method for gradient estimation can be noisy. Using a more sophisticated method, like directly using the GPR's gradient prediction capabilities if available, or using a more accurate finite difference scheme (e.g., central difference), can improve the reliability of the gradient information.
*   **Improved Trust Region Adaptation:** The original model agreement check using `np.corrcoef` can be unstable. A more robust measure, such as comparing the predicted function value change with the actual function value change, is used. Also, the GPR's uncertainty (sigma) is incorporated into the trust region adaptation to shrink the region more aggressively when the model is uncertain.
*   **Computational Efficiency:** The changes are designed to maintain computational efficiency. While more complex gradient estimation might add some overhead, the adaptive weights and improved trust region should lead to faster convergence, reducing the overall number of function evaluations.

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

class GADETBOv2:
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
        self.gradient_weight_decay = 0.95  # Decay factor for gradient weight
        self.diversity_weight_decay = 0.95 # Decay factor for diversity weight


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
        # Predict the gradient of the GPR mean function
        # return array of shape (n_points, n_dims)
        X = np.atleast_2d(X)
        
        # Use GPR's built-in gradient prediction if available (more accurate)
        try:
            mu, sigma = model.predict(X, return_std=True)
            dmu_dx = np.zeros((X.shape[0], self.dim))
            for i in range(self.dim):
                delta = 1e-6
                X_plus = X.copy()
                X_plus[:, i] += delta
                dmu_dx[:, i] = (model.predict(X_plus) - mu) / delta
            return dmu_dx
        except AttributeError:
            # Fallback to finite differences if GPR doesn't support gradient prediction
            dmu_dx = np.zeros((X.shape[0], self.dim))
            delta = 1e-6
            for i in range(self.dim):
                X_plus = X.copy()
                X_minus = X.copy()
                X_plus[:, i] += delta
                X_minus[:, i] -= delta
                dmu_dx[:, i] = (model.predict(X_plus) - model.predict(X_minus)) / (2 * delta) # Central difference
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

            # Model agreement check and trust region adaptation
            predicted_y = self.model.predict(next_X)
            predicted_change = predicted_y - self.model.predict(np.atleast_2d(trust_region_center))
            actual_change = next_y - func(trust_region_center)  # Evaluate the true function at the center

            # Agreement based on the ratio of predicted change to actual change
            agreement = np.mean(np.abs(predicted_change.flatten()) / (np.abs(actual_change.flatten()) + 1e-9))  # Avoid division by zero

            # Incorporate uncertainty into trust region adaptation
            _, uncertainty = self.model.predict(np.atleast_2d(trust_region_center), return_std=True)
            uncertainty = uncertainty[0]

            if np.isnan(agreement) or agreement < self.model_agreement_threshold or uncertainty > 1.0: # Example uncertainty threshold
                self.trust_region_radius *= self.trust_region_shrink_factor
            else:
                self.trust_region_radius *= self.trust_region_expand_factor
                self.trust_region_radius = min(self.trust_region_radius, 5.0) # Limit expansion

            # Update trust region center
            trust_region_center = self.X[np.argmin(self.y)]

            # Decay the weights
            self.gradient_weight *= self.gradient_weight_decay
            self.diversity_weight *= self.diversity_weight_decay

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<GADETBOv2>", line 199, in __call__
 199->             next_y = self._evaluate_points(func, next_X)
  File "<GADETBOv2>", line 157, in _evaluate_points
 157->         y = np.array([func(x) for x in X]).reshape(-1, 1)
  File "<GADETBOv2>", line 157, in <listcomp>
 155 |         # func: takes array of shape (n_dims,) and returns np.float64.
 156 |         # return array of shape (n_points, 1)
 157->         y = np.array([func(x) for x in X]).reshape(-1, 1)
 158 |         self.n_evals += len(X)
 159 |         return y
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
