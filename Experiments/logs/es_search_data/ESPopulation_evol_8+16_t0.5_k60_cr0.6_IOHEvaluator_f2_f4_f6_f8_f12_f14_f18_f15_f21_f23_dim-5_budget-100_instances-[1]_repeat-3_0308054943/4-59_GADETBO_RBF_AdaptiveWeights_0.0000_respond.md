# Description
**GADETBO-RBF-AdaptiveWeights: Gradient-Aware Diversity Enhanced Trust Region Bayesian Optimization with RBF Kernel and Adaptive Weights.** This algorithm refines GADETBO-RBF by introducing adaptive weights for the gradient and diversity terms in the acquisition function. The weights are adjusted based on the trust region radius and the model's uncertainty, promoting exploration when the trust region is large or the model is uncertain, and exploitation when the trust region is small and the model is confident. This adaptive weighting strategy aims to improve the balance between exploration and exploitation, leading to better optimization performance. Additionally, a dynamic batch size is implemented to further enhance exploration-exploitation balance.

# Justification
The key improvements are:

1.  **Adaptive Weights for Gradient and Diversity:** The original GADETBO-RBF uses fixed weights for the gradient and diversity terms. Adapting these weights based on the trust region radius and model uncertainty allows the algorithm to dynamically adjust its exploration-exploitation balance. When the trust region is large or the model has high uncertainty (high sigma), the gradient and diversity weights are increased to encourage exploration. Conversely, when the trust region is small and the model is confident (low sigma), the weights are decreased to favor exploitation. This is achieved using a sigmoid function to map the trust region radius and the average predicted standard deviation to a weight between 0 and 1, which is then used to scale the gradient and diversity weights.

2.  **Dynamic Batch Size:** The batch size is adapted based on the trust region radius. When the trust region is larger, a larger batch size is used to explore more broadly. When the trust region is smaller, a smaller batch size is used to exploit the local region more carefully.

These changes aim to improve the algorithm's ability to efficiently explore the search space and converge to the global optimum.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import warnings
from scipy.stats import spearmanr

class GADETBO_RBF_AdaptiveWeights:
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
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        self.model = None
        self.exploration_factor = 1.0

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
        
        # Adaptive weights for gradient and diversity
        trust_region_weight = self._sigmoid(self.trust_region_radius)
        avg_sigma = np.mean(sigma)
        uncertainty_weight = self._sigmoid(avg_sigma)
        
        adaptive_gradient_weight = self.gradient_weight * (trust_region_weight + uncertainty_weight) / 2
        adaptive_diversity_weight = self.diversity_weight * (trust_region_weight + uncertainty_weight) / 2

        # Add gradient-based exploration term
        if self.X is not None:
            dmu_dx = self._predict_gradient(X, model)
            gradient_norm = np.linalg.norm(dmu_dx, axis=1, keepdims=True)
            ei = ei + adaptive_gradient_weight * gradient_norm

        # Add diversity term
        if self.X is not None:
            distances = pairwise_distances(X, self.X)
            median_distances = np.median(distances, axis=1).reshape(-1, 1)
            ei = ei + adaptive_diversity_weight * median_distances

        return ei

    def _sigmoid(self, x):
        # Sigmoid function to map values to a range between 0 and 1
        return 1 / (1 + np.exp(-x))

    def _predict_gradient(self, X, model):
        # Predict the gradient of the GPR mean function using central differences
        # return array of shape (n_points, n_dims)
        X = np.atleast_2d(X)
        dmu_dx = np.zeros((X.shape[0], self.dim))
        
        # Efficient gradient calculation using central differences
        delta = 1e-6
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
        while self.n_evals < self.budget:
            # Optimization
            self.model = self._fit_model(self.X, self.y)
            if self.model is None:
                print("GP model fitting failed. Returning current best.")
                return self.best_y, self.best_x

            # Dynamic batch size
            batch_size = int(3 + 7 * self._sigmoid(self.trust_region_radius)) # batch_size between 3 and 10

            # select points by acquisition function
            next_X = self._select_next_points(batch_size, trust_region_center)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Model agreement check using Spearman correlation
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
  File "<GADETBO_RBF_AdaptiveWeights>", line 202, in __call__
 202->             next_y = self._evaluate_points(func, next_X)
  File "<GADETBO_RBF_AdaptiveWeights>", line 158, in _evaluate_points
 158->         y = np.array([func(x) for x in X]).reshape(-1, 1)
  File "<GADETBO_RBF_AdaptiveWeights>", line 158, in <listcomp>
 156 |         # func: takes array of shape (n_dims,) and returns np.float64.
 157 |         # return array of shape (n_points, 1)
 158->         y = np.array([func(x) for x in X]).reshape(-1, 1)
 159 |         self.n_evals += len(X)
 160 |         return y
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
