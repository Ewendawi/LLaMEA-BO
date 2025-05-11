# Description
**Adaptive Diversity and Trust Region Enhanced Bayesian Optimization (ADTREBO):** This algorithm refines TREDABO by introducing adaptive diversity weighting, dynamic adjustment of the number of clusters based on trust region size, and an enhanced model agreement check using a weighted average of past agreements. It aims to improve the balance between exploration and exploitation within the trust region framework.

# Justification
The key improvements are:

1.  **Adaptive Diversity Weighting:** The `diversity_weight` is now dynamically adjusted based on the trust region radius. When the trust region is small (exploitation), the diversity weight is reduced to focus on local refinement. When the trust region is large (exploration), the diversity weight is increased to encourage broader exploration.

2.  **Dynamic Number of Clusters:** The number of clusters used for diversity enhancement is now adjusted based on the trust region radius. A larger trust region implies a larger search space, thus requiring more clusters to effectively explore diverse regions.

3.  **Enhanced Model Agreement Check:** The model agreement check is improved by using a weighted average of past agreements. This provides a more robust and stable assessment of model quality, reducing the risk of premature trust region shrinkage or expansion based on a single noisy observation. Exponential Moving Average is used to give more weight to recent agreements.

4. **Initial Exploration Improvement**: Instead of using a fixed Sobol sequence for initial exploration, the initial samples are now generated using Latin Hypercube Sampling (LHS). LHS provides better space-filling properties compared to Sobol, especially for higher dimensions, leading to a more representative initial dataset.

These changes aim to address the limitations of the original TREDABO algorithm by providing a more adaptive and robust approach to balancing exploration and exploitation within the trust region framework.

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
from scipy.stats import qmc

class ADTREBO:
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
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 2.0
        self.model_agreement_threshold = 0.75
        self.best_x = None
        self.best_y = float('inf')
        self.ucb_kappa = 2.0 # Initial value
        self.ucb_kappa_max = 5.0
        self.ucb_kappa_min = 1.0
        self.diversity_weight = 0.1
        self.n_clusters = 5
        self.agreement_history = []
        self.agreement_ema_alpha = 0.1 # Exponential Moving Average alpha

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points using Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=42)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X, model):
        # Implement acquisition function
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

        # Add diversity term
        if self.X is not None:
            distances = pairwise_distances(X, self.X)
            min_distances = np.min(distances, axis=1).reshape(-1, 1)
            ei = ei + self.diversity_weight * min_distances

        return ei

    def _select_next_points(self, batch_size, model, trust_region_center):
        # Select the next points to evaluate

        # Sample within the trust region using Sobol sequence
        sobol_engine = qmc.Sobol(d=self.dim, seed=42)
        samples = sobol_engine.random(n=100 * batch_size)

        # Scale and shift samples to be within the trust region
        scaled_samples = trust_region_center + self.trust_region_radius * (2 * samples - 1)

        # Clip samples to stay within the problem bounds
        scaled_samples = np.clip(scaled_samples, self.bounds[0], self.bounds[1])

        # Identify diverse regions using clustering
        n_clusters = max(2, min(int(self.trust_region_radius * 2), 10)) # Dynamic n_clusters
        if self.X is not None:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.X)
            cluster_centers = kmeans.cluster_centers_
        else:
            cluster_centers = self._sample_points(n_clusters)

        # Sample candidate points from each cluster
        candidate_points = []
        for center in cluster_centers:
            # Sample points around the cluster center
            candidates = np.random.normal(loc=center, scale=0.5, size=(100, self.dim))
            candidates = np.clip(candidates, self.bounds[0], self.bounds[1])
            candidate_points.extend(candidates)
        candidate_points = np.array(candidate_points)

        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_points, model)

        # Select top batch_size points based on acquisition function values
        indices = np.argsort(-acquisition_values.flatten())[:batch_size]
        selected_points = candidate_points[indices]

        return selected_points

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
        
        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        
        trust_region_center = self.X[np.argmin(self.y)] # Initialize trust region center

        batch_size = 5
        while self.n_evals < self.budget:
            # Optimization
            model = self._fit_model(self.X, self.y)
            
            # select points by acquisition function
            next_X = self._select_next_points(batch_size, model, trust_region_center)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Model agreement check (EMA of agreement)
            predicted_y = model.predict(next_X)
            agreement = np.corrcoef(next_y.flatten(), predicted_y.flatten())[0, 1]
            if np.isnan(agreement):
                agreement = 0.0 # Handle NaN agreement

            if not self.agreement_history:
                agreement_ema = agreement
            else:
                agreement_ema = self.agreement_ema_alpha * agreement + (1 - self.agreement_ema_alpha) * self.agreement_history[-1]

            self.agreement_history.append(agreement_ema)

            # Adjust trust region size
            if agreement_ema < self.model_agreement_threshold:
                self.trust_region_radius *= self.trust_region_shrink_factor
                self.ucb_kappa = min(self.ucb_kappa * 1.1, self.ucb_kappa_max) # Increase kappa for exploration
                self.diversity_weight = min(self.diversity_weight * 1.1, 0.5) # Increase diversity weight

            else:
                self.trust_region_radius *= self.trust_region_expand_factor
                self.trust_region_radius = min(self.trust_region_radius, 5.0) # Limit expansion
                self.ucb_kappa = max(self.ucb_kappa * 0.9, self.ucb_kappa_min) # Decrease kappa for exploitation
                self.diversity_weight = max(self.diversity_weight * 0.9, 0.01) # Decrease diversity weight

            # Update trust region center
            trust_region_center = self.X[np.argmin(self.y)]

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ADTREBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1387 with standard deviation 0.1045.

took 65.35 seconds to run.