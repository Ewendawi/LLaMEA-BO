# Description
**GADETBO_Pareto: Gradient-Aware Diversity Enhanced Trust Region Bayesian Optimization with Pareto-based Acquisition Balancing and Kernel Selection.** This algorithm enhances GADETBO by introducing a Pareto-based approach to balance multiple acquisition functions and dynamically selecting the kernel based on model agreement. It utilizes both Matern and RBF kernels and chooses the kernel that yields better model agreement within the trust region. The acquisition function is a weighted sum of Expected Improvement (EI), gradient-based exploration, and diversity enhancement, with weights determined by Pareto optimization. The trust region is adaptively adjusted based on model agreement, and clustering is used to identify diverse regions for sampling.

# Justification
The key improvements are:

1.  **Pareto-based Acquisition Balancing:** Instead of fixed weights for EI, gradient, and diversity terms, Pareto optimization is used to find a set of non-dominated weight combinations. This allows for a more adaptive balance between exploration and exploitation, especially in different stages of the optimization process.

2.  **Dynamic Kernel Selection:** Both Matern and RBF kernels are maintained, and the kernel that yields better model agreement (Spearman correlation) within the trust region is selected for the next iteration. This leverages the strengths of both kernels, as Matern is generally better for non-smooth functions, while RBF is better for smooth functions.

3.  **Adaptive Trust Region:** The trust region size is dynamically adjusted based on the model agreement, which helps to focus the search in promising regions while avoiding premature convergence.

4.  **Central Difference Gradient Estimation:** Using central differences for gradient estimation provides a more accurate gradient estimate compared to forward differences, which can improve the gradient-based exploration.

These enhancements address the limitations of GADETBO and GADETBO\_RBF by providing a more robust and adaptive optimization strategy. The Pareto-based acquisition balancing ensures a better trade-off between exploration and exploitation, while the dynamic kernel selection allows the algorithm to adapt to the characteristics of the objective function. The central difference gradient estimation improves the accuracy of the gradient-based exploration.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import warnings
from scipy.stats import spearmanr
import itertools

class GADETBO_Pareto:
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
        self.n_clusters = 5
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 2.0
        self.model_agreement_threshold = 0.75
        self.matern_kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        self.rbf_kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        self.model = None
        self.kernel_type = "matern" # Initial kernel type
        self.acquisition_weights = [0.33, 0.33, 0.34] # EI, Gradient, Diversity. Initial weights

        # Pareto front of acquisition weights
        self.pareto_front = [[0.33, 0.33, 0.34]]

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
            if self.kernel_type == "matern":
                kernel = self.matern_kernel
            else:
                kernel = self.rbf_kernel
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)
            if self.kernel_type == "matern":
                self.matern_kernel = model.kernel_  # Update kernel with optimized parameters
            else:
                self.rbf_kernel = model.kernel_
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
        else:
            gradient_norm = np.zeros_like(ei)

        # Add diversity term
        if self.X is not None:
            distances = pairwise_distances(X, self.X)
            median_distances = np.median(distances, axis=1).reshape(-1, 1)
        else:
            median_distances = np.zeros_like(ei)

        ei_weight, gradient_weight, diversity_weight = self.acquisition_weights
        acquisition = ei_weight * ei + gradient_weight * gradient_norm + diversity_weight * median_distances

        return acquisition

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

    def _update_pareto_front(self, X, y, model):
        # Update the pareto front of acquisition weights
        # Generate a new set of acquisition weights
        num_new_weights = 5
        new_weights = np.random.dirichlet(np.ones(3), size=num_new_weights).tolist()

        # Add current weights
        new_weights.append(self.acquisition_weights)

        # Evaluate the new weights
        for weights in new_weights:
            self.acquisition_weights = weights
            acquisition_values = self._acquisition_function(X, model)
            agreement = self._calculate_model_agreement(X, y, model)
            
            # Add to pareto front if not dominated
            is_dominated = False
            for existing_weights in self.pareto_front:
                self.acquisition_weights = existing_weights
                existing_acquisition_values = self._acquisition_function(X, model)
                existing_agreement = self._calculate_model_agreement(X, y, model)

                if np.mean(existing_acquisition_values) >= np.mean(acquisition_values) and existing_agreement >= agreement:
                    is_dominated = True
                    break
            
            if not is_dominated:
                self.pareto_front.append(weights)
                # Remove dominated weights from pareto front
                self.pareto_front = [w for w in self.pareto_front if not self._is_dominated(w, X, y, model)]
    
    def _is_dominated(self, weights, X, y, model):
        # Check if a set of weights is dominated by any other weights in the pareto front
        self.acquisition_weights = weights
        acquisition_values = self._acquisition_function(X, model)
        agreement = self._calculate_model_agreement(X, y, model)
        
        for existing_weights in self.pareto_front:
            if weights == existing_weights:
                continue
            self.acquisition_weights = existing_weights
            existing_acquisition_values = self._acquisition_function(X, model)
            existing_agreement = self._calculate_model_agreement(X, y, model)
            
            if np.mean(existing_acquisition_values) >= np.mean(acquisition_values) and existing_agreement >= agreement:
                return True
        return False

    def _calculate_model_agreement(self, X, y, model):
        # Calculate model agreement using Spearman correlation
        predicted_y = model.predict(X)
        agreement, _ = spearmanr(y.flatten(), predicted_y.flatten())
        if np.isnan(agreement):
            return -1.0
        return agreement

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

            # Kernel selection
            if self.X is not None and len(self.X) > 5:
                # Fit models with both kernels
                matern_model = GaussianProcessRegressor(kernel=self.matern_kernel, n_restarts_optimizer=5, alpha=1e-5)
                rbf_model = GaussianProcessRegressor(kernel=self.rbf_kernel, n_restarts_optimizer=5, alpha=1e-5)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    matern_model.fit(self.X, self.y)
                    rbf_model.fit(self.X, self.y)

                # Calculate model agreement for both kernels
                matern_agreement = self._calculate_model_agreement(self.X, self.y, matern_model)
                rbf_agreement = self._calculate_model_agreement(self.X, self.y, rbf_model)

                # Select kernel with better agreement
                if matern_agreement >= rbf_agreement:
                    self.kernel_type = "matern"
                    self.model = matern_model
                    self.matern_kernel = matern_model.kernel_
                else:
                    self.kernel_type = "rbf"
                    self.model = rbf_model
                    self.rbf_kernel = rbf_model.kernel_

            # Update Pareto front
            if self.X is not None and len(self.X) > 5:
                self._update_pareto_front(self.X, self.y, self.model)
            
            # Select acquisition weights from Pareto front
            self.acquisition_weights = self.pareto_front[np.random.randint(len(self.pareto_front))]

            # select points by acquisition function
            next_X = self._select_next_points(batch_size, trust_region_center)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Model agreement check using Spearman correlation
            agreement = self._calculate_model_agreement(next_X, next_y, self.model)

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
## Feedback
 The algorithm GADETBO_Pareto got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1461 with standard deviation 0.0995.

took 256.37 seconds to run.