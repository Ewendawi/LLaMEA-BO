# Description
**AEBBO_DKAB_EnhancedPlusPlus:** This algorithm enhances AEBBO_DKAB_EnhancedPlus with an improved local search strategy and a more adaptive exploration-exploitation trade-off. The local search now incorporates a more robust selection of the local search method based on the GP's confidence, using both gradient-based and SLSQP methods with dynamic weighting. The exploration-exploitation trade-off is further refined by incorporating a measure of the diversity of the sampled points, promoting exploration in regions with less dense sampling. A dynamic adjustment of the kernel length scale based on the best function value found so far is also incorporated.

# Justification
The key improvements focus on:

1.  **Enhanced Local Search:** The previous version switched between gradient-based and SLSQP local search based on a hard threshold. This version uses a weighted combination of both methods, allowing for a smoother transition and better exploitation of local information. The weighting is based on the GP's uncertainty.
2.  **Diversity-Aware Exploration:** The exploration weight is now also influenced by the distance to the nearest neighbor in the sampled points. This encourages exploration in less-explored regions of the search space.
3.  **Best Value-Aware Kernel Adaptation:** The kernel length scale is adapted based on the best function value found so far. This allows the GP to focus on promising regions of the search space.
4.  **Adaptive Candidate Set Size:** The number of candidates sampled is now adapted based on the GP's uncertainty. When the GP is uncertain, a larger candidate set is used to promote exploration.

These changes aim to improve the algorithm's ability to balance exploration and exploitation, leading to better performance on the BBOB test suite.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc, norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize, Bounds, approx_fprime
from scipy.spatial.distance import cdist, nearest_neighbors
from scipy.stats import gaussian_kde

class AEBBO_DKAB_EnhancedPlusPlus:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim # number of initial samples

        # Do not add any other arguments without a default value
        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.acq_strategy = "EI+UCB"
        self.initial_exploration_weight = 0.2
        self.exploration_weight = self.initial_exploration_weight
        self.batch_size = 2
        self.kernel_length_scale = 1.0
        self.local_search_restarts = 3
        self.length_scale_weight = 0.5 # Weight for previous length scale in adaptation
        self.gradient_local_search = True # Flag to use gradient-based local search
        self.local_search_weight = 0.5 # Weight for gradient-based local search

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        if self.X is None or len(self.X) < self.dim + 1:
            sampler = qmc.LatinHypercube(d=self.dim)
            samples = sampler.random(n=n_points)
            return qmc.scale(samples, self.bounds[0], self.bounds[1])
        else:
            try:
                kde = gaussian_kde(self.X.T)
                samples = kde.resample(n_points)
                samples = np.clip(samples.T, self.bounds[0], self.bounds[1])
                return samples
            except np.linalg.LinAlgError:
                sampler = qmc.LatinHypercube(d=self.dim)
                samples = sampler.random(n=n_points)
                return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature

        # Dynamic kernel adaptation
        distances = cdist(X, X, metric='euclidean')
        distances = np.triu(distances, k=1)
        distances = distances[distances > 0]
        mean_distance = np.mean(distances) if len(distances) > 0 else 1.0
        _, variances = self.gp.predict(X, return_std=True) if self.gp else (None, np.ones(len(X)))
        mean_variance = np.mean(variances)

        # Best value aware kernel adaptation
        best_value_factor = np.exp(-abs(self.best_y) / 10)

        # Smooth kernel length scale adaptation
        self.kernel_length_scale = self.length_scale_weight * self.kernel_length_scale + (1 - self.length_scale_weight) * mean_distance * (1 + mean_variance) * best_value_factor

        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.kernel_length_scale, length_scale_bounds="fixed")
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.acq_strategy == "EI":
            return self._expected_improvement(X)
        elif self.acq_strategy == "UCB":
            return self._upper_confidence_bound(X)
        elif self.acq_strategy == "EI+UCB":
            ei = self._expected_improvement(X)
            ucb = self._upper_confidence_bound(X)
            return ei + self.exploration_weight * ucb
        else:
            raise ValueError("Invalid acquisition function strategy.")

    def _expected_improvement(self, X):
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-6)  # avoid division by zero
        gamma = (self.best_y - mu) / (sigma + 1e-9) # avoid division by zero
        ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return ei.reshape(-1, 1)

    def _upper_confidence_bound(self, X):
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu.reshape(-1, 1) + 2 * sigma.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        selected_points = []
        # Adaptive candidate set size
        _, mean_sigma = self.gp.predict(self.X, return_std=True) if self.X is not None else (None, [1.0])
        candidate_size = int(100 * self.dim * (1 + np.mean(mean_sigma)))
        candidates = self._sample_points(candidate_size) # Generate a larger candidate set

        for _ in range(batch_size):
            acq_values = self._acquisition_function(candidates)
            best_index = np.argmax(acq_values)
            selected_point = candidates[best_index]

            # Local search to refine the selected point
            def obj(x):
                return -self._acquisition_function(x.reshape(1, -1))[0, 0]  # Negate for minimization

            bounds = Bounds(self.bounds[0], self.bounds[1])
            best_x = None
            best_obj = float('inf')

            # Adaptive local search restarts
            adaptive_restarts = int(self.local_search_restarts * (1 - self.n_evals / self.budget)) + 1
            mu, sigma = self.gp.predict(selected_point.reshape(1, -1), return_std=True)
            sigma = sigma[0]

            # Weighted combination of gradient-based and SLSQP local search
            def obj_grad(x):
                return approx_fprime(x, lambda x: -self._acquisition_function(x.reshape(1, -1))[0, 0], epsilon=1e-6)

            res_lbfgs = minimize(obj, selected_point, method='L-BFGS-B', jac=obj_grad, bounds=bounds)
            res_slsqp = minimize(obj, selected_point, method='SLSQP', bounds=bounds, options={'maxiter': adaptive_restarts*5})

            # Weighted combination of the results
            weight = 1 - np.clip(sigma, 0, 1)  # Higher confidence (lower sigma) -> higher weight for L-BFGS-B
            combined_fun = weight * res_lbfgs.fun + (1 - weight) * res_slsqp.fun

            if combined_fun < best_obj:
                best_obj = combined_fun
                best_x = weight * res_lbfgs.x + (1 - weight) * res_slsqp.x

            if best_x is not None:
                selected_point = best_x

            selected_points.append(selected_point)

            # Remove the selected point from the candidates to avoid duplicates in the batch
            candidates = np.delete(candidates, best_index, axis=0)

        return np.array(selected_points)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

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
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Optimization loop
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Dynamic batch size adjustment
            mean_sigma = np.mean(self.gp.predict(self.X, return_std=True)[1])
            self.batch_size = max(1, min(self.batch_size + (1 if mean_sigma > 0.5 else -1), 5))
            next_X = self._select_next_points(min(self.batch_size, self.budget - self.n_evals))

            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Adaptive acquisition balancing (Thompson Sampling inspired)
            mean_sigma = np.mean(self.gp.predict(self.X, return_std=True)[1])
            range_y = np.max(self.y) - np.min(self.y) if len(self.y) > 1 and np.max(self.y) != np.min(self.y) else 1.0

            # Diversity-aware exploration
            if self.X is not None and len(self.X) > 1:
                tree = nearest_neighbors.KDTree(self.X)
                distances, _ = tree.query(next_X, k=1)
                diversity_factor = np.mean(distances)
            else:
                diversity_factor = 1.0

            self.exploration_weight = self.initial_exploration_weight * (mean_sigma / range_y) * (1 - self.n_evals / self.budget) * diversity_factor
            self.exploration_weight = np.clip(self.exploration_weight, 0.01, self.initial_exploration_weight)

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<AEBBO_DKAB_EnhancedPlusPlus>", line 7, in <module>
   5 | from sklearn.gaussian_process.kernels import RBF, ConstantKernel
   6 | from scipy.optimize import minimize, Bounds, approx_fprime
   7-> from scipy.spatial.distance import cdist, nearest_neighbors
   8 | from scipy.stats import gaussian_kde
   9 | 
ImportError: cannot import name 'nearest_neighbors' from 'scipy.spatial.distance' (/home/hpcl2325/envs/llmbo/lib/python3.10/site-packages/scipy/spatial/distance.py)
