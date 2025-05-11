# Description
**GADAPTBO: Gradient-Aware Diversity Adaptive Pareto Trust Region Bayesian Optimization:** This algorithm combines the strengths of GADETBO and APTRBO, incorporating gradient information, diversity enhancement, and Pareto-based multi-acquisition steering within an adaptive trust region framework. It uses a Gaussian Process Regression (GPR) model with a Matérn kernel for surrogate modeling. The acquisition function is a Pareto front of Expected Improvement (EI), Upper Confidence Bound (UCB), a gradient-based exploration term, and a distance-based diversity term, evaluated within an adaptive trust region. The size of the trust region is adaptively adjusted based on the agreement between the GPR model and the true objective function. The UCB's kappa parameter is dynamically adjusted based on the trust region size to balance exploration and exploitation. Instead of K-means clustering, a more efficient farthest point sampling is used for diversity.

# Justification
This algorithm builds upon the existing GADETBO and APTRBO algorithms by integrating their key strengths.

*   **Gradient-Awareness:** Incorporating gradient information, as in GADETBO, helps to guide the search towards promising regions, especially in high-dimensional spaces.
*   **Diversity Enhancement:** The diversity term from GADETBO helps to prevent premature convergence and encourages exploration of different regions of the search space. Farthest point sampling provides a computationally efficient way to achieve diversity.
*   **Pareto-Based Acquisition Steering:** Using a Pareto front of multiple acquisition functions (EI, UCB) as in APTRBO allows for a more robust and balanced exploration-exploitation trade-off.
*   **Adaptive Trust Region:** The adaptive trust region mechanism helps to focus the search on promising regions while also allowing for exploration when the model is uncertain or inaccurate.
*   **Dynamic UCB Kappa:** Adjusting the UCB kappa parameter based on the trust region size provides a dynamic way to balance exploration and exploitation. When the trust region is small (high confidence), a smaller kappa encourages exploitation. When the trust region is large (low confidence), a larger kappa encourages exploration.
*   **Computational Efficiency:** The use of farthest point sampling instead of k-means clustering improves the computational efficiency of the algorithm, especially in high-dimensional spaces.

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

class GADAPTBO:
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
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 2.0
        self.model_agreement_threshold = 0.75
        self.kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        self.model = None
        self.acquisition_functions = ['ei', 'ucb', 'gradient', 'diversity']
        self.ucb_kappa = 2.0 # Initial value
        self.ucb_kappa_max = 5.0
        self.ucb_kappa_min = 1.0

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

    def _acquisition_function(self, X, model, acq_type='ei'):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if self.best_y is None:
            if acq_type == 'ei' or acq_type == 'ucb':
                return np.zeros_like(mu)
            elif acq_type == 'gradient':
                dmu_dx = self._predict_gradient(X, model)
                return np.linalg.norm(dmu_dx, axis=1, keepdims=True)
            elif acq_type == 'diversity':
                if self.X is None:
                    return np.zeros_like(mu)
                distances = pairwise_distances(X, self.X)
                return np.min(distances, axis=1).reshape(-1, 1)
            else:
                raise ValueError("Invalid acquisition function type.")

        imp = self.best_y - mu
        Z = imp / sigma

        if acq_type == 'ei':
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0  # handle zero sigma
            return ei
        elif acq_type == 'ucb':
            ucb = mu + self.ucb_kappa * sigma
            return ucb
        elif acq_type == 'gradient':
            dmu_dx = self._predict_gradient(X, model)
            return np.linalg.norm(dmu_dx, axis=1, keepdims=True) * self.gradient_weight
        elif acq_type == 'diversity':
            if self.X is None:
                return np.zeros_like(mu)
            distances = pairwise_distances(X, self.X)
            return np.min(distances, axis=1).reshape(-1, 1) * self.diversity_weight
        else:
            raise ValueError("Invalid acquisition function type.")

    def _predict_gradient(self, X, model):
        # Predict the gradient of the GPR mean function
        # return array of shape (n_points, n_dims)
        X = np.atleast_2d(X)
        dmu_dx = np.zeros((X.shape[0], self.dim))
        
        # Efficient gradient calculation using finite differences
        delta = 1e-6
        for i in range(self.dim):
            X_plus = X.copy()
            X_plus[:, i] += delta
            dmu_dx[:, i] = (model.predict(X_plus) - model.predict(X)) / delta
        
        return dmu_dx

    def _is_pareto_efficient(self, points):
        """
        Find the pareto-efficient points
        :param points: An n by m matrix of points
        :return: A boolean array of length n, True for pareto-efficient points, False otherwise
        """
        is_efficient = np.ones(points.shape[0], dtype=bool)
        for i, c in enumerate(points):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(points[is_efficient] > c, axis=1) | (points[is_efficient] == c).all(axis=1)
                is_efficient[i] = True  # Keep current point
        return is_efficient

    def _select_farthest_points(self, X, n_points):
        """
        Selects n_points from X using farthest point sampling.
        """
        if X.shape[0] <= n_points:
            return X

        selected_indices = [np.random.randint(X.shape[0])]
        remaining_indices = list(range(X.shape[0]))
        remaining_indices.remove(selected_indices[0])

        for _ in range(n_points - 1):
            distances = pairwise_distances(X[selected_indices], X[remaining_indices])
            min_distances = np.min(distances, axis=0)
            farthest_point_index = remaining_indices[np.argmax(min_distances)]
            selected_indices.append(farthest_point_index)
            remaining_indices.remove(farthest_point_index)

        return X[selected_indices]

    def _select_next_points(self, batch_size, model, trust_region_center):
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

        acquisition_values = np.zeros((scaled_samples.shape[0], len(self.acquisition_functions)))

        for i, acq_type in enumerate(self.acquisition_functions):
            acquisition_values[:, i] = self._acquisition_function(scaled_samples, model, acq_type).flatten()

        # Find Pareto front
        is_efficient = self._is_pareto_efficient(acquisition_values)
        pareto_points = scaled_samples[is_efficient]

        # Select top batch_size points from Pareto front
        if len(pareto_points) > batch_size:
            # Select farthest points from Pareto front
            selected_points = self._select_farthest_points(pareto_points, batch_size)
        else:
            # Use all Pareto points if less than or equal to batch_size
            selected_points = pareto_points

            # If still less than batch_size, sample randomly
            if len(selected_points) < batch_size:
                remaining = batch_size - len(selected_points)
                random_points = self._sample_points(remaining)
                selected_points = np.vstack((selected_points, random_points))

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
            next_X = self._select_next_points(batch_size, self.model, trust_region_center)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Model agreement check (simplified)
            predicted_y = self.model.predict(next_X)
            agreement = np.corrcoef(next_y.flatten(), predicted_y.flatten())[0, 1]

            # Adjust trust region size
            if np.isnan(agreement) or agreement < self.model_agreement_threshold:
                self.trust_region_radius *= self.trust_region_shrink_factor
                self.ucb_kappa = min(self.ucb_kappa * 1.1, self.ucb_kappa_max) # Increase kappa for exploration
            else:
                self.trust_region_radius *= self.trust_region_expand_factor
                self.trust_region_radius = min(self.trust_region_radius, 5.0) # Limit expansion
                self.ucb_kappa = max(self.ucb_kappa * 0.9, self.ucb_kappa_min) # Decrease kappa for exploitation

            # Update trust region center
            trust_region_center = self.X[np.argmin(self.y)]

        return self.best_y, self.best_x
```
## Feedback
 The algorithm GADAPTBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1714 with standard deviation 0.1060.

took 99.12 seconds to run.