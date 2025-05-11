# Description
**GRADEAPARETO_TRBO: Gradient-Enhanced Adaptive Pareto Trust Region Bayesian Optimization with Dynamic Acquisition Balancing and Gradient-based Refinement.** This algorithm integrates gradient information, adaptive regularization, and Pareto-based multi-acquisition function optimization within an adaptive trust region framework, similar to GRADEAPARETOBO and ARPTRBO_v2. It enhances the exploration-exploitation balance by dynamically adjusting acquisition function weights within the Pareto front based on trust region size and model agreement, and refines promising candidate solutions using gradient-based optimization within the trust region. A key feature is the use of both finite difference gradient estimation for the acquisition function and directly incorporating the GPR's gradient prediction into the acquisition function.

# Justification
This algorithm builds upon the strengths of GADETBO and ARPTRBO_v2. GADETBO incorporates gradient information effectively, while ARPTRBO_v2 uses a Pareto front of acquisition functions and adaptive regularization. The combination aims to leverage the benefits of both approaches.

1.  **Gradient-Enhanced Acquisition:** The acquisition function incorporates both EI and a gradient-based term, similar to GADETBO. This encourages exploration in regions with high potential for improvement and large gradients. The gradient is estimated using finite differences for computational efficiency. The GPR gradient prediction is also directly used in the acquisition function.
2.  **Pareto-Based Multi-Acquisition:** The algorithm maintains a Pareto front of multiple acquisition functions (EI, PI, UCB) to balance exploration and exploitation, as in ARPTRBO_v2.
3.  **Adaptive Regularization:** An adaptive regularization term is included in the acquisition function to prevent overfitting and promote exploration in uncertain regions.
4.  **Trust Region Management:** A trust region strategy is used to constrain the search space and improve the reliability of the surrogate model. The trust region size is adaptively adjusted based on the agreement between the model and the true objective function, as in both GADETBO and ARPTRBO_v2.
5.  **Gradient-Based Refinement:** After selecting candidate solutions from the Pareto front, a gradient-based optimization step is performed within the trust region to refine these solutions further. This can lead to faster convergence and improved performance.
6.  **Dynamic Acquisition Balancing:** The weights of the acquisition functions on the Pareto front are dynamically adjusted based on the trust region size and model agreement. When the trust region is small and the model agreement is high, more weight is given to exploitation-oriented acquisition functions (e.g., EI). When the trust region is large and the model agreement is low, more weight is given to exploration-oriented acquisition functions (e.g., UCB).
7.  **Spearman Rank Correlation:** The model agreement is checked using Spearman's rank correlation, which is more robust to outliers than Pearson correlation, as used in ARPTRBO_v2.
8.  **Efficient Gradient Calculation:** The gradient is estimated using finite differences, which is computationally efficient.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import warnings

class GRADEAPARETO_TRBO:
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
        self.model_agreement_threshold = 0.5
        self.best_x = None
        self.best_y = float('inf')
        self.acquisition_functions = ['ei', 'pi', 'ucb']
        self.ucb_kappa = 2.0
        self.ucb_kappa_max = 5.0
        self.ucb_kappa_min = 1.0
        self.reg_weight = 0.1
        self.exploration_factor = 0.01
        self.min_trust_region_radius = 0.1
        self.gradient_weight = 0.01
        self.diversity_weight = 0.1
        self.n_clusters = 5
        self.kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)

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

    def _acquisition_function(self, X, model, iteration, acq_type='ei'):
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

            if acq_type == 'ei':
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma <= 1e-6] = 0.0  # handle zero sigma
            elif acq_type == 'pi':
                ei = norm.cdf(Z)
            elif acq_type == 'ucb':
                ei = mu + self.ucb_kappa * sigma
            else:
                raise ValueError("Invalid acquisition function type.")

        # Adaptive Regularization
        reg_weight = self.reg_weight * (1 - (iteration / self.budget))
        regularization_term = -reg_weight * np.linalg.norm(mu / (sigma + 1e-6), axis=1, keepdims=True)**2
        ei = ei + regularization_term + self.exploration_factor * sigma

        # Add gradient-based exploration term
        if self.X is not None:
            dmu_dx = self._predict_gradient(X, model)
            gradient_norm = np.linalg.norm(dmu_dx, axis=1, keepdims=True)
            ei = ei + self.gradient_weight * gradient_norm

        return ei

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

    def _select_next_points(self, batch_size, model, iteration, trust_region_center, agreement):
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
            acquisition_values[:, i] = self._acquisition_function(scaled_samples, model, iteration, acq_type).flatten()

        # Find Pareto front
        is_efficient = self._is_pareto_efficient(acquisition_values)
        pareto_points = scaled_samples[is_efficient]
        pareto_acq_values = acquisition_values[is_efficient]

        # Dynamic Acquisition Balancing
        if len(pareto_points) > 0:
            weights = np.zeros(len(self.acquisition_functions))
            if agreement > self.model_agreement_threshold:
                weights[0] = 0.7  # EI
                weights[1] = 0.2  # PI
                weights[2] = 0.1  # UCB
            else:
                weights[0] = 0.2  # EI
                weights[1] = 0.3  # PI
                weights[2] = 0.5  # UCB

            # Normalize weights
            weights /= np.sum(weights)

            # Weighted Pareto selection
            weighted_acq_values = np.dot(pareto_acq_values, weights)
            indices = np.argsort(-weighted_acq_values)[:batch_size]
            selected_points = pareto_points[indices]

            # Gradient-Based Refinement
            refined_points = []
            for point in selected_points:
                refined_point = self._gradient_refinement(point, model)
                refined_points.append(refined_point)
            refined_points = np.array(refined_points)

        else:
            refined_points = self._sample_points(batch_size)

        return refined_points

    def _gradient_refinement(self, x, model):
        # Refine a point using gradient-based optimization within the trust region
        def obj_func(x):
            return -model.predict(x.reshape(1, -1))[0]  # Negative for maximization

        bounds = [(max(self.bounds[0][i], x[i] - self.trust_region_radius),
                   min(self.bounds[1][i], x[i] + self.trust_region_radius)) for i in range(self.dim)]

        result = minimize(obj_func, x, method='L-BFGS-B', bounds=bounds)
        return result.x

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

            # Model agreement check using Spearman's rank correlation
            predicted_y = self.model.predict(self.X)
            agreement, _ = spearmanr(self.y.flatten(), predicted_y.flatten())

            # select points by acquisition function
            next_X = self._select_next_points(batch_size, self.model, iteration, trust_region_center, agreement)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Adjust trust region size
            if np.isnan(agreement) or agreement < self.model_agreement_threshold:
                self.trust_region_radius *= self.trust_region_shrink_factor
                self.ucb_kappa = min(self.ucb_kappa * 1.1, self.ucb_kappa_max) # Increase kappa for exploration
            else:
                self.trust_region_radius *= self.trust_region_expand_factor
                self.trust_region_radius = min(self.trust_region_radius, 5.0) # Limit expansion
                self.ucb_kappa = max(self.ucb_kappa * 0.9, self.ucb_kappa_min) # Decrease kappa for exploitation

            self.trust_region_radius = max(self.trust_region_radius, self.min_trust_region_radius) # Ensure minimum trust region size

            # Update exploration factor based on trust region size
            self.exploration_factor = 0.01 * (self.trust_region_radius / 5.0)

            # Update trust region center
            trust_region_center = self.X[np.argmin(self.y)]
            iteration += batch_size

        return self.best_y, self.best_x
```
## Feedback
 The algorithm GRADEAPARETO_TRBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1397 with standard deviation 0.0973.

took 434.32 seconds to run.