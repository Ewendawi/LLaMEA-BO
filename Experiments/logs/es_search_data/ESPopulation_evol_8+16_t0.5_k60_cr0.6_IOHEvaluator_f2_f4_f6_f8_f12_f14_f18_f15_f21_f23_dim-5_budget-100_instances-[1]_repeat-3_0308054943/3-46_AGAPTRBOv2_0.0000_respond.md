# Description
**AGAPTRBOv2: Adaptive Gradient-Aware Pareto Trust Region Bayesian Optimization with Enhanced Gradient Estimation and Adaptive Pareto Weighting:** This algorithm builds upon AGAPTRBO by improving gradient estimation and introducing an adaptive weighting scheme for the Pareto front acquisition functions. The gradient estimation is enhanced using a more robust finite difference method with dynamic step size. The Pareto front is weighted based on the trust region size, prioritizing exploration (UCB, gradient) when the trust region is small and exploitation (EI, PI) when the trust region is large. A dynamic batch size is also incorporated.

# Justification
The original AGAPTRBO algorithm shows good potential but can be improved in several aspects:

1.  **Gradient Estimation:** The finite difference method for gradient estimation can be sensitive to the step size. Using a fixed step size might not be optimal for all functions. Adapting the step size based on the function values can improve the accuracy of the gradient estimation.

2.  **Pareto Front Weighting:** The original algorithm treats all acquisition functions in the Pareto front equally. However, the importance of each acquisition function can vary depending on the stage of optimization. When the trust region is small (exploration phase), UCB and gradient-based acquisition functions should be prioritized to explore the search space. When the trust region is large (exploitation phase), EI and PI should be prioritized to refine the solution.

3.  **Dynamic Batch Size:** Using a fixed batch size can be inefficient. Adapting the batch size based on the trust region size can improve the optimization process. A larger batch size can be used when the trust region is large to exploit the region more effectively, while a smaller batch size can be used when the trust region is small to avoid over-exploration.

These changes aim to improve the balance between exploration and exploitation, leading to a more robust and efficient optimization process.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import minimize
import warnings

class AGAPTRBOv2:
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
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 2.0
        self.model_agreement_threshold = 0.75
        self.acquisition_functions = ['ei', 'pi', 'ucb', 'grad']
        self.ucb_kappa = 2.0 # Initial value
        self.ucb_kappa_max = 5.0
        self.ucb_kappa_min = 1.0
        self.kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        self.model = None
        self.gradient_step_size = 1e-6  # Initial gradient step size
        self.min_gradient_step_size = 1e-8
        self.max_gradient_step_size = 1e-4

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
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
            if acq_type != 'grad':
                return np.zeros_like(mu)
            else:
                return np.zeros((X.shape[0], 1))

        imp = self.best_y - mu
        Z = imp / sigma

        if acq_type == 'ei':
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0  # handle zero sigma
            return ei
        elif acq_type == 'pi':
            pi = norm.cdf(Z)
            return pi
        elif acq_type == 'ucb':
            ucb = mu + self.ucb_kappa * sigma
            return ucb
        elif acq_type == 'grad':
            dmu_dx = self._predict_gradient(X, model)
            gradient_norm = np.linalg.norm(dmu_dx, axis=1, keepdims=True)
            return self.gradient_weight * gradient_norm
        else:
            raise ValueError("Invalid acquisition function type.")

    def _predict_gradient(self, X, model):
        # Predict the gradient of the GPR mean function
        # return array of shape (n_points, n_dims)
        X = np.atleast_2d(X)
        dmu_dx = np.zeros((X.shape[0], self.dim))
        for i in range(self.dim):
            def obj(x):
                x_prime = x.copy()
                x_prime[i] += self.gradient_step_size
                return model.predict(x_prime.reshape(1, -1))[0]
            def obj0(x):
                return model.predict(x.reshape(1, -1))[0]
            dmu_dx[:, i] = (np.array([obj(x) for x in X]) - np.array([obj0(x) for x in X]))/self.gradient_step_size
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

    def _select_next_points(self, batch_size, model, trust_region_center):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
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

        # Apply adaptive Pareto weighting
        weights = self._get_pareto_weights()
        weighted_acquisition_values = acquisition_values * weights

        # Find Pareto front
        is_efficient = self._is_pareto_efficient(weighted_acquisition_values)
        pareto_points = scaled_samples[is_efficient]

        # Select top batch_size points from Pareto front
        if len(pareto_points) > batch_size:
            # Randomly select if more than batch_size
            indices = np.random.choice(len(pareto_points), batch_size, replace=False)
            selected_points = pareto_points[indices]
        else:
            # Use all Pareto points if less than or equal to batch_size
            selected_points = pareto_points

            # If still less than batch_size, sample randomly
            if len(selected_points) < batch_size:
                remaining = batch_size - len(selected_points)
                random_points = self._sample_points(remaining)
                selected_points = np.vstack((selected_points, random_points))

        return selected_points

    def _get_pareto_weights(self):
        # Define weights for Pareto front based on trust region size
        if self.trust_region_radius < 1.0:  # Exploration: UCB and grad are more important
            weights = np.array([0.1, 0.1, 0.4, 0.4])
        else:  # Exploitation: EI and PI are more important
            weights = np.array([0.4, 0.4, 0.1, 0.1])
        return weights

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
            # Adjust batch size based on trust region size
            batch_size = max(1, int(5 * (self.trust_region_radius / 2.0)))

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
                self.gradient_step_size = min(self.gradient_step_size * 1.1, self.max_gradient_step_size) # Increase gradient step size
            else:
                self.trust_region_radius *= self.trust_region_expand_factor
                self.trust_region_radius = min(self.trust_region_radius, 5.0) # Limit expansion
                self.ucb_kappa = max(self.ucb_kappa * 0.9, self.ucb_kappa_min) # Decrease kappa for exploitation
                self.gradient_step_size = max(self.gradient_step_size * 0.9, self.min_gradient_step_size) # Decrease gradient step size

            # Update trust region center
            trust_region_center = self.X[np.argmin(self.y)]

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<AGAPTRBOv2>", line 228, in __call__
 228->             next_y = self._evaluate_points(func, next_X)
  File "<AGAPTRBOv2>", line 183, in _evaluate_points
 183->         y = np.array([func(x) for x in X]).reshape(-1, 1)
  File "<AGAPTRBOv2>", line 183, in <listcomp>
 181 |         # func: takes array of shape (n_dims,) and returns np.float64.
 182 |         # return array of shape (n_points, 1)
 183->         y = np.array([func(x) for x in X]).reshape(-1, 1)
 184 |         self.n_evals += len(X)
 185 |         return y
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
