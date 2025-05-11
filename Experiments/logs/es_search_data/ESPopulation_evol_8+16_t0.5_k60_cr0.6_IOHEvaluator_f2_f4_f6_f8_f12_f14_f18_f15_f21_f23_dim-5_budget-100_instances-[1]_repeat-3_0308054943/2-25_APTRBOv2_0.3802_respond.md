# Description
**APTRBO-v2: Adaptive Pareto Trust Region Bayesian Optimization with Enhanced Exploration and Model Tuning:** This algorithm refines APTRBO by incorporating several key improvements. Firstly, it introduces a dynamic adjustment of the batch size based on the trust region radius, promoting exploration with larger batches when the trust region is large and exploitation with smaller batches when the trust region is small. Secondly, it utilizes a weighted Pareto selection, favoring points with higher EI values. Thirdly, it incorporates a more robust model agreement check using a combination of correlation and RMSE, and uses this information to dynamically adjust the trust region shrink/expand factors. Lastly, it includes a mechanism to periodically re-fit the GP model with an increased `n_restarts_optimizer` to further refine the surrogate model.

# Justification
The improvements are justified as follows:

*   **Dynamic Batch Size:** Adjusting the batch size allows for a more efficient allocation of function evaluations. Larger batch sizes in early stages or when the trust region is large promote exploration, while smaller batch sizes in later stages or when the trust region is small focus on exploitation.
*   **Weighted Pareto Selection:** The original APTRBO randomly selects from the Pareto front. Weighting the selection towards EI encourages the algorithm to prioritize points that are not only Pareto-efficient but also have high expected improvement, leading to faster convergence.
*   **Enhanced Model Agreement Check:** Using both correlation and RMSE provides a more comprehensive assessment of model agreement compared to using only correlation. This leads to a more reliable adjustment of the trust region size.
*   **Dynamic Trust Region Factors:** Adjusting the shrink and expand factors based on model agreement provides a finer control over the trust region adaptation.
*   **Periodic Model Refitting:** Periodically refitting the GP model with more restarts allows for a more accurate surrogate model, especially in later stages of the optimization when more data is available.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


class APTRBOv2:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0  # the number of function evaluations
        self.n_init = 2 * dim
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 2.0
        self.model_agreement_threshold = 0.75
        self.best_x = None
        self.best_y = float('inf')
        self.acquisition_functions = ['ei', 'pi', 'ucb']
        self.ucb_kappa = 2.0  # Initial value
        self.ucb_kappa_max = 5.0
        self.ucb_kappa_min = 1.0
        self.model_refit_interval = 10  # Refit the model every n iterations
        self.n_restarts_optimizer = 5  # Initial number of restarts
        self.n_restarts_optimizer_enhanced = 10 # Number of restarts for periodic refitting

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y, enhanced=False):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        n_restarts = self.n_restarts_optimizer_enhanced if enhanced else self.n_restarts_optimizer
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X, model, acq_type='ei'):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if self.best_y is None:
            return np.zeros_like(mu)

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
        else:
            raise ValueError("Invalid acquisition function type.")

    def _is_pareto_efficient(self, points):
        """
        Find the pareto-efficient points
        :param points: An n by m matrix of points
        :return: A boolean array of length n, True for pareto-efficient points, False otherwise
        """
        is_efficient = np.ones(points.shape[0], dtype=bool)
        for i, c in enumerate(points):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(points[is_efficient] > c, axis=1) | (
                            points[is_efficient] == c).all(axis=1)
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

        # Find Pareto front
        is_efficient = self._is_pareto_efficient(acquisition_values)
        pareto_points = scaled_samples[is_efficient]
        pareto_acq_values = acquisition_values[is_efficient]

        # Select top batch_size points from Pareto front, weighted by EI
        if len(pareto_points) > batch_size:
            # Weight by EI
            ei_index = self.acquisition_functions.index('ei')
            ei_values = pareto_acq_values[:, ei_index]
            probabilities = ei_values / np.sum(ei_values)
            indices = np.random.choice(len(pareto_points), batch_size, replace=False, p=probabilities)
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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        trust_region_center = self.X[np.argmin(self.y)]  # Initialize trust region center

        iteration = 0
        while self.n_evals < self.budget:
            # Optimization
            enhanced_refit = (iteration % self.model_refit_interval == 0)
            model = self._fit_model(self.X, self.y, enhanced=enhanced_refit)

            # Dynamic batch size
            batch_size = max(1, int(5 * self.trust_region_radius / 5.0)) # Scale batch size with trust region

            # select points by acquisition function
            next_X = self._select_next_points(batch_size, model, trust_region_center)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Model agreement check
            predicted_y = model.predict(next_X)
            agreement = np.corrcoef(next_y.flatten(), predicted_y.flatten())[0, 1]
            rmse = np.sqrt(mean_squared_error(next_y, predicted_y))

            # Adjust trust region size
            if np.isnan(agreement) or agreement < self.model_agreement_threshold or rmse > 0.5: # Added RMSE check
                self.trust_region_radius *= self.trust_region_shrink_factor
                self.trust_region_shrink_factor = min(0.9, self.trust_region_shrink_factor * 1.1) # Dampen shrinking
                self.ucb_kappa = min(self.ucb_kappa * 1.1, self.ucb_kappa_max)  # Increase kappa for exploration
            else:
                self.trust_region_radius *= self.trust_region_expand_factor
                self.trust_region_radius = min(self.trust_region_radius, 5.0)  # Limit expansion
                self.trust_region_expand_factor = max(1.1, self.trust_region_expand_factor * 0.9) # Dampen expanding
                self.ucb_kappa = max(self.ucb_kappa * 0.9, self.ucb_kappa_min)  # Decrease kappa for exploitation

            # Update trust region center
            trust_region_center = self.X[np.argmin(self.y)]
            
            iteration += 1

        return self.best_y, self.best_x
```
## Feedback
 The algorithm APTRBOv2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1583 with standard deviation 0.1072.

took 513.25 seconds to run.