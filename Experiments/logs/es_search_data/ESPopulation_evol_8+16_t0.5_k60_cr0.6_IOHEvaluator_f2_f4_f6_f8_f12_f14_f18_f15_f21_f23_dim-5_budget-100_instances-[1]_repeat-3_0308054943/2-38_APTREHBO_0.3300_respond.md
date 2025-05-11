# Description
**APTR-EHBO: Adaptive Pareto Trust Region with Enhanced Hybrid Bayesian Optimization:** This algorithm synergistically combines the strengths of APTRBO and EHBBORLS. It leverages the adaptive trust region management and Pareto-based acquisition steering from APTRBO, along with the adaptive exploration-refinement strategy and local search of EHBBORLS. The core idea is to use the trust region to focus the search while using Pareto-based acquisition to balance multiple objectives (EI, PI, UCB). Thompson sampling with adaptive candidate selection, informed by the uncertainty of the GP model, is performed within the trust region. Finally, local search refines the selected points.

# Justification
The combination of APTRBO and EHBBORLS aims to improve both exploration and exploitation.

*   **Adaptive Trust Region and Pareto Acquisition (APTRBO):** The trust region adaptively constrains the search space, focusing on promising regions. The Pareto-based acquisition steering balances exploration and exploitation by considering multiple acquisition functions. This approach is effective at navigating complex search spaces.

*   **Adaptive Exploration-Refinement and Local Search (EHBBORLS):** The adaptive Thompson sampling allows for efficient exploration by adjusting the number of candidates based on the uncertainty of the GP model. The local search provides a mechanism to refine the selected points, improving exploitation.

*   **Synergistic Combination:** The trust region from APTRBO guides the Thompson sampling in EHBBORLS, focusing the exploration-refinement process. The Pareto-based acquisition ensures a balanced approach to exploration and exploitation within the trust region. The adaptive candidate selection in Thompson sampling dynamically adjusts the exploration based on the GP model's uncertainty.

*   **Computational Efficiency:** The algorithm is designed to be computationally efficient. The adaptive candidate selection in Thompson sampling reduces the number of candidates when the GP model is confident, saving computational resources. The local search is performed only on the selected points, further reducing the computational cost.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C
from scipy.optimize import minimize

class APTREHBO:
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
        self.acquisition_functions = ['ei', 'pi', 'ucb']
        self.ucb_kappa = 2.0 # Initial value
        self.ucb_kappa_max = 5.0
        self.ucb_kappa_min = 1.0
        self.iteration = 0

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
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
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
                is_efficient[is_efficient] = np.any(points[is_efficient] > c, axis=1) | (points[is_efficient] == c).all(axis=1)
                is_efficient[i] = True  # Keep current point
        return is_efficient

    def _select_next_points(self, batch_size, model, trust_region_center):
        # Select the next points to evaluate
        # Combine Pareto-based acquisition with adaptive Thompson sampling and local search
        
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

        # Adaptive Thompson Sampling on Pareto front
        if len(pareto_points) > 0:
            mu, sigma = model.predict(pareto_points, return_std=True)
            avg_sigma = np.mean(sigma)
            num_candidates = int(len(pareto_points) * (1 + avg_sigma))  # Scale candidates by uncertainty
            num_candidates = max(batch_size, min(num_candidates, len(pareto_points)))

            # Sample from the posterior
            sampled_values = np.random.normal(mu, sigma)

            # Select top batch_size candidates
            indices = np.argsort(sampled_values)[:batch_size]
            selected_points = pareto_points[indices]
        else:
            # If no Pareto points, sample randomly within the trust region
            selected_points = self._sample_points(batch_size)
            selected_points = trust_region_center + self.trust_region_radius * (2 * selected_points -1)
            selected_points = np.clip(selected_points, self.bounds[0], self.bounds[1])

        # Local search refinement with SLSQP and adaptive bounds
        refined_points = []
        ls_frac = min(1.0, self.iteration / (self.budget / (2 * batch_size))) # scale local search range
        ls_bounds = ls_frac * (self.bounds[1] - self.bounds[0]) / 2 # adaptive bounds
        for point in selected_points:
            local_bounds = [(max(self.bounds[0][i], point[i] - ls_bounds[i]), min(self.bounds[1][i], point[i] + ls_bounds[i])) for i in range(self.dim)]
            res = minimize(lambda x: model.predict(x.reshape(1, -1))[0], point, bounds=local_bounds, method='SLSQP')
            refined_points.append(res.x)

        return np.array(refined_points)

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
            self.iteration += 1
            # Optimization
            model = self._fit_model(self.X, self.y)
            
            # select points by acquisition function
            next_X = self._select_next_points(batch_size, model, trust_region_center)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Model agreement check (simplified)
            predicted_y = model.predict(next_X)
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
## Error
 Traceback (most recent call last):
  File "<APTREHBO>", line 188, in __call__
 188->             next_y = self._evaluate_points(func, next_X)
  File "<APTREHBO>", line 147, in _evaluate_points
 147->         y = np.array([func(x) for x in X]).reshape(-1, 1)
  File "<APTREHBO>", line 147, in <listcomp>
 145 |         # func: takes array of shape (n_dims,) and returns np.float64.
 146 |         # return array of shape (n_points, 1)
 147->         y = np.array([func(x) for x in X]).reshape(-1, 1)
 148 |         self.n_evals += len(X)
 149 |         return y
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
