# Description
**APTRBO-v2: Adaptive Pareto Trust Region Bayesian Optimization with Enhanced Pareto Selection and Dynamic Batch Size:** This algorithm builds upon APTRBO by introducing enhancements to the Pareto front selection process and incorporating a dynamic batch size strategy. The Pareto front selection is improved by using a hypervolume contribution metric to select points that contribute most to the diversity of the Pareto front. Additionally, the batch size is dynamically adjusted based on the trust region radius, allowing for more exploration when the trust region is large and more exploitation when the trust region is small.

# Justification
The key improvements are:

1.  **Hypervolume Contribution-Based Pareto Selection:** Instead of randomly selecting points from the Pareto front, this enhancement prioritizes selecting points that maximize the hypervolume contribution to the Pareto front. This promotes diversity in the selected points and helps to explore different regions of the search space more effectively.

2.  **Dynamic Batch Size:** The batch size is dynamically adjusted based on the trust region radius. This allows the algorithm to adapt to the local landscape of the objective function. When the trust region is large, a larger batch size is used to explore the search space more broadly. When the trust region is small, a smaller batch size is used to exploit the local optimum more effectively. This adaptive strategy can improve the overall performance of the algorithm.

3.  **Gradient-based refinement:** After the initial Pareto selection, a gradient-based method (L-BFGS-B) is used to refine the selected points within the trust region. This allows for more precise optimization and can lead to better solutions.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

class APTRBOv2:
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

    def _calculate_hypervolume_contribution(self, points):
        """
        Calculate the hypervolume contribution of each point in the Pareto front.
        """
        if len(points) <= 2:
            return np.ones(len(points))  # Assign equal contribution if few points

        hull = ConvexHull(points)
        vertices = points[hull.vertices]
        hypervolume = hull.volume

        contributions = np.zeros(len(points))
        for i in range(len(points)):
            temp_points = np.delete(points, i, axis=0)
            if len(temp_points) > 2:
                try:
                    temp_hull = ConvexHull(temp_points)
                    contributions[i] = hypervolume - temp_hull.volume
                except:
                    contributions[i] = 0
            else:
                contributions[i] = 0

        return contributions

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
        pareto_acquisition_values = acquisition_values[is_efficient]

        # Select top batch_size points from Pareto front based on hypervolume contribution
        if len(pareto_points) > 0:
            hypervolume_contributions = self._calculate_hypervolume_contribution(pareto_acquisition_values)
            probabilities = hypervolume_contributions / np.sum(hypervolume_contributions)
            indices = np.random.choice(len(pareto_points), min(batch_size, len(pareto_points)), replace=False, p=probabilities)
            selected_points = pareto_points[indices]

            # Refine selected points using gradient-based optimization
            refined_points = []
            for point in selected_points:
                def objective(x):
                    acq_values = np.zeros(len(self.acquisition_functions))
                    for i, acq_type in enumerate(self.acquisition_functions):
                        acq_values[i] = self._acquisition_function(x.reshape(1, -1), model, acq_type).flatten()[0]
                    return -np.sum(acq_values)  # Minimize the negative sum of acquisition values

                result = minimize(objective, point, method='L-BFGS-B', bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)])
                refined_points.append(result.x)
            selected_points = np.array(refined_points)

        else:
            selected_points = np.array([])

        # If still less than batch_size, sample randomly
        if len(selected_points) < batch_size:
            remaining = batch_size - len(selected_points)
            random_points = self._sample_points(remaining)
            if len(selected_points) > 0:
                selected_points = np.vstack((selected_points, random_points))
            else:
                selected_points = random_points

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
            model = self._fit_model(self.X, self.y)

            # Dynamically adjust batch size
            batch_size = int(min(10, 2 + self.trust_region_radius))

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
  File "<APTRBOv2>", line 221, in __call__
 221->             next_X = self._select_next_points(batch_size, model, trust_region_center)
  File "<APTRBOv2>", line 144, in _select_next_points
 144->             hypervolume_contributions = self._calculate_hypervolume_contribution(pareto_acquisition_values)
  File "<APTRBOv2>", line 97, in _calculate_hypervolume_contribution
  95 |             return np.ones(len(points))  # Assign equal contribution if few points
  96 | 
  97->         hull = ConvexHull(points)
  98 |         vertices = points[hull.vertices]
  99 |         hypervolume = hull.volume
  File "_qhull.pyx", line 2448, in scipy.spatial._qhull.ConvexHull.__init__
  File "_qhull.pyx", line 358, in scipy.spatial._qhull._Qhull.__init__
scipy.spatial._qhull.QhullError: QH6421 qhull internal error (qh_maxsimplex): qh.MAXwidth required for qh_maxsimplex.  Used to estimate determinate

While executing:  | qhull i Qt
Options selected for Qhull 2019.1.r 2019/06/21:
  run-id 2041822294  incidence  Qtriangulate  _pre-merge  _zero-centrum
  _max-width  0  Error-roundoff 5.7e-14  _one-merge 4e-13  _near-inside 2e-12
  Visible-distance 1.1e-13  U-max-coplanar 1.1e-13  Width-outside 2.3e-13
  _wide-facet 6.8e-13  _maxoutside 4.5e-13

A Qhull internal error has occurred.  Please send the input and output to
qhull_bug@qhull.org. If you can duplicate the error with logging ('T4z'), please
include the log file.

