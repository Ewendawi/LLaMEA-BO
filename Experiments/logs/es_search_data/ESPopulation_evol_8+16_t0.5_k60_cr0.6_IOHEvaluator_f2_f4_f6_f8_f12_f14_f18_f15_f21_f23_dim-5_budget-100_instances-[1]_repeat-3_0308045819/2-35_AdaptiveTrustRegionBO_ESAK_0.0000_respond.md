# Description
**AdaptiveTrustRegionBO with Enhanced Sampling and Adaptive Kappa (ATBO-ESAK):** This algorithm builds upon the AdaptiveTrustRegionBO framework by incorporating enhancements to the trust region sampling strategy and the dynamic kappa parameter in the Lower Confidence Bound (LCB) acquisition function. The sampling strategy is refined to promote diversity and exploration within the trust region by using a Sobol sequence instead of uniform sampling and adding a small probability of sampling from the entire search space. The kappa parameter is dynamically adjusted based on both the optimization progress and the estimated uncertainty from the Gaussian Process (GP).

# Justification
1.  **Sobol Sequence Sampling:** Replacing uniform sampling with a Sobol sequence aims to generate more evenly distributed points within the trust region. This can lead to better coverage of the search space and potentially identify promising regions more efficiently than random sampling.
2.  **Global Sampling Probability:** Adding a small probability of sampling from the entire search space introduces a mechanism for escaping local optima and promoting global exploration, especially in the later stages of optimization.
3.  **Kappa Adjustment Based on GP Uncertainty:** The original dynamic kappa parameter only depended on the optimization progress. By incorporating the estimated uncertainty (sigma) from the GP, the algorithm can adapt its exploration-exploitation balance more effectively. Higher uncertainty leads to a larger kappa, favoring exploration, while lower uncertainty leads to a smaller kappa, favoring exploitation. This ensures that the algorithm focuses on reducing uncertainty in unexplored regions and exploits promising regions where the GP is more confident.
4.  **Batch Size:** The batch size is increased to `min(4, self.dim)` to allow for more parallel exploration within the trust region.
5.  **Kernel Tuning:** The length scale bounds of the RBF kernel are set to be adaptive to the dimension of the problem, which helps the GP model to better capture the underlying function's characteristics.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

class AdaptiveTrustRegionBO_ESAK:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(5*dim, self.budget//10)

        # Do not add any other arguments without a default value
        self.gp = None
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.radius_decay = 0.95
        self.radius_increase = 1.1
        self.min_radius = 0.1
        self.success_ratio = 0.0 #ratio to track the success of trust region
        self.random_restart_prob = 0.05 #Probability of random restart
        self.global_sampling_prob = 0.05

    def _sample_points(self, n_points):
        # sample points within the trust region or randomly
        # return array of shape (n_points, n_dims)
        if self.best_x is None or np.random.rand() < self.random_restart_prob:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            points = qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            points = []
            n_trust_region = int(n_points * (1 - self.global_sampling_prob))
            n_global = n_points - n_trust_region

            #Sobol sequence sampling within trust region
            if n_trust_region > 0:
                sampler = qmc.Sobol(d=self.dim, scramble=True)
                sample = sampler.random(n=n_trust_region)
                scaled_sample = 2 * sample - 1  # Scale to [-1, 1]
                trust_region_points = self.best_x + self.trust_region_radius * scaled_sample
                trust_region_points = np.clip(trust_region_points, self.bounds[0], self.bounds[1])
                points.extend(trust_region_points)

            # Random sampling from the entire search space
            if n_global > 0:
                sampler = qmc.LatinHypercube(d=self.dim)
                sample = sampler.random(n=n_global)
                global_points = qmc.scale(sample, self.bounds[0], self.bounds[1])
                points.extend(global_points)

            points = np.array(points)

        return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        length_scale_bounds = (1e-2 * self.dim, 1e2 * self.dim)
        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=length_scale_bounds) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-6)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function: Lower Confidence Bound
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            mu, sigma = self.gp.predict(X, return_std=True)
            sigma = np.clip(sigma, 1e-9, np.inf)

            # Dynamic kappa for exploration-exploitation balance
            kappa = 1.0 + 2.0 * (1.0 - self.n_evals / self.budget) * (1 + np.mean(sigma))  # Reduce kappa over time, increase with uncertainty
            LCB = mu - kappa * sigma
            return LCB.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        candidates = self._sample_points(batch_size * 10) #Over sample
        acquisition_values = self._acquisition_function(candidates)
        best_indices = np.argsort(acquisition_values.flatten())[:batch_size]  # Select top batch_size
        return candidates[best_indices]

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
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]
            self.success_ratio = 1.0 #reset success ratio
        else:
            self.success_ratio *= 0.75 #reduce success ratio if not improving

    def _adjust_trust_region(self):
        # Adjust the trust region size based on the success
        if self.success_ratio > 0.5:
            self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)

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
        batch_size = min(4, self.dim)
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Select points by acquisition function
            next_X = self._select_next_points(batch_size)

            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)

            # Update the evaluated points
            self._update_eval_points(next_X, next_y)

            # Adjust trust region radius
            self._adjust_trust_region()

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<AdaptiveTrustRegionBO_ESAK>", line 155, in __call__
 155->             next_y = self._evaluate_points(func, next_X)
  File "<AdaptiveTrustRegionBO_ESAK>", line 104, in _evaluate_points
 104->         y = np.array([func(x) for x in X]).reshape(-1, 1)
  File "<AdaptiveTrustRegionBO_ESAK>", line 104, in <listcomp>
 102 |         # func: takes array of shape (n_dims,) and returns np.float64.
 103 |         # return array of shape (n_points, 1)
 104->         y = np.array([func(x) for x in X]).reshape(-1, 1)
 105 |         self.n_evals += len(X)
 106 |         return y
llamea.utils.BOOverBudgetException: ('OverBudgetException', 'The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.')
