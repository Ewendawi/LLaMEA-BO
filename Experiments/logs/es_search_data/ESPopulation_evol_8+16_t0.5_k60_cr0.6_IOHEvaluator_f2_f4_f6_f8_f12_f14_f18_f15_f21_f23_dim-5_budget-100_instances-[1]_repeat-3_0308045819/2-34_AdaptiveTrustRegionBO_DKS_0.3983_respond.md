# Description
**AdaptiveTrustRegionBO with Dynamic Kappa and Improved Sampling (ATBO-DKS):** This algorithm builds upon the Adaptive Trust Region Bayesian Optimization (ATBO) framework by introducing a dynamic kappa parameter in the Lower Confidence Bound (LCB) acquisition function and refining the trust region sampling strategy. The kappa parameter adapts based on the optimization progress, prioritizing exploration in early stages and exploitation later. The sampling strategy within the trust region is enhanced by incorporating a Sobol sequence to generate more diverse and evenly distributed candidate points. This aims to improve the exploration-exploitation balance and avoid premature convergence, leading to better performance on the BBOB test suite.

# Justification
The key components are justified as follows:

1.  **Dynamic Kappa:** A fixed kappa value in LCB can lead to suboptimal exploration-exploitation balance. Adapting kappa allows the algorithm to explore more in the beginning when the GP model is uncertain and exploit more as the model becomes more accurate. The specific adaptation rule is chosen to be simple and computationally efficient while still providing a reasonable dynamic adjustment.

2.  **Sobol Sequence Sampling:** The original implementation uses uniform random sampling within the trust region. Sobol sequences offer better space-filling properties compared to uniform random sampling, especially in higher dimensions. This ensures that the candidate points are more diverse and cover the trust region more effectively, leading to better exploration of the local landscape.

3.  **Computational Efficiency:** The changes are designed to be computationally efficient. The dynamic kappa update is a simple calculation, and Sobol sequence generation is relatively fast. This ensures that the algorithm remains competitive in terms of runtime.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

class AdaptiveTrustRegionBO_DKS:
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
        self.kappa = 2.0  # Initial exploration-exploitation trade-off
        self.kappa_decay = 0.99  # Decay rate for kappa
        self.kappa_min = 0.1  # Minimum value for kappa
        self.sobol_engine = qmc.Sobol(d=self.dim, scramble=True)

    def _sample_points(self, n_points):
        # sample points within the trust region or randomly
        # return array of shape (n_points, n_dims)
        if self.best_x is None or np.random.rand() < self.random_restart_prob:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            # Use Sobol sequence for sampling within the trust region
            sample = self.sobol_engine.random(n=n_points)
            scaled_sample = qmc.scale(sample, -1.0, 1.0)  # Scale to [-1, 1]
            points = self.best_x + self.trust_region_radius * scaled_sample
            # Clip to bounds
            points = np.clip(points, self.bounds[0], self.bounds[1])
            return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
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
            #sigma = np.clip(sigma, 1e-9, np.inf)
            LCB = mu - self.kappa * sigma #kappa is exploration-exploitation trade-off
            #LCB = mu - 2.0 * sigma  # Using a fixed kappa for simplicity
            return LCB.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        candidates = self._sample_points(batch_size)
        acquisition_values = self._acquisition_function(candidates)
        best_index = np.argmin(acquisition_values)
        return candidates[best_index].reshape(1, -1)

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

    def _update_kappa(self):
        # Decay kappa value
        self.kappa = max(self.kappa * self.kappa_decay, self.kappa_min)

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
        batch_size = min(1, self.dim)
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

            # Update kappa
            self._update_kappa()

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AdaptiveTrustRegionBO_DKS got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1714 with standard deviation 0.1066.

took 6.37 seconds to run.