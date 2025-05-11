You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- ATRBO_DKAI: 0.0961, 924.42 seconds, Adaptive Trust Region Bayesian Optimization with Dynamic Kappa and Adaptive Initial Exploration (ATRBO-DKAI) refines ATRBO by incorporating a dynamic adjustment of the exploration-exploitation trade-off parameter (kappa) and an adaptive strategy for initial exploration. The initial number of samples is made adaptive to the dimension of the search space. The kappa is adjusted dynamically based on the success of previous iterations. If the new point improves the best-seen value, kappa is decreased to promote exploitation. Conversely, if the new point does not improve the best-seen value, kappa is increased to encourage exploration. The initial sampling incorporates both uniform sampling of the entire space and sampling around the best point found so far to encourage faster initial convergence. The shrinking/expanding factor `rho` for the trust region is also dynamically adjusted based on the history of successful/unsuccessful moves.


- ATSPBO: 0.0846, 1290.76 seconds, Adaptive Trust Region Stochastic Patch Bayesian Optimization (ATSPBO) combines the strengths of ATRBO and SPBO, addressing their weaknesses. It utilizes a trust region approach from ATRBO to focus the search around promising regions while incorporating stochastic patches from SPBO for efficient exploration in high-dimensional spaces. A key modification is that instead of training the GP on the full space and evaluating the acquisition function only on the patch (which is what SPBO was doing and produced an error), ATSPBO projects both the sampled candidate points *and* the training data to the stochastic patch for GP training and acquisition function evaluation. This resolves the dimensionality mismatch error. The trust region radius and patch size are dynamically adjusted based on the optimization progress and remaining budget. The acquisition function is Lower Confidence Bound (LCB) applied within the selected patch, allowing for balanced exploration and exploitation.


- ATRBO_DKRA: 0.0696, 315.82 seconds, Adaptive Trust Region Bayesian Optimization with Dynamic Kappa and Radius Adjustment (ATRBO-DKRA) builds upon the ATRBO algorithm by introducing a more sophisticated method for adjusting the exploration-exploitation trade-off (kappa) and the trust region radius. Instead of a fixed shrinking factor (rho), ATRBO-DKRA dynamically adjusts these parameters based on the recent history of successful and unsuccessful steps. This allows for a more responsive adaptation to the local landscape of the optimization problem. Furthermore, it introduces a minimum number of evaluations before adjusting trust region radius and kappa to avoid premature convergence, which also speeds up the run time.


- ATRBO_DKAR: 0.0688, 335.10 seconds, Adaptive Trust Region Bayesian Optimization with Dynamic Kappa and Adaptive Rho (ATRBO-DKAR) enhances the original ATRBO by introducing a more responsive and adaptable trust region strategy. Key improvements include:
1.  Dynamic Kappa: Instead of fixed bounds, `kappa` (exploration-exploitation trade-off) now adapts based on the GP's uncertainty within the trust region. If the GP's predicted variance is high, `kappa` increases to encourage exploration. Conversely, low variance leads to exploitation.
2.  Adaptive Rho: The shrinking factor `rho` is also made dynamic. Its adjustment is based on a "success rate" within the trust region. If recent samples have led to improvements, `rho` decreases (slower shrinking) to allow for more focused exploitation. If not, `rho` increases to quickly shrink the trust region and explore elsewhere.
3.  Trust Region Center Adaptation: The trust region center is now adapted after each iteration. If the new evaluation point improves the best objective, the trust region center is set as the evaluation point. This is beneficial when the next evaluation point is far from the current best location and has better performance.
4.  Stochastic Trust Region Radius: Add a stochasticity when updating the trust region radius to avoid premature convergence.




The selected solutions to update are:
## ATRBO_DKAI
Adaptive Trust Region Bayesian Optimization with Dynamic Kappa and Adaptive Initial Exploration (ATRBO-DKAI) refines ATRBO by incorporating a dynamic adjustment of the exploration-exploitation trade-off parameter (kappa) and an adaptive strategy for initial exploration. The initial number of samples is made adaptive to the dimension of the search space. The kappa is adjusted dynamically based on the success of previous iterations. If the new point improves the best-seen value, kappa is decreased to promote exploitation. Conversely, if the new point does not improve the best-seen value, kappa is increased to encourage exploration. The initial sampling incorporates both uniform sampling of the entire space and sampling around the best point found so far to encourage faster initial convergence. The shrinking/expanding factor `rho` for the trust region is also dynamically adjusted based on the history of successful/unsuccessful moves.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ATRBO_DKAI:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0  # the number of function evaluations
        self.n_init = min(20 * dim, self.budget // 5) # increased samples for initial exploration
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.5  # Initial trust region radius
        self.rho = 0.95  # Shrinking factor
        self.kappa = 2.0  # Exploration-exploitation trade-off for LCB
        self.success_history = [] # Keep track of successful moves

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, center=None, radius=None):
        # sample points within the trust region
        if center is None:
            center = (self.bounds[1] + self.bounds[0]) / 2
        if radius is None:
            radius = np.max(self.bounds[1] - self.bounds[0]) / 2

        sampler = qmc.Sobol(d=self.dim, scramble=True)
        points = sampler.random(n=n_points)
        points = qmc.scale(points, -1, 1)  # Scale to [-1, 1]

        # Project points to a hypersphere
        lengths = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / lengths * np.random.uniform(0, 1, size=lengths.shape) ** (1 / self.dim)

        points = points * radius + center

        # Clip to the bounds
        points = np.clip(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp):
        # Implement Lower Confidence Bound acquisition function
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        return mu - self.kappa * sigma

    def _select_next_point(self, gp):
        # Select the next point to evaluate within the trust region
        n_samples = 100 * self.dim
        samples = self._sample_points(n_samples, center=self.best_x, radius=self.trust_region_radius)
        acq_values = self._acquisition_function(samples, gp)
        best_index = np.argmin(acq_values)
        return samples[best_index]

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
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

        # Initial exploration: Combine uniform sampling with sampling around the best seen point
        n_uniform = self.n_init // 2
        n_around_best = self.n_init - n_uniform

        initial_X_uniform = self._sample_points(n_uniform)
        initial_X_best = self._sample_points(n_around_best, center=self.bounds[1]/2, radius=np.max(self.bounds[1] - self.bounds[0]) / 4) # Sampling around the middle of the search space as initial guess

        initial_X = np.vstack((initial_X_uniform, initial_X_best))
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Initialize best_x to a random initial point
        self.best_x = self.X[np.argmin(self.y)].copy()

        while self.n_evals < self.budget:
            # Optimization
            gp = self._fit_model(self.X, self.y)

            # Select next point
            next_x = self._select_next_point(gp)
            next_y = self._evaluate_points(func, next_x.reshape(1, -1))
            self._update_eval_points(next_x.reshape(1, -1), next_y)

            # Adjust trust region radius and kappa
            if next_y < self.best_y:
                self.trust_region_radius /= self.rho  # Expand
                self.kappa *= self.rho * 0.9 # Reduced kappa decrease.
                self.success_history.append(True)
            else:
                self.trust_region_radius *= self.rho  # Shrink
                self.kappa /= (self.rho*0.9) # increase kappa more when unsuccessful
                self.success_history.append(False)

            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)
            self.kappa = np.clip(self.kappa, 0.1, 10.0)
            
            # Adjust rho based on success history
            if len(self.success_history) > 10:
                success_rate = np.mean(self.success_history[-10:])
                self.rho = 0.9 + 0.09 * success_rate #adaptive rho. Higher success rate leads to higher rho, and thus slower shrinking.

        return self.best_y, self.best_x

```
The algorithm ATRBO_DKAI got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.0961 with standard deviation 0.0878.

took 924.42 seconds to run.

## ATRBO_DKAR
Adaptive Trust Region Bayesian Optimization with Dynamic Kappa and Adaptive Rho (ATRBO-DKAR) enhances the original ATRBO by introducing a more responsive and adaptable trust region strategy. Key improvements include:
1.  Dynamic Kappa: Instead of fixed bounds, `kappa` (exploration-exploitation trade-off) now adapts based on the GP's uncertainty within the trust region. If the GP's predicted variance is high, `kappa` increases to encourage exploration. Conversely, low variance leads to exploitation.
2.  Adaptive Rho: The shrinking factor `rho` is also made dynamic. Its adjustment is based on a "success rate" within the trust region. If recent samples have led to improvements, `rho` decreases (slower shrinking) to allow for more focused exploitation. If not, `rho` increases to quickly shrink the trust region and explore elsewhere.
3.  Trust Region Center Adaptation: The trust region center is now adapted after each iteration. If the new evaluation point improves the best objective, the trust region center is set as the evaluation point. This is beneficial when the next evaluation point is far from the current best location and has better performance.
4.  Stochastic Trust Region Radius: Add a stochasticity when updating the trust region radius to avoid premature convergence.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ATRBO_DKAR:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(10 * dim, self.budget // 5)
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.5
        self.rho = 0.95
        self.kappa = 2.0
        self.success_history = []  # Track recent success
        self.success_window = 5  # Window size for success rate calculation
        self.rng = np.random.RandomState(42)  # Consistent random state

    def _sample_points(self, n_points, center=None, radius=None):
        if center is None:
            center = (self.bounds[1] + self.bounds[0]) / 2
        if radius is None:
            radius = np.max(self.bounds[1] - self.bounds[0]) / 2

        sampler = qmc.Sobol(d=self.dim, scramble=True)
        points = sampler.random(n=n_points)
        points = qmc.scale(points, -1, 1)

        lengths = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / lengths * self.rng.uniform(0, 1, size=lengths.shape) ** (1 / self.dim)

        points = points * radius + center
        points = np.clip(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp):
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        return mu - self.kappa * sigma

    def _select_next_point(self, gp):
        n_samples = 100 * self.dim
        samples = self._sample_points(n_samples, center=self.best_x, radius=self.trust_region_radius)
        acq_values = self._acquisition_function(samples, gp)
        best_index = np.argmin(acq_values)
        return samples[best_index]

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        self.best_x = self.X[np.argmin(self.y)].copy()

        while self.n_evals < self.budget:
            # Optimization
            gp = self._fit_model(self.X, self.y)
            next_x = self._select_next_point(gp)
            next_y = self._evaluate_points(func, next_x.reshape(1, -1))
            self._update_eval_points(next_x.reshape(1, -1), next_y)

            # Update success history
            success = next_y < self.best_y
            self.success_history.append(success[0])
            if len(self.success_history) > self.success_window:
                self.success_history = self.success_history[-self.success_window:]

            # Adapt kappa based on GP variance within trust region
            mu, sigma = gp.predict(self._sample_points(100, center=self.best_x, radius=self.trust_region_radius), return_std=True)
            avg_sigma = np.mean(sigma)
            self.kappa = np.clip(self.kappa * (1 + avg_sigma), 0.1, 10.0)

            # Adapt rho based on success rate
            success_rate = np.mean(self.success_history) if self.success_history else 0.5
            self.rho = np.clip(0.9 + (0.5 - success_rate) / 5, 0.7, 0.99)

            # Adjust trust region radius with stochasticity
            if success:
                self.trust_region_radius /= (self.rho + self.rng.normal(0, 0.01))
                self.best_x = next_x.copy()
            else:
                self.trust_region_radius *= (self.rho + self.rng.normal(0, 0.01))

            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)

        return self.best_y, self.best_x

```
The algorithm ATRBO_DKAR got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.0688 with standard deviation 0.0649.

took 335.10 seconds to run.

Combine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it. If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.




Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

