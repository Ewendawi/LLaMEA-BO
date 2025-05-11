You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- ATRBO_DKAISCA_LCB: 0.2012, 665.30 seconds, **ATRBO_DKAISCA_LCB**: Adaptive Trust Region Bayesian Optimization with Dynamic Kappa, Adaptive Initial Exploration, Stochastic Radius, Center Adaptation with Success-rate-based Radius Modification, and Lower Confidence Bound Enhancement. This algorithm refines ATRBO_DKAISCA by incorporating a more robust kappa adaptation strategy based on both GP uncertainty and success history, and introduces a mechanism to dynamically adjust the LCB acquisition function's exploration-exploitation trade-off based on the success rate within the trust region. The stochastic radius expansion is also modified to be adaptive to the dimensionality of the problem.


- ATRBO_DKAICSA_Enhanced: 0.1951, 1663.71 seconds, **ATRBO_DKAICSA_Enhanced**: Adaptive Trust Region Bayesian Optimization with Dynamic Kappa, Adaptive Initial Exploration (Latin Hypercube), Trust Region Center Adaptation, Stochastic Radius Adjustment, and Enhanced GP Fitting. This algorithm combines the strengths of ATRBO_DKAICSA and ATRBO_DKAIS_TRCA, incorporating trust region center adaptation, dynamic kappa based on GP uncertainty and success history, adaptive initial exploration using Latin Hypercube sampling, stochastic radius adjustment to escape local optima, and adaptive rho based on success history. It also enhances the GP fitting by using a more robust kernel and optimizing hyperparameters more thoroughly. The initial exploration is enhanced with Latin Hypercube sampling and a focus on the center of the search space.


- ATRBO_DKAISCA_EI: 0.1950, 219.34 seconds, **ATRBO_DKAISCA_EI**: Adaptive Trust Region Bayesian Optimization with Dynamic Kappa, Adaptive Initial Exploration, Stochastic Radius, Center Adaptation with Success-rate-based Radius Modification, and Expected Improvement. This algorithm enhances ATRBO_DKAISCA by replacing the Lower Confidence Bound (LCB) acquisition function with Expected Improvement (EI). EI is less sensitive to the kappa parameter and can lead to better exploration-exploitation balance, especially in complex search spaces. Furthermore, a dynamic adjustment of the exploration-exploitation balance is introduced based on the uncertainty of the Gaussian Process model, and also the trust region radius is clipped to prevent excessive shrinking.


- ATRBO_HDKRAES: 0.1901, 166.62 seconds, **ATRBO-HDKRAES: Adaptive Trust Region Bayesian Optimization with Hybrid Kappa, Dynamic Kappa Radius Adjustment, Enhanced Exploration, and Success-rate-based Radius Scaling.** This algorithm combines the strengths of ATRBO_DKAISCA and ATRBO_HKRAE. It incorporates hybrid kappa adaptation (success/failure ratio and GP variance), dynamic radius adjustment based on success/failure, enhanced exploration by sampling from a wider region with a certain probability, and success-rate-based radius scaling for more aggressive expansion when the search is consistently improving. The initial exploration is enhanced by using Latin Hypercube sampling. Furthermore, we introduce a mechanism to prevent premature shrinkage of the trust region by setting a minimum radius relative to the initial radius.




The selected solutions to update are:
## ATRBO_DKAISCA_LCB
**ATRBO_DKAISCA_LCB**: Adaptive Trust Region Bayesian Optimization with Dynamic Kappa, Adaptive Initial Exploration, Stochastic Radius, Center Adaptation with Success-rate-based Radius Modification, and Lower Confidence Bound Enhancement. This algorithm refines ATRBO_DKAISCA by incorporating a more robust kappa adaptation strategy based on both GP uncertainty and success history, and introduces a mechanism to dynamically adjust the LCB acquisition function's exploration-exploitation trade-off based on the success rate within the trust region. The stochastic radius expansion is also modified to be adaptive to the dimensionality of the problem.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ATRBO_DKAISCA_LCB:
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
        self.success_rate_threshold = 0.7 # Threshold for aggressive expansion
        self.kappa_min = 0.1
        self.kappa_max = 10.0
        self.success_rate_window = 10
        self.tr_center_adaptation_frequency = 5
        self.iteration = 0

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, center=None, radius=None):
        # sample points within the trust region
        if center is None:
            center = (self.bounds[1] + self.bounds[0]) / 2
        if radius is None:
            radius = np.max(self.bounds[1] - self.bounds[0]) / 2

        sampler = qmc.LatinHypercube(d=self.dim, seed=42)
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

        # Adjust kappa based on success rate
        if len(self.success_history) > self.success_rate_window:
            success_rate = np.mean(self.success_history[-self.success_rate_window:])
        else:
            success_rate = 0.5 # Initial guess

        kappa_adjusted = self.kappa * (1.0 - 0.5 * success_rate) # Reduce kappa if success rate is high

        return mu - kappa_adjusted * sigma

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
        n_uniform = self.n_init // 3
        n_around_best = self.n_init // 3
        n_lhs = self.n_init - n_uniform - n_around_best

        initial_X_uniform = self._sample_points(n_uniform)
        initial_X_best = self._sample_points(n_around_best, center=self.bounds[1]/2, radius=np.max(self.bounds[1] - self.bounds[0]) / 4) # Sampling around the middle of the search space as initial guess
        
        sampler = qmc.LatinHypercube(d=self.dim, seed=42)
        initial_X_lhs = sampler.random(n=n_lhs)
        initial_X_lhs = qmc.scale(initial_X_lhs, self.bounds[0][0], self.bounds[1][0])

        initial_X = np.vstack((initial_X_uniform, initial_X_best, initial_X_lhs))
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
            mu, sigma = gp.predict(next_x.reshape(1, -1), return_std=True)
            sigma = sigma[0]

            if len(self.success_history) > 0:
                last_success = self.success_history[-1]
            else:
                last_success = False

            if next_y < self.best_y:
                self.trust_region_radius /= self.rho  # Expand
                self.kappa *= np.clip(self.rho * 0.9 * (1 - sigma), 0.1, 1.0) # Reduced kappa decrease, also consider GP's uncertainty
                self.success_history.append(True)
                self.best_x = next_x.copy() # global best, keep it.
            else:
                self.trust_region_radius *= self.rho  # Shrink
                # Stochastic expansion
                self.trust_region_radius *= (1 + np.random.normal(0, 0.05) * np.sqrt(self.dim))
                self.kappa /= np.clip(self.rho * 0.9 * (1 - sigma), 0.1, 1.0) # increase kappa more when unsuccessful
                self.success_history.append(False)

            # Trust region center adaptation: adapt the trust region to the best point within the region
            if self.iteration % self.tr_center_adaptation_frequency == 0:
                samples_in_tr = self._sample_points(n_points=100, center=self.best_x, radius=self.trust_region_radius)
                mu_tr, _ = gp.predict(samples_in_tr, return_std=True)
                best_index_tr = np.argmin(mu_tr)
                self.best_x = samples_in_tr[best_index_tr].copy()

            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)
            self.kappa = np.clip(self.kappa, self.kappa_min, self.kappa_max)

            # Adjust rho based on success history
            if len(self.success_history) > self.success_rate_window:
                success_rate = np.mean(self.success_history[-self.success_rate_window:])
                self.rho = 0.9 + 0.09 * success_rate #adaptive rho. Higher success rate leads to higher rho, and thus slower shrinking.

                # Aggressive expansion if success rate is high
                if success_rate > self.success_rate_threshold:
                    self.trust_region_radius /= (self.rho * (1 + (success_rate - self.success_rate_threshold)))

                # Adapt TR center adaptation frequency
                self.tr_center_adaptation_frequency = max(1, int(5 * (1 - success_rate)))

            self.iteration += 1

        return self.best_y, self.best_x

```
The algorithm ATRBO_DKAISCA_LCB got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.2012 with standard deviation 0.0978.

took 665.30 seconds to run.

## ATRBO_DKAISCA_EI
**ATRBO_DKAISCA_EI**: Adaptive Trust Region Bayesian Optimization with Dynamic Kappa, Adaptive Initial Exploration, Stochastic Radius, Center Adaptation with Success-rate-based Radius Modification, and Expected Improvement. This algorithm enhances ATRBO_DKAISCA by replacing the Lower Confidence Bound (LCB) acquisition function with Expected Improvement (EI). EI is less sensitive to the kappa parameter and can lead to better exploration-exploitation balance, especially in complex search spaces. Furthermore, a dynamic adjustment of the exploration-exploitation balance is introduced based on the uncertainty of the Gaussian Process model, and also the trust region radius is clipped to prevent excessive shrinking.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ATRBO_DKAISCA_EI:
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
        self.kappa = 2.0  # Exploration-exploitation trade-off for EI
        self.success_history = [] # Keep track of successful moves
        self.success_rate_threshold = 0.7 # Threshold for aggressive expansion
        self.min_trust_region_radius = 0.1 # Minimum trust region radius


        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, center=None, radius=None):
        # sample points within the trust region
        if center is None:
            center = (self.bounds[1] + self.bounds[0]) / 2
        if radius is None:
            radius = np.max(self.bounds[1] - self.bounds[0]) / 2

        sampler = qmc.LatinHypercube(d=self.dim, seed=42)
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
        # Implement Expected Improvement acquisition function
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Ensure sigma is non-zero to avoid division by zero
        sigma = np.maximum(sigma, 1e-6)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Adjust exploration-exploitation based on GP uncertainty
        exploration_factor = 1.0 + np.mean(sigma)
        ei = ei * exploration_factor

        return -ei  # We want to maximize EI, but minimize the acquisition function

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
                self.best_x = next_x.copy() # global best, keep it.
            else:
                self.trust_region_radius *= self.rho  # Shrink
                # Stochastic expansion
                self.trust_region_radius *= (1 + np.random.normal(0, 0.05) * np.sqrt(self.dim))
                self.kappa /= (self.rho*0.9) # increase kappa more when unsuccessful
                self.success_history.append(False)
                
            # Trust region center adaptation: adapt the trust region to the best point within the region
            samples_in_tr = self._sample_points(n_points=100, center=self.best_x, radius=self.trust_region_radius)
            mu_tr, _ = gp.predict(samples_in_tr, return_std=True)
            best_index_tr = np.argmin(mu_tr)
            self.best_x = samples_in_tr[best_index_tr].copy()

            self.trust_region_radius = np.clip(self.trust_region_radius, self.min_trust_region_radius, np.max(self.bounds[1] - self.bounds[0]) / 2)
            self.kappa = np.clip(self.kappa, 0.1, 10.0)
            
            # Adjust rho based on success history
            if len(self.success_history) > 10:
                success_rate = np.mean(self.success_history[-10:])
                self.rho = 0.9 + 0.09 * success_rate #adaptive rho. Higher success rate leads to higher rho, and thus slower shrinking.

                # Aggressive expansion if success rate is high
                if success_rate > self.success_rate_threshold:
                    self.trust_region_radius /= (self.rho * (1 + (success_rate - self.success_rate_threshold)))

        return self.best_y, self.best_x

```
The algorithm ATRBO_DKAISCA_EI got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1950 with standard deviation 0.1055.

took 219.34 seconds to run.

Combine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it. If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.




Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

