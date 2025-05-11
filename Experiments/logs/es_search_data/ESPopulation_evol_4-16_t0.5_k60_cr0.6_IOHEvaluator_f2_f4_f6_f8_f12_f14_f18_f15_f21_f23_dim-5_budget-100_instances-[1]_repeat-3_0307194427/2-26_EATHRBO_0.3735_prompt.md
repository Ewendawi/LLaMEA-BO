You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- AdaptiveTrustRegionBO_DR: 0.1808, 106.20 seconds, **AdaptiveTrustRegionBO with Dynamic Radius and EI Improvement (ATRBO-DR)**: This algorithm builds upon the AdaptiveTrustRegionBO by introducing a dynamic radius adjustment strategy based on the Expected Improvement (EI) values. Instead of a fixed decay/growth rate, the radius is adjusted proportionally to the average EI of the evaluated points within the trust region. This allows for more aggressive shrinking when significant improvements are observed and slower expansion when improvements are marginal. Additionally, a mechanism is added to re-center the trust region to the best observed point if the current center is far away from it.


- AdaptiveEnsembleTrustRegionBO: 0.1785, 150.44 seconds, **Adaptive Ensemble Trust Region Bayesian Optimization (AETRBO)**: This algorithm combines the strengths of Adaptive Trust Region BO (ATBO) and Bayesian Ensemble BO (BEBO). It uses an ensemble of Gaussian Process Regression (GPR) models with different kernels to improve robustness and exploration, similar to BEBO. It also incorporates a dynamically adjusted trust region, like ATBO, to balance exploration and exploitation. The trust region's center and radius are adapted based on the ensemble's predictive uncertainty and the observed improvement. An evolutionary strategy is used to select the next points within the trust region, leveraging the acquisition function derived from the ensemble's predictions. This approach aims to efficiently explore the search space while focusing on promising regions, providing a balance between global exploration and local refinement.


- EnhancedEfficientHybridBO: 0.1685, 174.45 seconds, **Enhanced Efficient Hybrid Bayesian Optimization (EEHBBO)**: This algorithm builds upon the EfficientHybridBO by incorporating adaptive kernel lengthscale optimization and a dynamic batch size strategy. It combines Gaussian Process Regression (GPR) with Expected Improvement (EI) for exploration-exploitation balance. It employs a Latin Hypercube Sampling (LHS) for initial exploration. The kernel lengthscale of the GPR model is optimized periodically using L-BFGS-B to better capture the function's characteristics. The batch size is dynamically adjusted based on the uncertainty of the GPR model to balance exploration and exploitation.


- AdaptiveHybridBO_DE: 0.1670, 5.98 seconds, **Adaptive Hybrid Bayesian Optimization with Dynamic Exploration (AHBO-DE)**: This algorithm combines the adaptive trust region approach of AdaptiveTrustRegionBO with the efficient Gaussian process modeling and diversity-promoting selection of EfficientHybridBO. It uses a dynamic exploration strategy that balances trust region exploitation with broader exploration based on the uncertainty of the GP model. It employs a simplified GP model for computational efficiency and incorporates a dynamic adjustment of the exploration-exploitation balance based on the optimization progress.




The selected solutions to update are:
## AdaptiveTrustRegionBO_DR
**AdaptiveTrustRegionBO with Dynamic Radius and EI Improvement (ATRBO-DR)**: This algorithm builds upon the AdaptiveTrustRegionBO by introducing a dynamic radius adjustment strategy based on the Expected Improvement (EI) values. Instead of a fixed decay/growth rate, the radius is adjusted proportionally to the average EI of the evaluated points within the trust region. This allows for more aggressive shrinking when significant improvements are observed and slower expansion when improvements are marginal. Additionally, a mechanism is added to re-center the trust region to the best observed point if the current center is far away from it.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from scipy.optimize import minimize

class AdaptiveTrustRegionBO_DR:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * (dim + 1)
        self.trust_region_center = np.zeros(dim)  # Initialize trust region center
        self.trust_region_radius = 2.5  # Initial trust region radius (half of the search space)
        self.min_radius = 0.1
        self.radius_decay_base = 0.95
        self.radius_grow_base = 1.1
        self.gp = None
        self.ei_scaling = 0.1 # Scaling factor for EI-based radius adjustment
        self.recentering_threshold = 0.5 # Threshold for re-centering trust region

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points within the trust region
        # return array of shape (n_points, n_dims)
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        sample = sampler.random(n=n_points)
        scaled_sample = qmc.scale(sample, -1.0, 1.0) # Scale to [-1, 1]
        
        # Map to trust region
        points = self.trust_region_center + scaled_sample * self.trust_region_radius
        
        # Clip to bounds
        points = np.clip(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * \
                 RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + \
                 WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-7, 1e-3))
        
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function (Expected Improvement)
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None or self.X is None or self.y is None:
            return np.zeros((len(X), 1))  # Return zero if model is not fitted yet

        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)
        y_best = np.min(self.y)
        gamma = (y_best - mu) / sigma
        ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Sample candidate points
        candidates = self._sample_points(100 * self.dim)
        
        # Calculate acquisition function values
        ei = self._acquisition_function(candidates)
        
        # Select top batch_size candidates based on EI
        selected_indices = np.argsort(ei.flatten())[-batch_size:]
        selected_points = candidates[selected_indices]
        
        return selected_points

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)
    
    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        # Do not change the function signature
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        
        while self.n_evals < self.budget:
            # Fit the Gaussian Process model
            gp = self._fit_model(self.X, self.y)
            
            # Select the next points to evaluate
            batch_size = min(self.budget - self.n_evals, 5)
            next_X = self._select_next_points(batch_size)
            
            # Evaluate the points
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Update the best solution
            best_idx = np.argmin(self.y)
            current_best_y = self.y[best_idx][0]
            current_best_x = self.X[best_idx]

            # Calculate average EI of evaluated points
            ei_values = self._acquisition_function(next_X)
            avg_ei = np.mean(ei_values)

            # Adjust trust region radius based on EI
            if current_best_y < best_y:
                # Improvement: shrink radius based on EI
                self.trust_region_radius *= (self.radius_decay_base + self.ei_scaling * avg_ei)
                self.trust_region_center = current_best_x
                best_y = current_best_y
                best_x = current_best_x
            else:
                # No improvement: expand radius, but slower if EI is high
                self.trust_region_radius *= (self.radius_grow_base - self.ei_scaling * avg_ei)
                self.trust_region_radius = min(self.trust_region_radius, 2.5) # Limit to initial radius
            
            self.trust_region_radius = max(self.trust_region_radius, self.min_radius)

            # Re-center trust region if current center is far from best point
            distance = np.linalg.norm(self.trust_region_center - best_x)
            if distance > self.recentering_threshold * self.trust_region_radius:
                self.trust_region_center = best_x

        return best_y, best_x

```
The algorithm AdaptiveTrustRegionBO_DR got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1808 with standard deviation 0.1042.

took 106.20 seconds to run.

## AdaptiveHybridBO_DE
**Adaptive Hybrid Bayesian Optimization with Dynamic Exploration (AHBO-DE)**: This algorithm combines the adaptive trust region approach of AdaptiveTrustRegionBO with the efficient Gaussian process modeling and diversity-promoting selection of EfficientHybridBO. It uses a dynamic exploration strategy that balances trust region exploitation with broader exploration based on the uncertainty of the GP model. It employs a simplified GP model for computational efficiency and incorporates a dynamic adjustment of the exploration-exploitation balance based on the optimization progress.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.spatial.distance import cdist

class AdaptiveHybridBO_DE:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * (dim + 1)
        self.trust_region_center = np.zeros(dim)
        self.trust_region_radius = 2.5
        self.min_radius = 0.1
        self.radius_decay = 0.95
        self.radius_grow = 1.1
        self.exploration_weight = 0.1  # Initial exploration weight
        self.exploration_decay = 0.98 # Decay exploration as optimization progresses

    def _sample_points(self, n_points, use_trust_region=True):
        if use_trust_region:
            sampler = qmc.Sobol(d=self.dim, scramble=False)
            sample = sampler.random(n=n_points)
            scaled_sample = qmc.scale(sample, -1.0, 1.0)
            points = self.trust_region_center + scaled_sample * self.trust_region_radius
            points = np.clip(points, self.bounds[0], self.bounds[1])
        else:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            points = qmc.scale(sample, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        kernel = C(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-5)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp, y_best):
        mu, sigma = gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)
        gamma = (y_best - mu) / sigma
        ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return ei

    def _select_next_points(self, gp, y_best, batch_size):
        # Dynamic exploration-exploitation balance
        num_trust_region = int(batch_size * (1 - self.exploration_weight))
        num_random = batch_size - num_trust_region

        # Sample points from trust region
        if num_trust_region > 0:
            candidates_tr = self._sample_points(100 * self.dim, use_trust_region=True)
            ei_tr = self._acquisition_function(candidates_tr, gp, y_best)
            selected_indices_tr = np.argsort(ei_tr)[-num_trust_region:]
            selected_points_tr = candidates_tr[selected_indices_tr]
        else:
            selected_points_tr = np.empty((0, self.dim))

        # Sample points randomly
        if num_random > 0:
            candidates_rand = self._sample_points(100 * self.dim, use_trust_region=False)
            ei_rand = self._acquisition_function(candidates_rand, gp, y_best)
            selected_indices_rand = np.argsort(ei_rand)[-num_random:]
            selected_points_rand = candidates_rand[selected_indices_rand]
        else:
            selected_points_rand = np.empty((0, self.dim))

        selected_points = np.concatenate([selected_points_tr, selected_points_rand], axis=0)

        if self.X is not None:
            distances = cdist(selected_points, self.X)
            min_distances = np.min(distances, axis=1)
            selected_points = selected_points[min_distances > 0.1] # Enforce diversity

            if len(selected_points) < batch_size:
                remaining_needed = batch_size - len(selected_points)
                candidates = self._sample_points(100 * self.dim)
                ei = self._acquisition_function(candidates, gp, y_best)
                additional_indices = np.argsort(ei)[:-batch_size-1:-1][:remaining_needed]
                additional_points = candidates[additional_indices]
                selected_points = np.concatenate([selected_points, additional_points], axis=0)

        return selected_points[:batch_size]

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)
    
    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init, use_trust_region=False)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        self.trust_region_center = best_x # Initialize trust region center with best initial point

        while self.n_evals < self.budget:
            gp = self._fit_model(self.X, self.y)
            batch_size = min(self.budget - self.n_evals, 5)
            next_X = self._select_next_points(gp, best_y, batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            best_idx = np.argmin(self.y)
            current_best_y = self.y[best_idx][0]
            current_best_x = self.X[best_idx]

            if current_best_y < best_y:
                self.trust_region_center = current_best_x
                self.trust_region_radius *= self.radius_decay
                best_y = current_best_y
                best_x = current_best_x
            else:
                if self.trust_region_radius < 1.0:
                    self.trust_region_radius *= self.radius_grow
                    self.trust_region_radius = min(self.trust_region_radius, 2.5)
            
            self.trust_region_radius = max(self.trust_region_radius, self.min_radius)
            self.exploration_weight *= self.exploration_decay # Reduce exploration over time

        return best_y, best_x

```
The algorithm AdaptiveHybridBO_DE got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1670 with standard deviation 0.0963.

took 5.98 seconds to run.

Combine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it. If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.




Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

