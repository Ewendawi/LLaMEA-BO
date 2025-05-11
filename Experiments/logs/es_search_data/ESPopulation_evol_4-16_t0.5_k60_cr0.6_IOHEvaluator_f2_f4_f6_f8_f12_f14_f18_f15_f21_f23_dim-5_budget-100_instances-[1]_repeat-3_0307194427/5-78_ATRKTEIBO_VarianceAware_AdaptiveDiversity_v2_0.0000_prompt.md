You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- ATRKTEIBO_VarianceAware_v2: 0.1934, 371.93 seconds, **ATRKTEIBO-VarianceAware-v2**: This enhanced algorithm builds upon ATRKTEIBO-VarianceAware by introducing a more adaptive EI scaling mechanism, refining the trust region radius adjustment, and incorporating a dynamic diversity threshold. The EI scaling is now dynamically adjusted based on the iteration number, promoting exploration early on and exploitation later. The trust region radius adjustment is refined by considering a weighted average of the EI and EI variance, allowing for a smoother transition between exploration and exploitation. Finally, the diversity threshold is also dynamically adjusted based on the trust region radius, preventing premature convergence in small trust regions and promoting diversity in larger ones.


- ATRKTEIBO_VarianceAware_AdaptiveDiversity: 0.1920, 358.71 seconds, **ATRKTEIBO_VarianceAware_AdaptiveDiversity:** This algorithm builds upon ATRKTEIBO-VarianceAware by introducing an adaptive diversity threshold in the point selection process. Instead of using a fixed diversity threshold, it dynamically adjusts the threshold based on the current trust region radius and the iteration number. This aims to prevent premature convergence in small trust regions while also allowing for more focused exploitation in later iterations. The EI scaling factor is also dynamically adjusted to improve the balance between exploration and exploitation.


- ATRKBO_VAD: 0.1918, 572.11 seconds, **Adaptive Trust Region with Variance-Aware Radius Control and EI-based Diversity (ATRKBO-VAD)**: This algorithm builds upon the strengths of ATRKTEIBO_VarianceAware and ATREKBO_VEB by combining variance-aware trust region radius control with an enhanced EI-based diversity mechanism. It uses the EI variance to dynamically adjust the trust region radius, promoting exploration when the variance is high and exploitation when it's low. The diversity mechanism selects points based on a combination of distance to existing points and EI value, ensuring a balance between exploration and exploitation. Kernel optimization is performed periodically using SLSQP. The batch size is dynamically adjusted based on the GP's uncertainty, and a minimum decay rate is enforced for the trust region radius. We also introduce a weighted EI and distance diversity measure to enhance exploration.


- ATREKVDIDBO: 0.1912, 478.71 seconds, **Adaptive Trust Region with EI-Variance, Kernel Tuning, Dynamic Batching, and Improved Diversity (ATREKVDIDBO)**: This algorithm builds upon ATREKVD_BO by incorporating an improved diversity mechanism and a more robust kernel optimization strategy. The diversity mechanism is enhanced by considering not only the distance to existing points but also the predicted uncertainty (sigma) from the GP model. This helps to select points that are both diverse and located in regions of high uncertainty. The kernel optimization is improved by using a more sophisticated optimization algorithm (SLSQP) and initializing the length scale based on the median distance between points to provide a better starting point for the optimization.




The selected solution to update is:
**ATRKTEIBO_VarianceAware_AdaptiveDiversity:** This algorithm builds upon ATRKTEIBO-VarianceAware by introducing an adaptive diversity threshold in the point selection process. Instead of using a fixed diversity threshold, it dynamically adjusts the threshold based on the current trust region radius and the iteration number. This aims to prevent premature convergence in small trust regions while also allowing for more focused exploitation in later iterations. The EI scaling factor is also dynamically adjusted to improve the balance between exploration and exploitation.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.optimize import minimize, Bounds
from scipy.spatial.distance import cdist, pdist, squareform

class ATRKTEIBO_VarianceAware_AdaptiveDiversity:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * (dim + 1)
        self.trust_region_center = np.zeros(dim)
        self.trust_region_radius = 2.5
        self.min_radius = 0.1
        self.radius_decay_base = 0.95
        self.radius_grow_base = 1.1
        self.gp = None
        self.ei_scaling = 0.1
        self.initial_ei_scaling = 0.1
        self.recentering_threshold = 0.5
        self.length_scale = 1.0
        self.kernel_optim_interval = 5
        self.initial_diversity_threshold = 0.1
        self.diversity_threshold = 0.1
        self.diversity_adaptation_rate = 0.05

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        sample = sampler.random(n=n_points)
        scaled_sample = qmc.scale(sample, -1.0, 1.0)
        points = self.trust_region_center + scaled_sample * self.trust_region_radius
        points = np.clip(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * \
                 RBF(length_scale=self.length_scale, length_scale_bounds=(1e-3, 1e3)) + \
                 WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-7, 1e-3))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        self.gp.fit(X, y)
        return self.gp

    def _optimize_kernel(self):
        # Initialize length_scale with median distance
        distances = pdist(self.X)
        if len(distances) > 0:
            self.length_scale = np.median(distances)
        else:
            self.length_scale = 1.0

        def obj(length_scale):
            kernel = ConstantKernel(constant_value=1.0, constant_value_bounds="fixed") * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 10))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-5)
            gp.fit(self.X, self.y)
            return -gp.log_marginal_likelihood()

        bounds = Bounds(1e-2, 10)
        res = minimize(obj, x0=self.length_scale, method='SLSQP', bounds=bounds)
        self.length_scale = res.x[0]

    def _acquisition_function(self, X):
        if self.gp is None or self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)
        y_best = np.min(self.y)
        gamma = (y_best - mu) / sigma
        ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(100 * self.dim)
        ei = self._acquisition_function(candidates)

        # Enhanced diversity mechanism using EI
        if self.X is not None:
            distances = cdist(candidates, self.X)
            min_distances = np.min(distances, axis=1)
            # Filter candidates based on both distance and EI
            diversity_mask = min_distances > self.diversity_threshold
            ei_threshold = np.quantile(ei, 0.25)  # Consider only top 75% EI values
            ei_mask = ei.flatten() >= ei_threshold
            selected_candidates = candidates[diversity_mask & ei_mask]
            selected_ei = ei[diversity_mask & ei_mask]

            if len(selected_candidates) < batch_size:
                # If not enough diverse candidates, add some more based on EI only
                remaining_needed = batch_size - len(selected_candidates)
                additional_indices = np.argsort(ei.flatten())[::-1][:remaining_needed]
                additional_candidates = candidates[additional_indices]
                selected_candidates = np.concatenate([selected_candidates, additional_candidates], axis=0)
                selected_ei = np.concatenate([selected_ei, ei[additional_indices]], axis=0)
            candidates = selected_candidates
            ei = selected_ei

        selected_indices = np.argsort(ei.flatten())[-batch_size:]
        selected_points = candidates[selected_indices]
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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        iteration = 0
        while self.n_evals < self.budget:
            # Kernel Optimization
            if iteration % self.kernel_optim_interval == 0:
                self._optimize_kernel()

            # Fit GP model
            gp = self._fit_model(self.X, self.y)

            # Dynamic batch size
            mu, sigma = gp.predict(self.X, return_std=True)
            ei_values = self._acquisition_function(self.X)
            ei_variance = np.var(ei_values)
            avg_sigma = np.mean(sigma)
            batch_size = min(self.budget - self.n_evals, int(np.ceil(5 * (avg_sigma / np.std(self.bounds))))) if np.std(self.bounds) > 0 else min(self.budget - self.n_evals, 5)
            # Adjust batch size based on EI variance
            batch_size = int(batch_size * (1 + self.ei_scaling * ei_variance))
            batch_size = max(1, min(batch_size, self.budget - self.n_evals))

            # Select next points
            next_X = self._select_next_points(batch_size)

            # Evaluate points
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Update best solution
            best_idx = np.argmin(self.y)
            current_best_y = self.y[best_idx][0]
            current_best_x = self.X[best_idx]

            # Calculate average EI and variance of evaluated points
            ei_values = self._acquisition_function(next_X)
            avg_ei = np.mean(ei_values)
            ei_variance = np.var(ei_values)

            # Adjust trust region radius based on EI and EI variance
            if current_best_y < best_y:
                # Improvement: shrink radius based on EI, slower if EI variance is high
                self.trust_region_radius *= (self.radius_decay_base + self.ei_scaling * avg_ei) * (1 - self.ei_scaling * ei_variance)
                self.trust_region_center = current_best_x
                best_y = current_best_y
                best_x = current_best_x
            else:
                # No improvement: expand radius, but slower if EI is high, faster if EI variance is high
                self.trust_region_radius *= (self.radius_grow_base - self.ei_scaling * avg_ei) * (1 + self.ei_scaling * ei_variance)
                self.trust_region_radius = min(self.trust_region_radius, 2.5)  # Limit to initial radius

            self.trust_region_radius = max(self.trust_region_radius, self.min_radius)

            # Re-center trust region if current center is far from best point
            distance = np.linalg.norm(self.trust_region_center - best_x)
            if distance > self.recentering_threshold * self.trust_region_radius:
                self.trust_region_center = best_x

            # Adapt diversity threshold
            self.diversity_threshold = self.initial_diversity_threshold * (self.trust_region_radius / 2.5) * (1 - iteration / self.budget)
            self.diversity_threshold = np.clip(self.diversity_threshold, 0.01, 0.5)

            # Adapt EI scaling
            self.ei_scaling = self.initial_ei_scaling * (1 - iteration / self.budget)
            self.ei_scaling = np.clip(self.ei_scaling, 0.01, 0.2)

            iteration += 1

        return best_y, best_x

```
The algorithm ATRKTEIBO_VarianceAware_AdaptiveDiversity got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1920 with standard deviation 0.1151.

took 358.71 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

