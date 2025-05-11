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
**Adaptive Trust Region with EI-Variance, Kernel Tuning, Dynamic Batching, and Improved Diversity (ATREKVDIDBO)**: This algorithm builds upon ATREKVD_BO by incorporating an improved diversity mechanism and a more robust kernel optimization strategy. The diversity mechanism is enhanced by considering not only the distance to existing points but also the predicted uncertainty (sigma) from the GP model. This helps to select points that are both diverse and located in regions of high uncertainty. The kernel optimization is improved by using a more sophisticated optimization algorithm (SLSQP) and initializing the length scale based on the median distance between points to provide a better starting point for the optimization.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.optimize import minimize, Bounds
from scipy.spatial.distance import cdist

class ATREKVDIDBO:
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
        self.ei_variance_scaling = 0.05
        self.recentering_threshold = 0.5
        self.length_scale = 1.0
        self.kernel_optim_interval = 5
        self.initial_diversity_threshold = 0.1
        self.diversity_threshold = self.initial_diversity_threshold
        self.min_decay = 0.8

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
        # Initialize length scale based on median distance
        if self.X is not None and len(self.X) > 1:
            distances = cdist(self.X, self.X)
            distances = np.triu(distances, k=1)
            median_distance = np.median(distances[distances > 0])
            initial_length_scale = median_distance
        else:
            initial_length_scale = self.length_scale

        def obj(length_scale):
            kernel = ConstantKernel(constant_value=1.0, constant_value_bounds="fixed") * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 10))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-5)
            gp.fit(self.X, self.y)
            return -gp.log_marginal_likelihood()

        bounds = Bounds(1e-2, 10)
        res = minimize(obj, x0=initial_length_scale, method='SLSQP', bounds=bounds)
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

        if self.gp is not None and self.X is not None and self.y is not None:
            mu, sigma = self.gp.predict(candidates, return_std=True)
            sigma = np.clip(sigma, 1e-9, np.inf)
        else:
            sigma = np.ones(len(candidates))  # Assign equal uncertainty if GP is not yet fitted

        selected_indices = np.argsort(ei.flatten())[-batch_size:]
        selected_points = candidates[selected_indices]

        if self.X is not None:
            distances = cdist(selected_points, self.X)
            min_distances = np.min(distances, axis=1)

            # Incorporate uncertainty into diversity metric
            diversity_metric = min_distances + self.diversity_threshold * sigma[selected_indices]
            selected_points = selected_points[diversity_metric > self.diversity_threshold]

            if len(selected_points) < batch_size:
                remaining_needed = batch_size - len(selected_points)
                additional_indices = np.argsort(ei.flatten())[:-batch_size-1:-1][:remaining_needed]
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
            _, sigma = gp.predict(self.X, return_std=True)
            avg_sigma = np.mean(sigma)
            batch_size = min(self.budget - self.n_evals, int(np.ceil(5 * (avg_sigma / np.std(self.bounds))))) if np.std(self.bounds) > 0 else min(self.budget - self.n_evals, 5)
            batch_size = max(1, batch_size)

            # Adapt diversity threshold
            self.diversity_threshold = self.initial_diversity_threshold * (self.trust_region_radius / 2.5) # Scale diversity threshold with trust region radius

            # Select next points
            next_X = self._select_next_points(batch_size)

            # Evaluate points
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Update best solution
            best_idx = np.argmin(self.y)
            current_best_y = self.y[best_idx][0]
            current_best_x = self.X[best_idx]

            # Calculate EI values and statistics
            ei_values = self._acquisition_function(next_X)
            avg_ei = np.mean(ei_values)
            ei_variance = np.var(ei_values)

            # Adjust trust region radius based on EI and its variance
            if current_best_y < best_y:
                # Improvement: shrink radius, considering EI and variance
                decay_rate = self.radius_decay_base + self.ei_scaling * avg_ei - self.ei_variance_scaling * ei_variance
                decay_rate = max(decay_rate, self.min_decay)
                self.trust_region_radius *= decay_rate
                self.trust_region_center = current_best_x
                best_y = current_best_y
                best_x = current_best_x
            else:
                # No improvement: expand radius, but slower if EI is high
                self.trust_region_radius *= (self.radius_grow_base - self.ei_scaling * avg_ei + self.ei_variance_scaling * ei_variance)
                self.trust_region_radius = min(self.trust_region_radius, 2.5)  # Limit to initial radius

            self.trust_region_radius = max(self.trust_region_radius, self.min_radius)

            # Re-center trust region if current center is far from best point
            distance = np.linalg.norm(self.trust_region_center - best_x)
            if distance > self.recentering_threshold * self.trust_region_radius:
                self.trust_region_center = best_x

            iteration += 1

        return best_y, best_x

```
The algorithm ATREKVDIDBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1912 with standard deviation 0.1173.

took 478.71 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

