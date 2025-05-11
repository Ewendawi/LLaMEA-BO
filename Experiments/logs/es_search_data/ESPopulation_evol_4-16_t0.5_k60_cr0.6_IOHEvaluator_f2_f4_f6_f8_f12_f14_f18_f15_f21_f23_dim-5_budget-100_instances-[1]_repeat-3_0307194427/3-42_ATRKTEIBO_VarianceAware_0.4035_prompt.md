You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- AdaptiveTrustRegionBO_DREV: 0.1916, 111.35 seconds, **AdaptiveTrustRegionBO with Dynamic Radius and EI-Variance Improvement (ATRBO-DREV)**: This algorithm builds upon the AdaptiveTrustRegionBO_DR by incorporating the variance of the EI values into the radius adjustment strategy. Instead of solely relying on the average EI, it considers both the average EI and the variance of EI to balance exploration and exploitation more effectively. High average EI with low variance suggests exploitation, while high average EI with high variance indicates potential for further exploration. The radius adjustment is modified to account for this variance. Additionally, a lower bound on the radius decay is introduced to prevent premature convergence.


- ATRKTEIBO: 0.1867, 614.60 seconds, **Adaptive Trust Region with Kernel Tuning and EI-based Batching (ATRKTEIBO)**: This algorithm combines the adaptive trust region management from AdaptiveTrustRegionBO_DR with the kernel lengthscale optimization and dynamic batch sizing from EnhancedEfficientHybridBO. It uses Expected Improvement (EI) for acquisition and leverages both EI and GP uncertainty for trust region and batch size adjustments. This aims to balance local exploitation within the trust region with global exploration, while adapting the GP model to the function's characteristics and efficiently utilizing the evaluation budget.


- AdaptiveEnsembleTrustRegionBO_EIR: 0.1796, 148.93 seconds, **Adaptive Ensemble Trust Region BO with EI-based Radius and Recenter (AETRBO-EIR)**: This algorithm combines the strengths of AdaptiveTrustRegionBO_DR and AdaptiveEnsembleTrustRegionBO. It uses an ensemble of Gaussian Process Regression (GPR) models with different kernels to improve robustness. It also incorporates a dynamically adjusted trust region, where the radius is adjusted based on the Expected Improvement (EI) values and the trust region is re-centered when the current center is far from the best observed point. Furthermore, it uses a more efficient point selection strategy based on EI values within the trust region.


- AdaptiveTrustRegionDKEBO: 0.1758, 41.00 seconds, **Adaptive Trust Region with Dynamic Kernel and EI-based Radius Adjustment (ATRDKEBO)**: This algorithm combines the adaptive trust region approach with dynamic radius adjustment from AdaptiveTrustRegionBO_DR and the adaptive kernel lengthscale optimization from EnhancedEfficientHybridBO. It uses Expected Improvement (EI) for acquisition and adjusts the trust region radius based on EI values. Additionally, it incorporates periodic kernel lengthscale optimization using L-BFGS-B to better capture the function's characteristics. A diversity-promoting mechanism is added to the next point selection to avoid premature convergence.




The selected solution to update is:
**Adaptive Trust Region with Kernel Tuning and EI-based Batching (ATRKTEIBO)**: This algorithm combines the adaptive trust region management from AdaptiveTrustRegionBO_DR with the kernel lengthscale optimization and dynamic batch sizing from EnhancedEfficientHybridBO. It uses Expected Improvement (EI) for acquisition and leverages both EI and GP uncertainty for trust region and batch size adjustments. This aims to balance local exploitation within the trust region with global exploration, while adapting the GP model to the function's characteristics and efficiently utilizing the evaluation budget.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

class ATRKTEIBO:
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
        self.recentering_threshold = 0.5
        self.length_scale = 1.0
        self.kernel_optim_interval = 5
        self.diversity_threshold = 0.1

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
        def obj(length_scale):
            kernel = ConstantKernel(constant_value=1.0, constant_value_bounds="fixed") * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 10))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-5)
            gp.fit(self.X, self.y)
            return -gp.log_marginal_likelihood()

        res = minimize(obj, x0=self.length_scale, method='L-BFGS-B', bounds=[(1e-2, 10)])
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

        selected_indices = np.argsort(ei.flatten())[-batch_size:]
        selected_points = candidates[selected_indices]

        if self.X is not None:
            distances = cdist(selected_points, self.X)
            min_distances = np.min(distances, axis=1)
            selected_points = selected_points[min_distances > self.diversity_threshold]
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

            # Select next points
            next_X = self._select_next_points(batch_size)

            # Evaluate points
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Update best solution
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
                self.trust_region_radius = min(self.trust_region_radius, 2.5)  # Limit to initial radius

            self.trust_region_radius = max(self.trust_region_radius, self.min_radius)

            # Re-center trust region if current center is far from best point
            distance = np.linalg.norm(self.trust_region_center - best_x)
            if distance > self.recentering_threshold * self.trust_region_radius:
                self.trust_region_center = best_x

            iteration += 1

        return best_y, best_x

```
The algorithm ATRKTEIBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1867 with standard deviation 0.1177.

took 614.60 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

