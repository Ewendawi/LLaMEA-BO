# Description
**Adaptive Ensemble Trust Region Bayesian Optimization with EI-based Radius Adaptation and Momentum (AETRBO-EIM)**: This algorithm refines the AdaptiveEnsembleTrustRegionBO by introducing a more sophisticated trust region radius adaptation strategy based on the Expected Improvement (EI) and incorporating momentum to smooth the trust region center updates. The radius is adjusted proportionally to the average EI of the evaluated points within the trust region, allowing for more aggressive shrinking when significant improvements are observed and slower expansion when improvements are marginal. A momentum term is added to the trust region center update to prevent oscillations and promote smoother convergence.

# Justification
1.  **EI-based Radius Adaptation:** Using the EI to adjust the trust region radius provides a more informed approach compared to fixed decay/growth rates. High EI values indicate promising regions, justifying a smaller radius for exploitation. Conversely, low EI values suggest a need for broader exploration with a larger radius.
2.  **Momentum for Trust Region Center:** Adding momentum to the trust region center update helps to smooth the optimization process. It prevents the center from oscillating rapidly between different promising regions, promoting more stable and consistent convergence.
3. **Computational Efficiency**: The updates are designed to be computationally efficient by reusing the EI calculations and adding a simple momentum term.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel
from scipy.optimize import minimize

class AdaptiveEnsembleTrustRegionBO_EIM:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * (dim + 1)
        self.ensemble_size = 3
        self.kernels = [
            C(1.0, constant_value_bounds=(1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
            C(1.0, constant_value_bounds=(1e-2, 1e2)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5),
            C(1.0, constant_value_bounds=(1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
        ]
        self.gps = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-5) for kernel in self.kernels]
        self.update_interval = 5
        self.last_gp_update = 0
        self.trust_region_center = np.zeros(dim)
        self.trust_region_radius = 2.5
        self.min_radius = 0.1
        self.radius_decay_rate = 0.95
        self.radius_grow_rate = 1.1
        self.ei_threshold = 0.01  # Threshold for significant EI
        self.center_momentum = 0.5  # Momentum for trust region center update
        self.previous_center = np.zeros(dim) # Store previous center for momentum

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        sample = sampler.random(n=n_points)
        scaled_sample = qmc.scale(sample, -1.0, 1.0)
        points = self.trust_region_center + scaled_sample * self.trust_region_radius
        points = np.clip(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        for gp in self.gps:
            gp.fit(X, y)
        return self.gps

    def _acquisition_function(self, X):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu_list = []
        sigma_list = []
        for gp in self.gps:
            mu, sigma = gp.predict(X, return_std=True)
            mu_list.append(mu)
            sigma_list.append(sigma)

        mu_ensemble = np.mean(mu_list, axis=0)
        sigma_ensemble = np.sqrt(np.mean(np.square(sigma_list) + np.square(mu_list), axis=0) - np.square(mu_ensemble))
        sigma_ensemble = np.clip(sigma_ensemble, 1e-9, np.inf)
        y_best = np.min(self.y)
        gamma = (y_best - mu_ensemble) / sigma_ensemble
        ei = sigma_ensemble * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        population_size = 50
        n_generations = 10
        mutation_rate = 0.1

        population = self._sample_points(population_size)

        for _ in range(n_generations):
            fitness = self._acquisition_function(population).flatten()

            selected_indices = np.random.choice(population_size, size=population_size, replace=True)
            parents = population[selected_indices]

            offspring = parents.copy()
            for i in range(population_size):
                for j in range(self.dim):
                    if np.random.rand() < mutation_rate:
                        offspring[i, j] += np.random.normal(0, 0.5)
                        offspring[i, j] = np.clip(offspring[i, j], self.bounds[0][j], self.bounds[1][j])

            fitness_offspring = self._acquisition_function(offspring).flatten()

            for i in range(population_size):
                if fitness_offspring[i] > fitness[i]:
                    population[i] = offspring[i]

        fitness = self._acquisition_function(population).flatten()
        selected_indices = np.argsort(fitness)[-batch_size:]
        selected_points = population[selected_indices]

        return selected_points

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
        self.previous_center = best_x.copy()

        while self.n_evals < self.budget:
            if self.n_evals - self.last_gp_update >= self.update_interval:
                self._fit_model(self.X, self.y)
                self.last_gp_update = self.n_evals

            batch_size = min(self.budget - self.n_evals, 5)
            next_X = self._select_next_points(batch_size)

            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            best_idx = np.argmin(self.y)
            current_best_y = self.y[best_idx][0]
            current_best_x = self.X[best_idx]

            # Calculate average EI within the trust region
            ei_values = self._acquisition_function(self.X).flatten()
            avg_ei = np.mean(ei_values)

            if current_best_y < best_y:
                # Update trust region center with momentum
                self.trust_region_center = (self.center_momentum * self.previous_center +
                                          (1 - self.center_momentum) * current_best_x)
                self.previous_center = self.trust_region_center.copy()
                
                # Adjust radius based on EI
                if avg_ei > self.ei_threshold:
                    self.trust_region_radius *= self.radius_decay_rate  # Exploit
                else:
                    self.trust_region_radius *= self.radius_grow_rate  # Explore

                best_y = current_best_y
                best_x = current_best_x
            else:
                 self.trust_region_radius *= self.radius_grow_rate

            self.trust_region_radius = max(self.trust_region_radius, self.min_radius)
            self.trust_region_radius = min(self.trust_region_radius, 2.5)

        return best_y, best_x
```
## Feedback
 The algorithm AdaptiveEnsembleTrustRegionBO_EIM got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1735 with standard deviation 0.1044.

took 150.57 seconds to run.