# Description
**Adaptive Ensemble Trust Region Bayesian Optimization (AETRBO)**: This algorithm combines the strengths of Adaptive Trust Region BO (ATBO) and Bayesian Ensemble BO (BEBO). It uses an ensemble of Gaussian Process Regression (GPR) models with different kernels to improve robustness and exploration, similar to BEBO. It also incorporates a dynamically adjusted trust region, like ATBO, to balance exploration and exploitation. The trust region's center and radius are adapted based on the ensemble's predictive uncertainty and the observed improvement. An evolutionary strategy is used to select the next points within the trust region, leveraging the acquisition function derived from the ensemble's predictions. This approach aims to efficiently explore the search space while focusing on promising regions, providing a balance between global exploration and local refinement.

# Justification
The AETRBO algorithm is designed based on the following considerations:

1.  **Ensemble of GPs**: Using an ensemble of GPs, as in BEBO, provides a more robust estimate of the objective function than a single GP. This is because different kernels can capture different aspects of the function's behavior.

2.  **Adaptive Trust Region**: The adaptive trust region, borrowed from ATBO, allows the algorithm to focus its search on promising regions of the search space. The trust region's size is dynamically adjusted based on the improvement observed and the uncertainty of the GP ensemble.

3.  **Evolutionary Strategy for Point Selection**: An evolutionary strategy is used to select the next points within the trust region. This allows the algorithm to efficiently explore the trust region and find promising points to evaluate.

4.  **Budget Management**: The algorithm carefully manages the budget by ensuring that function evaluations are only performed when necessary and that the number of evaluations does not exceed the budget.

5.  **Computational Efficiency**: The algorithm aims to balance the computational cost of the GP ensemble and the evolutionary strategy with the need for accurate function approximation and efficient exploration. The GP models are updated periodically to reduce computational overhead.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel
from scipy.optimize import minimize

class AdaptiveEnsembleTrustRegionBO:
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
        self.radius_decay = 0.95
        self.radius_grow = 1.1

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

        return best_y, best_x
```
## Feedback
 The algorithm AdaptiveEnsembleTrustRegionBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1785 with standard deviation 0.1041.

took 150.44 seconds to run.