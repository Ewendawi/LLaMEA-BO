# Description
**Hybrid Ensemble Trust Region Bayesian Optimization with Dynamic Exploration (HETRBO-DE)**: This algorithm synergistically combines the strengths of Adaptive Ensemble Trust Region BO (AETRBO) and Adaptive Hybrid BO with Dynamic Exploration (AHBO-DE). It leverages an ensemble of Gaussian Process Regression (GPR) models with diverse kernels for robust modeling of the objective function, similar to AETRBO. It also incorporates a dynamic exploration strategy inspired by AHBO-DE, which balances trust region exploitation with broader exploration based on a dynamically adjusted exploration weight. The trust region's center and radius are adapted based on the ensemble's predictive uncertainty and observed improvements. The next points are selected by combining trust region sampling, random sampling, and diversity-promoting mechanisms. A simplified evolutionary strategy is used to refine the points within the trust region.

# Justification
This algorithm aims to improve upon existing BO methods by:

1.  **Robust Modeling:** Employing an ensemble of GPs with different kernels enhances the robustness of the model, mitigating the risk of overfitting to a particular kernel choice. This is directly inherited from AETRBO.
2.  **Efficient Exploration-Exploitation Balance:** Dynamically adjusting the exploration weight allows the algorithm to adapt its search strategy based on the optimization progress. Initially, a higher exploration weight encourages broader exploration of the search space. As the optimization progresses and the trust region converges, the exploration weight is reduced, focusing the search on exploiting the promising regions. This is inspired by AHBO-DE.
3.  **Trust Region Adaptation:** Adapting the trust region based on both the ensemble's uncertainty and the observed improvements ensures that the search is focused on promising regions while also allowing for expansion when necessary.
4.  **Diversity Promotion:** Incorporating a diversity-promoting mechanism ensures that the selected points are not clustered too closely together, which can lead to premature convergence.
5.  **Computational Efficiency:** The simplified GP model and evolutionary strategy contribute to the computational efficiency of the algorithm.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel
from scipy.spatial.distance import cdist

class HybridEnsembleTrustRegionBO_DE:
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
        # Dynamic exploration-exploitation balance
        num_trust_region = int(batch_size * (1 - self.exploration_weight))
        num_random = batch_size - num_trust_region

        # Sample points from trust region
        if num_trust_region > 0:
            candidates_tr = self._sample_points(100 * self.dim, use_trust_region=True)
            ei_tr = self._acquisition_function(candidates_tr).flatten()
            selected_indices_tr = np.argsort(ei_tr)[-num_trust_region:]
            selected_points_tr = candidates_tr[selected_indices_tr]
        else:
            selected_points_tr = np.empty((0, self.dim))

        # Sample points randomly
        if num_random > 0:
            candidates_rand = self._sample_points(100 * self.dim, use_trust_region=False)
            ei_rand = self._acquisition_function(candidates_rand).flatten()
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
                candidates = self._sample_points(100 * self.dim, use_trust_region=True)
                ei = self._acquisition_function(candidates).flatten()
                additional_indices = np.argsort(ei)[:-batch_size-1:-1][:remaining_needed]
                additional_points = candidates[additional_indices]
                selected_points = np.concatenate([selected_points, additional_points], axis=0)

        # Simplified Evolutionary Strategy
        population_size = len(selected_points)
        mutation_rate = 0.1
        for i in range(population_size):
            for j in range(self.dim):
                if np.random.rand() < mutation_rate:
                    selected_points[i, j] += np.random.normal(0, 0.1) # Smaller mutation
                    selected_points[i, j] = np.clip(selected_points[i, j], self.bounds[0][j], self.bounds[1][j])

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
        initial_X = self._sample_points(self.n_init, use_trust_region=False)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        self.trust_region_center = best_x

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
            self.exploration_weight *= self.exploration_decay # Reduce exploration over time

        return best_y, best_x
```
## Feedback
 The algorithm HybridEnsembleTrustRegionBO_DE got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1676 with standard deviation 0.1052.

took 118.16 seconds to run.