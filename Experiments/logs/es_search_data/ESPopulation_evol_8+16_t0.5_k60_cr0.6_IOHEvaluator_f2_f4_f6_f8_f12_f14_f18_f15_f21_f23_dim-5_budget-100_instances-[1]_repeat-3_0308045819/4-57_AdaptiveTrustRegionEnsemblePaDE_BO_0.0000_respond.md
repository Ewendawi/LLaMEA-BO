# Description
**Adaptive Trust Region Ensemble with Pareto-based Acquisition and Dynamic Differential Evolution (ATRE-PaDE):** This algorithm combines the strengths of AdaptiveEvolutionaryTrustRegionBO_DKAE and ParetoEnsembleAdaptiveTrustRegionBO. It leverages an ensemble of Gaussian Process (GP) models within an adaptive trust region to enhance robustness and handle model uncertainty. A Pareto-based approach balances exploration (diversity) and exploitation (Expected Improvement) when selecting candidate points within the trust region. The key innovation lies in dynamically adjusting the Differential Evolution (DE) parameters based on the optimization progress and the characteristics of the Pareto front, allowing for more efficient and targeted exploration within the trust region. The algorithm also incorporates a dynamic mechanism for switching between LCB and EI based on the optimization stage, promoting exploration initially and exploitation later.

# Justification
The algorithm is designed to address the limitations of its predecessors by:

1.  **Combining Ensemble and Pareto:** Leveraging the robustness of GP ensembles from ParetoEnsembleAdaptiveTrustRegionBO and the Pareto-based acquisition from AdaptiveParetoTrustRegionBO to balance exploration and exploitation.

2.  **Dynamic DE:** Enhancing the DE component from AdaptiveEvolutionaryTrustRegionBO_DKAE by dynamically adjusting its parameters (population size, mutation factor, crossover rate) based on the diversity and density of the Pareto front. This allows for more efficient exploration within the trust region.

3.  **Dynamic Acquisition Switching:** Introducing a mechanism to switch between LCB (Lower Confidence Bound) and EI (Expected Improvement) based on the optimization progress. LCB is used initially to promote exploration, while EI is used later to focus on exploitation.

4.  **Adaptive Trust Region:** Maintaining an adaptive trust region, similar to both parent algorithms, to focus the search on promising areas while controlling the step size. The trust region radius is adjusted based on the success ratio and GP model error within the region.

5.  **Computational Efficiency:** Balancing the computational cost of the GP ensemble and DE by dynamically adjusting the ensemble size and DE iterations based on the remaining budget.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc, norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution

class AdaptiveTrustRegionEnsemblePaDE_BO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(5 * dim, self.budget // 10)
        self.gp_ensemble = []
        self.ensemble_weights = []
        self.best_x = None
        self.best_y = float('inf')
        self.n_ensemble = 3
        self.trust_region_radius = 2.0 * np.sqrt(dim)
        self.radius_decay = 0.95
        self.radius_increase = 1.1
        self.min_radius = 0.1
        self.success_ratio = 0.0
        self.random_restart_prob = 0.05
        self.pareto_influence = 0.5
        self.pareto_decay = 0.95
        self.pareto_increase = 1.05
        self.de_pop_size = 10
        self.de_mutation = (0.5, 1)
        self.de_crossover = 0.7
        self.lcb_kappa = 2.0
        self.kappa_decay = 0.99
        self.min_kappa = 0.1
        self.gp_error_threshold = 0.1
        self.noise_estimate = 1e-4
        self.acquisition_type = "LCB" #or "EI"
        self.acquisition_switch_iter = self.budget // 3

    def _sample_points(self, n_points):
        if self.best_x is None or np.random.rand() < self.random_restart_prob:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            points = []
            while len(points) < n_points:
                sample = np.random.uniform(-1, 1, size=(self.dim,))
                sample = self.best_x + self.trust_region_radius * sample
                sample = np.clip(sample, self.bounds[0], self.bounds[1])
                points.append(sample)
            return np.array(points)

    def _fit_model(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        if not self.gp_ensemble:
            kernels = [
                ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed"),
                ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds="fixed", nu=0.5),
                ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds="fixed", nu=1.5)
            ]
            for i in range(self.n_ensemble):
                gp = GaussianProcessRegressor(kernel=kernels[i % len(kernels)], n_restarts_optimizer=0, alpha=1e-6)
                gp.fit(X_train, y_train)
                self.gp_ensemble.append(gp)
                self.ensemble_weights.append(1.0 / self.n_ensemble)
        else:
            for gp in self.gp_ensemble:
                gp.fit(X_train, y_train)

        val_errors = []
        for gp in self.gp_ensemble:
            y_pred, _ = gp.predict(X_val, return_std=True)
            error = np.mean((y_pred - y_val.flatten())**2)
            val_errors.append(error)

        val_errors = np.array(val_errors)
        weights = np.exp(-val_errors) / np.sum(np.exp(-val_errors))
        self.ensemble_weights = weights

    def _expected_improvement(self, X):
        ei = np.zeros((len(X), 1))
        for i, gp in enumerate(self.gp_ensemble):
            mu, sigma = gp.predict(X, return_std=True)
            sigma = np.clip(sigma, 1e-9, np.inf)
            imp = self.best_y - mu
            Z = imp / sigma
            ei += self.ensemble_weights[i] * (imp * norm.cdf(Z) + sigma * norm.pdf(Z)).reshape(-1, 1)
        return ei

    def _lower_confidence_bound(self, X):
        lcb = np.zeros((len(X), 1))
        for i, gp in enumerate(self.gp_ensemble):
            mu, sigma = gp.predict(X, return_std=True)
            lcb += self.ensemble_weights[i] * (mu - self.lcb_kappa * sigma).reshape(-1, 1)
        return lcb

    def _ensemble_variance(self, X):
        variance = np.zeros((len(X), 1))
        for i, gp in enumerate(self.gp_ensemble):
            _, sigma = gp.predict(X, return_std=True)
            variance += self.ensemble_weights[i] * sigma.reshape(-1, 1)
        return variance

    def _diversity_metric(self, X):
        if self.X is None:
            return np.ones((len(X), 1))
        else:
            distances = np.min(cdist(X, self.X), axis=1)
            return distances.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        n_candidates = min(100 * self.dim, self.budget - self.n_evals)
        candidates = self._sample_points(n_candidates)

        if self.acquisition_type == "EI":
            ei = self._expected_improvement(candidates)
            ei_normalized = (ei - np.min(ei)) / (np.max(ei) - np.min(ei)) if np.max(ei) != np.min(ei) else np.zeros_like(ei)
            acquisition = ei_normalized
        else:
            lcb = self._lower_confidence_bound(candidates)
            lcb_normalized = (lcb - np.min(lcb)) / (np.max(lcb) - np.min(lcb)) if np.max(lcb) != np.min(lcb) else np.zeros_like(lcb)
            acquisition = lcb_normalized

        diversity = self._diversity_metric(candidates)
        diversity_normalized = (diversity - np.min(diversity)) / (np.max(diversity) - np.min(diversity)) if np.max(diversity) != np.min(diversity) else np.zeros_like(diversity)

        F = np.hstack([acquisition, diversity_normalized])

        is_efficient = np.ones(F.shape[0], dtype=bool)
        for i, c in enumerate(F):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(F[is_efficient] >= c, axis=1)
                is_efficient[i] = True

        pareto_front = candidates[is_efficient]

        if len(pareto_front) > 0:
            # Dynamic DE parameters based on Pareto front
            self.de_pop_size = min(5 * self.dim, len(pareto_front))
            self.de_mutation = (0.5, 1) #adjust based on pareto front spread
            self.de_crossover = 0.7 #adjust based on pareto front density

            def de_objective(x):
                if self.acquisition_type == "EI":
                    ei = self._expected_improvement(x.reshape(1, -1))
                    return -ei[0,0] #minimize negative EI
                else:
                    lcb = self._lower_confidence_bound(x.reshape(1, -1))
                    return lcb[0,0]

            de_bounds = [(max(self.bounds[0][i], self.best_x[i] - self.trust_region_radius),
                          min(self.bounds[1][i], self.best_x[i] + self.trust_region_radius)) for i in range(self.dim)]

            remaining_evals = self.budget - self.n_evals
            maxiter = max(1, int(remaining_evals / (self.de_pop_size * self.dim * 2)))
            maxiter = min(maxiter, 50)

            result = differential_evolution(de_objective, de_bounds, popsize=self.de_pop_size, mutation=self.de_mutation,
                                          recombination=self.de_crossover, maxiter=maxiter, tol=0.01, disp=False)

            next_point = result.x.reshape(1, -1)

        else:
            next_point = self._sample_points(1)

        return next_point

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

        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]
            self.success_ratio = 1.0
        else:
            self.success_ratio *= 0.75

        self.noise_estimate = np.var(self.y)

    def _adjust_trust_region(self):
        if self.best_x is not None:
            distances = np.linalg.norm(self.X - self.best_x, axis=1)
            within_tr = distances < self.trust_region_radius
            if np.any(within_tr):
                X_tr = self.X[within_tr]
                y_tr = self.y[within_tr]
                mu, _ = self.gp.predict(X_tr, return_std=True)
                gp_error = np.mean(np.abs(mu.reshape(-1,1) - y_tr))
            else:
                gp_error = 0.0

            if self.success_ratio > 0.5 and gp_error < self.gp_error_threshold:
                self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
                self.pareto_influence = min(self.pareto_influence * self.pareto_increase, 1.0)
            else:
                self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)
                self.pareto_influence = max(self.pareto_influence * self.pareto_decay, 0.0)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        batch_size = min(1, self.dim)
        while self.n_evals < self.budget:

            if self.n_evals > self.acquisition_switch_iter:
                self.acquisition_type = "EI"

            self._fit_model(self.X, self.y)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self._adjust_trust_region()

            self.lcb_kappa = max(self.lcb_kappa * self.kappa_decay, self.min_kappa)
            self.min_kappa = 0.1 + np.sqrt(self.noise_estimate)

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<AdaptiveTrustRegionEnsemblePaDE_BO>", line 230, in __call__
 230->             next_X = self._select_next_points(batch_size)
  File "<AdaptiveTrustRegionEnsemblePaDE_BO>", line 164, in _select_next_points
 164->             result = differential_evolution(de_objective, de_bounds, popsize=self.de_pop_size, mutation=self.de_mutation,
  File "<AdaptiveTrustRegionEnsemblePaDE_BO>", line 154, in de_objective
 154->                     lcb = self._lower_confidence_bound(x.reshape(1, -1))
  File "<AdaptiveTrustRegionEnsemblePaDE_BO>", line 99, in _lower_confidence_bound
  97 |         lcb = np.zeros((len(X), 1))
  98 |         for i, gp in enumerate(self.gp_ensemble):
  99->             mu, sigma = gp.predict(X, return_std=True)
 100 |             lcb += self.ensemble_weights[i] * (mu - self.lcb_kappa * sigma).reshape(-1, 1)
 101 |         return lcb
ValueError: Input X contains NaN.
GaussianProcessRegressor does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
