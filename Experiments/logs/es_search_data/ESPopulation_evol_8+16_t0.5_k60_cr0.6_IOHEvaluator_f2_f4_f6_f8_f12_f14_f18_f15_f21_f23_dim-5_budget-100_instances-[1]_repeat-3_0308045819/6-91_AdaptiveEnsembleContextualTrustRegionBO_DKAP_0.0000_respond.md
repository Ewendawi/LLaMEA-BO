# Description
**AdaptiveEnsembleContextualTrustRegionBO with Dynamic Kernel Adaptation and Pareto-based Exploration (AECTRBO-DKAP):** This algorithm refines the previous AdaptiveEnsembleContextualTrustRegionBO by incorporating dynamic kernel adaptation within the GP ensemble and introducing a Pareto-based approach to balance exploration and exploitation within the trust region. The kernel adaptation selects the best kernel for each GP in the ensemble based on a validation set performance, and the Pareto-based exploration uses both LCB and a diversity metric in the acquisition function. This aims to improve the GP model's accuracy and the algorithm's ability to find promising regions in the search space.

# Justification
1.  **Dynamic Kernel Adaptation:** The original algorithm uses a fixed set of kernels for the GP ensemble. By dynamically selecting the best kernel for each GP based on validation set performance, we can improve the accuracy of the GP models and adapt to varying landscape complexities. This is done by splitting the data into training and validation sets and evaluating the likelihood of the validation set given the GP trained on the training set.

2.  **Pareto-based Exploration:** The original algorithm uses LCB with a context penalty for exploration. Introducing a Pareto-based approach balances exploration (diversity) and exploitation (LCB) more effectively. This is achieved by considering both LCB and a diversity metric (Euclidean distance to the nearest existing point) as objectives in a Pareto sense. Differential Evolution (DE) is used to find Pareto-optimal solutions.

3.  **Computational Efficiency:** The kernel selection process adds some computational overhead, but by performing it every `kernel_selection_interval` iterations, we can amortize the cost. The Pareto-based exploration also adds some overhead due to the DE optimization, but the adaptive DE population size helps to control the cost.

4. **Noise Aware Kappa Adjustment:** The kappa parameter in LCB is crucial for balancing exploration and exploitation. Instead of solely relying on a decaying schedule, it is dynamically adjusted based on the estimated noise level in the objective function. This allows the algorithm to adapt its exploration-exploitation balance more effectively in noisy environments.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from scipy.optimize import differential_evolution
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
import warnings
from sklearn.model_selection import train_test_split

class AdaptiveEnsembleContextualTrustRegionBO_DKAP:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(5*dim, self.budget//10)

        self.n_ensemble = 3 # Number of GPs in the ensemble
        self.kernels = [
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
            ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5),
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1))
        ]
        self.gps = [GaussianProcessRegressor(kernel=self.kernels[i % len(self.kernels)], n_restarts_optimizer=2, alpha=1e-6) for i in range(self.n_ensemble)]
        self.active_kernels = [i % len(self.kernels) for i in range(self.n_ensemble)]

        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0 * np.sqrt(dim)
        self.radius_decay = 0.95
        self.radius_increase = 1.1
        self.min_radius = 0.1
        self.success_ratio = 0.0
        self.random_restart_prob = 0.05
        self.de_pop_size = 10
        self.lcb_kappa = 2.0
        self.kappa_decay = 0.99
        self.min_kappa = 0.1
        self.gp_error_threshold = 0.1
        self.noise_estimate = 1e-4
        self.knn = NearestNeighbors(n_neighbors=5)
        self.context_penalty = 0.1
        self.penalty_decay = 0.95
        self.min_penalty = 0.01
        self.kernel_selection_interval = 5 # Perform kernel selection every n iterations
        self.diversity_weight = 0.1 # Weight for diversity in Pareto front
        self.diversity_weight_decay = 0.95
        self.min_diversity_weight = 0.01

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
        self.knn.fit(X)
        for i, gp in enumerate(self.gps):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.kernel = self.kernels[self.active_kernels[i]]
                gp.fit(X, y)

    def _select_best_kernel(self, X, y, gp_index):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        best_kernel_index = self.active_kernels[gp_index]
        best_log_likelihood = -np.inf

        for i in range(len(self.kernels)):
            gp = GaussianProcessRegressor(kernel=self.kernels[i], n_restarts_optimizer=2, alpha=1e-6)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X_train, y_train)
            log_likelihood = gp.log_marginal_likelihood(gp.kernel_.theta_, clone_kernel=False)
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_kernel_index = i
        return best_kernel_index

    def _acquisition_function(self, X):
        mu_list = []
        sigma_list = []
        for gp in self.gps:
            mu, sigma = gp.predict(X, return_std=True)
            mu_list.append(mu)
            sigma_list.append(sigma)

        mu = np.mean(mu_list, axis=0)
        sigma = np.mean(sigma_list, axis=0)
        sigma = np.clip(sigma, 1e-9, np.inf)
        LCB = mu - self.lcb_kappa * sigma

        if self.X is not None and len(self.X) > 0:
            distances = euclidean_distances(X, self.X)
            min_distances = np.min(distances, axis=1)
            context_penalty = self.context_penalty * min_distances
            acquisition = LCB + context_penalty
        else:
            acquisition = LCB

        return acquisition.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        if self.best_x is None:
            return self._sample_points(batch_size)

        def de_objective(x):
            x = x.reshape(1, -1)
            mu_list = []
            sigma_list = []
            for gp in self.gps:
                mu, sigma = gp.predict(x, return_std=True)
                mu_list.append(mu)
                sigma_list.append(sigma)

            mu = np.mean(mu_list, axis=0)
            sigma = np.mean(sigma_list, axis=0)
            sigma = np.clip(sigma, 1e-9, np.inf)
            lcb = mu - self.lcb_kappa * sigma

            if self.X is not None and len(self.X) > 0:
                distances = euclidean_distances(x, self.X)
                min_distance = np.min(distances)
            else:
                min_distance = 10.0

            # Pareto objective: minimize LCB and maximize diversity
            return lcb - self.diversity_weight * min_distance

        de_bounds = [(max(self.bounds[0][i], self.best_x[i] - self.trust_region_radius),
                      min(self.bounds[1][i], self.best_x[i] + self.trust_region_radius)) for i in range(self.dim)]

        remaining_evals = self.budget - self.n_evals
        # Adaptive DE population size
        self.de_pop_size = max(5, min(20, int(remaining_evals / (self.dim * 5))))
        maxiter = max(1, int(remaining_evals / (self.de_pop_size * self.dim * 2)))
        maxiter = min(maxiter, 100)

        result = differential_evolution(de_objective, de_bounds, popsize=self.de_pop_size, maxiter=maxiter, tol=0.01, disp=False)

        return result.x.reshape(1, -1)

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
                mu_list = []
                for gp in self.gps:
                    mu, _ = gp.predict(X_tr, return_std=True)
                    mu_list.append(mu)
                mu = np.mean(mu_list, axis=0)
                gp_error = np.mean(np.abs(mu.reshape(-1,1) - y_tr))
            else:
                gp_error = 0.0

            if self.success_ratio > 0.5 and gp_error < self.gp_error_threshold:
                self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
            else:
                self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        batch_size = min(4, self.dim)
        while self.n_evals < self.budget:
            if self.n_evals % self.kernel_selection_interval == 0:
                for i in range(self.n_ensemble):
                    self.active_kernels[i] = self._select_best_kernel(self.X, self.y, i)

            self._fit_model(self.X, self.y)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self._adjust_trust_region()

            self.lcb_kappa = max(self.lcb_kappa * self.kappa_decay, self.min_kappa)
            self.min_kappa = 0.1 + np.sqrt(self.noise_estimate)
            self.context_penalty = max(self.context_penalty * self.penalty_decay, self.min_penalty)
            self.diversity_weight = max(self.diversity_weight * self.diversity_weight_decay, self.min_diversity_weight)

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<AdaptiveEnsembleContextualTrustRegionBO_DKAP>", line 211, in __call__
 211->                     self.active_kernels[i] = self._select_best_kernel(self.X, self.y, i)
  File "<AdaptiveEnsembleContextualTrustRegionBO_DKAP>", line 87, in _select_best_kernel
  85 |                 warnings.simplefilter("ignore")
  86 |                 gp.fit(X_train, y_train)
  87->             log_likelihood = gp.log_marginal_likelihood(gp.kernel_.theta_, clone_kernel=False)
  88 |             if log_likelihood > best_log_likelihood:
  89 |                 best_log_likelihood = log_likelihood
AttributeError: 'Product' object has no attribute 'theta_'. Did you mean: 'theta'?
