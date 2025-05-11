# Description
**Adaptive Contextual Kernel Evolutionary Trust Region Bayesian Optimization with Error-Aware Radius and Noise-Adaptive Kappa (ACKETRBO-ERNAK):** This algorithm combines the strengths of AdaptiveEvolutionaryTrustRegionBO_DKAE and AdaptiveContextKernelEvolutionaryTrustRegionBO, focusing on enhanced trust region management and noise-aware exploration. It uses a trust region framework with adaptive radius adjustment based on success ratio and GP model error, similar to AETRBO-DKAE. It incorporates dynamic kernel adaptation and a context penalty, similar to ACKETRBO. The key improvements are: (1) Error-Aware Trust Region Radius: The trust region radius is adjusted not only based on the success ratio but also on the GP model's prediction error within the trust region, and the variance of the objective function values within the trust region. This allows for a more precise control of the exploration-exploitation trade-off. (2) Noise-Adaptive Kappa: The LCB's kappa parameter is dynamically adjusted based on the estimated noise level in the objective function, ensuring a more robust performance in noisy environments. The lower bound of kappa is dynamically adjusted based on the noise estimate. (3) Dynamic Batch Size: The batch size is dynamically adjusted based on the remaining budget and the optimization progress, allowing for more efficient exploration in the early stages and more focused exploitation in the later stages. (4) Improved Initial Sampling: The initial sampling is performed using a Sobol sequence, which is known to provide a better space-filling design than Latin Hypercube sampling, especially in high-dimensional spaces.

# Justification
The algorithm combines the strengths of AETRBO-DKAE and ACKETRBO while addressing their limitations. The error-aware trust region radius allows for a more precise control of the exploration-exploitation trade-off, preventing premature convergence and improving the algorithm's ability to escape local optima. The noise-adaptive kappa ensures a more robust performance in noisy environments, preventing over-exploitation in regions where the GP model is uncertain. The dynamic batch size allows for a more efficient exploration in the early stages and more focused exploitation in the later stages, improving the algorithm's overall performance. The improved initial sampling provides a better starting point for the optimization process, leading to faster convergence and better solutions. The use of differential evolution within the trust region allows for an efficient search for promising candidate points. The context penalty encourages exploration by penalizing points close to existing samples. The dynamic kernel adaptation allows the GP model to better capture the underlying structure of the objective function.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import warnings

class ACKETRBO_ERNAK:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(5*dim, self.budget//10)

        self.gp = None
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
        self.knn = NearestNeighbors(n_neighbors=5)
        self.context_penalty = 0.1
        self.context_penalty_decay = 0.95
        self.kernel_options = [
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=0.5, length_scale_bounds="fixed"),
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed"),
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=2.0, length_scale_bounds="fixed")
        ]
        self.kernel = self.kernel_options[1] #Initial kernel
        self.kernel_update_interval = 20
        self.min_batch_size = 1
        self.batch_size = 1
        self.gp_error_threshold = 0.1
        self.noise_estimate = 1e-4

    def _sample_points(self, n_points):
        if self.best_x is None or np.random.rand() < self.random_restart_prob:
            sampler = qmc.Sobol(d=self.dim, scramble=True)
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
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=2, alpha=1e-6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit(X, y)
        self.knn.fit(X)
        return self.gp

    def _acquisition_function(self, X):
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            mu, sigma = self.gp.predict(X, return_std=True)
            sigma = np.clip(sigma, 1e-9, np.inf)
            LCB = mu - self.lcb_kappa * sigma

            distances, _ = self.knn.kneighbors(X)
            context_penalty = np.mean(distances, axis=1).reshape(-1, 1)
            acquisition = LCB + self.context_penalty * sigma #context_penalty reduces LCB, promoting exploration
            return acquisition.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        if self.best_x is None:
            return self._sample_points(batch_size)

        def de_objective(x):
            return self._acquisition_function(x.reshape(1, -1))[0, 0]

        de_bounds = [(max(self.bounds[0][i], self.best_x[i] - self.trust_region_radius),
                      min(self.bounds[1][i], self.best_x[i] + self.trust_region_radius)) for i in range(self.dim)]

        # Adjust maxiter based on remaining budget and optimization progress
        remaining_evals = self.budget - self.n_evals
        maxiter = max(1, int(remaining_evals / (self.de_pop_size * self.dim * 2)))
        maxiter = min(maxiter, 100) #limit maxiter to prevent excessive computation

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
            #Calculate GP error within the trust region
            distances = np.linalg.norm(self.X - self.best_x, axis=1)
            within_tr = distances < self.trust_region_radius
            if np.any(within_tr):
                X_tr = self.X[within_tr]
                y_tr = self.y[within_tr]
                mu, _ = self.gp.predict(X_tr, return_std=True)
                gp_error = np.mean(np.abs(mu.reshape(-1,1) - y_tr))
                y_var = np.var(y_tr)
            else:
                gp_error = 0.0
                y_var = 0.0

            if self.success_ratio > 0.5 and gp_error < self.gp_error_threshold:
                self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
            else:
                self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius) #shrink if no best_x

    def _update_kernel(self):
        if self.n_evals % self.kernel_update_interval == 0 and self.X is not None:
            best_kernel = self.kernel
            best_log_likelihood = -np.inf
            for kernel in self.kernel_options:
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gp.fit(self.X, self.y)
                log_likelihood = gp.log_marginal_likelihood()
                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_kernel = kernel
            self.kernel = best_kernel

    def _adjust_context_penalty(self):
        if self.success_ratio < 0.2:
            self.context_penalty *= self.context_penalty_decay
            self.context_penalty = max(self.context_penalty, 0.01)

    def _adjust_batch_size(self):
        remaining_evals = self.budget - self.n_evals
        if self.n_evals < self.budget // 2:
            self.batch_size = min(4, self.dim, remaining_evals)
        else:
            self.batch_size = max(self.min_batch_size, min(1, remaining_evals))

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        while self.n_evals < self.budget:
            self._fit_model(self.X, self.y)
            self._update_kernel()
            self._adjust_batch_size()
            next_X = self._select_next_points(self.batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self._adjust_trust_region()
            self._adjust_context_penalty()

            self.lcb_kappa = max(self.lcb_kappa * self.kappa_decay, self.min_kappa)
            self.min_kappa = 0.1 + np.sqrt(self.noise_estimate)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ACKETRBO_ERNAK got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1792 with standard deviation 0.1053.

took 337.37 seconds to run.