# Description
**AdaptiveContextKernelEvolutionaryTrustRegionBO with Dynamic Bandwidth and Kappa Adaptation (ACKETRBO-DBKA):** This algorithm enhances the ACKETRBO by introducing a more sophisticated approach to kernel bandwidth adaptation and kappa annealing. Instead of selecting from a discrete set of kernels, it optimizes the kernel's length scale using L-BFGS-B within a defined range. This allows for finer-grained kernel tuning. Furthermore, the kappa parameter in the LCB acquisition function is dynamically adjusted based on the estimated noise level of the objective function. This provides a more robust exploration-exploitation balance, especially in the early stages of optimization. The DE maxiter is also adjusted dynamically based on the remaining budget and the trust region size.

# Justification
The key improvements in ACKETRBO-DBKA are:

1.  **Continuous Kernel Bandwidth Adaptation:** Switching from discrete kernel selection to continuous optimization of the length scale allows for a more precise fit to the data, potentially capturing finer landscape details. L-BFGS-B is a suitable optimizer for this task, as it handles bounded optimization effectively.

2.  **Noise-Aware Kappa Adaptation:** The original algorithm annealed kappa without considering the actual noise level. Estimating the noise (using the variance of the GP predictions or the MSE on a held-out set) and adapting kappa accordingly provides a more robust and adaptive exploration-exploitation balance. Higher noise levels should correspond to higher kappa values for increased exploration.

3.  **Dynamic DE Iterations:** The number of DE iterations is dynamically adapted based on the remaining budget and the trust region size. This ensures that the DE optimization within the trust region is performed efficiently, allocating more iterations when the budget allows and the trust region is larger, and fewer iterations when the budget is limited or the search space is smaller.

These changes aim to improve the algorithm's ability to adapt to different problem landscapes and noise levels, leading to better overall performance.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution, minimize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.model_selection import train_test_split

class AdaptiveContextKernelEvolutionaryTrustRegionBO_DBKA:
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
        self.trust_region_radius = 2.0
        self.radius_decay = 0.95
        self.radius_increase = 1.1
        self.min_radius = 0.1
        self.success_ratio = 0.0
        self.random_restart_prob = 0.05
        self.de_pop_size = 10
        self.lcb_kappa = 2.0
        self.kappa_decay = 0.995
        self.knn = NearestNeighbors(n_neighbors=5)
        self.context_penalty = 0.1
        self.context_penalty_decay = 0.95
        self.initial_length_scale = 1.0
        self.length_scale_bounds = (0.1, 5.0)
        self.kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.initial_length_scale, length_scale_bounds=self.length_scale_bounds)
        self.kernel_update_interval = 20
        self.min_batch_size = 1
        self.noise_level = 0.01  # Initial guess for noise level
        self.noise_estimate_decay = 0.95

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
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=0, alpha=self.noise_level) #noise_level instead of fixed 1e-6
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

        # Adjust maxiter based on remaining budget and trust region size
        maxiter = max(1, int((self.budget / (self.de_pop_size * self.dim * 2) - self.n_evals/(self.de_pop_size * self.dim * 2)) * (self.trust_region_radius/2.0)))
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

    def _adjust_trust_region(self):
        if self.success_ratio > 0.5:
            self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)

    def _update_kernel(self):
        if self.n_evals % self.kernel_update_interval == 0 and self.X is not None:
            # Optimize length_scale using L-BFGS-B
            def neg_log_likelihood(length_scale):
                kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=length_scale, length_scale_bounds=self.length_scale_bounds)
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=self.noise_level)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gp.fit(self.X, self.y)
                return -gp.log_marginal_likelihood()

            result = minimize(neg_log_likelihood, x0=self.kernel.get_params()['rbf__length_scale'], bounds=[self.length_scale_bounds], method='L-BFGS-B')
            best_length_scale = result.x[0]
            self.kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=best_length_scale, length_scale_bounds=self.length_scale_bounds)

    def _adjust_context_penalty(self):
        if self.success_ratio < 0.2:
            self.context_penalty *= self.context_penalty_decay
            self.context_penalty = max(self.context_penalty, 0.01)

    def _adjust_batch_size(self):
        remaining_evals = self.budget - self.n_evals
        self.batch_size = max(self.min_batch_size, min(1, remaining_evals))

    def _estimate_noise_level(self):
        if self.X is None or len(self.X) < 5:
            return self.noise_level #Return initial noise level

        #Option 1: Use GP variance
        _, var = self.gp.predict(self.X, return_std=True)
        self.noise_level = self.noise_estimate_decay * self.noise_level + (1 - self.noise_estimate_decay) * np.mean(var)
        self.noise_level = np.clip(self.noise_level, 1e-6, 1.0)
        
        #Option 2: Split data and estimate MSE
        #X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, shuffle=True)
        #gp_test = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=0, alpha=1e-6)
        #with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")
        #    gp_test.fit(X_train, y_train)
        #y_pred, _ = gp_test.predict(X_test, return_std=True)
        #self.noise_level = self.noise_estimate_decay * self.noise_level + (1 - self.noise_estimate_decay) * mean_squared_error(y_test, y_pred)
        #self.noise_level = np.clip(self.noise_level, 1e-6, 1.0)

    def _adjust_kappa(self):
         self.lcb_kappa = 1.0 + 2.0 * np.exp(-self.noise_level / 0.1) #Higher noise, higher kappa

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        self._adjust_batch_size()
        while self.n_evals < self.budget:
            self._fit_model(self.X, self.y)
            self._update_kernel()
            next_X = self._select_next_points(self.batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self._adjust_trust_region()
            self._adjust_context_penalty()
            self._estimate_noise_level()
            self._adjust_kappa()
            self._adjust_batch_size()

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<AdaptiveContextKernelEvolutionaryTrustRegionBO_DBKA>", line 178, in __call__
 178->             self._update_kernel()
  File "<AdaptiveContextKernelEvolutionaryTrustRegionBO_DBKA>", line 135, in _update_kernel
 133 |                 return -gp.log_marginal_likelihood()
 134 | 
 135->             result = minimize(neg_log_likelihood, x0=self.kernel.get_params()['rbf__length_scale'], bounds=[self.length_scale_bounds], method='L-BFGS-B')
 136 |             best_length_scale = result.x[0]
 137 |             self.kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=best_length_scale, length_scale_bounds=self.length_scale_bounds)
KeyError: 'rbf__length_scale'
