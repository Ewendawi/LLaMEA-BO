# Description
**Hybrid Quantile Regret and Efficient Thompson Sampling Bayesian Optimization (HQRETSBO):** This algorithm combines the strengths of Quantile Regret BO (QRBO) and Efficient Hybrid BO (EHBBO) to achieve a robust and efficient optimization strategy. It uses a Gaussian Process (GP) surrogate model and adaptively switches between a CVaR-based acquisition function (from QRBO) and Thompson Sampling (from EHBBO) based on the optimization progress. A local search around the best-observed solution is also incorporated for refinement. The quantile level in the CVaR acquisition is dynamically adjusted, and the GP model is updated periodically to enhance computational efficiency.

# Justification
The HQRETSBO algorithm leverages the robustness of QRBO in handling noisy evaluations and outliers, while also benefiting from the efficiency of EHBBO's Thompson Sampling and periodic GP updates. The adaptive switching between CVaR and Thompson Sampling allows the algorithm to dynamically adjust its exploration-exploitation trade-off based on the optimization progress.

*   **Adaptive Acquisition Switching:** The algorithm starts with CVaR to handle potential initial uncertainty and outliers. As the optimization progresses and the GP model becomes more accurate, it switches to Thompson Sampling for more efficient exploration. This switch is controlled by a parameter `cvar_switch_threshold`.
*   **Quantile Decay and Minimum Level:** The quantile level in the CVaR acquisition function is decayed over time, but it is kept above a minimum level to prevent premature convergence.
*   **Periodic GP Updates:** The GP model is updated periodically to reduce computational cost, as in EHBBO.
*   **Local Search:** A local search around the best-observed solution is performed to refine the search, as in EHBBO.
*   **Computational Efficiency:** By combining periodic GP updates and adaptive acquisition switching, the algorithm aims to achieve a good balance between computational efficiency and optimization performance.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

class HQRETSBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.gp = None
        self.quantile_level = 0.9
        self.quantile_decay = 0.95
        self.min_quantile_level = 0.5
        self.best_y = np.inf
        self.best_x = None
        self.update_interval = 5
        self.local_search_radius = 0.5
        self.cvar_switch_threshold = 0.5  # Switch to Thompson Sampling when quantile level is below this
        self.use_cvar = True  # Start with CVaR

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        if self.gp is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.gp.predict(X, return_std=True)

        if self.use_cvar:
            # Calculate CVaR of the regret
            regret = mu - self.best_y
            alpha = self.quantile_level

            # CVaR approximation (using Gaussian quantiles)
            VaR = regret + sigma * norm.ppf(alpha)
            CVaR = regret - (sigma * norm.pdf(norm.ppf(alpha)) / (1 - alpha))

            # If alpha is close to 1, the above calculation can be unstable.
            # In this case, we can approximate CVaR with VaR.
            if alpha > 0.99:
                CVaR = VaR
            return CVaR.reshape(-1, 1)
        else:
            # Thompson Sampling
            xi = np.random.normal(mu, sigma)
            return xi.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        x_tries = self._sample_points(batch_size * 10)
        acq_values = self._acquisition_function(x_tries)

        if self.use_cvar:
            indices = np.argsort(acq_values.flatten())[:batch_size]  # Minimize CVaR
        else:
            indices = np.argsort(acq_values.flatten())[::-1][:batch_size]  # Maximize Thompson Sample
        return x_tries[indices]

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
        if self.y[best_index, 0] < self.best_y:
            self.best_x = self.X[best_index]
            self.best_y = self.y[best_index, 0]

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        iteration = 0
        while self.n_evals < self.budget:
            iteration += 1

            if iteration % self.update_interval == 0:
                self.gp = self._fit_model(self.X, self.y)

            batch_size = min(10, self.budget - self.n_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Decay quantile level
            self.quantile_level *= self.quantile_decay
            self.quantile_level = max(self.quantile_level, self.min_quantile_level)

            # Switch to Thompson Sampling if quantile level is low enough
            if self.quantile_level < self.cvar_switch_threshold:
                self.use_cvar = False

            # Local search around the best solution
            if self.best_x is not None and self.n_evals < self.budget:
                local_X = self._sample_points(1) * self.local_search_radius + self.best_x
                local_X = np.clip(local_X, self.bounds[0], self.bounds[1])
                local_y = self._evaluate_points(func, local_X)
                self._update_eval_points(local_X, local_y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm HQRETSBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1554 with standard deviation 0.0996.

took 0.77 seconds to run.