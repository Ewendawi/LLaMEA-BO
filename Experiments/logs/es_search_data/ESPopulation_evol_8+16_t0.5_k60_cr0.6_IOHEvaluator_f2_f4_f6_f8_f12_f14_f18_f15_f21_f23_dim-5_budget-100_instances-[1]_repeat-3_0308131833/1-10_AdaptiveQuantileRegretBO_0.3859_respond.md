# Description
**AdaptiveQuantileRegretBO (AQRBO):** This algorithm refines the QuantileRegretBO by dynamically adjusting the Gaussian Process kernel and incorporating a more robust CVaR estimation. It uses a Gaussian Process (GP) to model the objective function and estimates the distribution of the regret. The acquisition function is based on the Conditional Value at Risk (CVaR) of the regret, which is optimized to select the next points. The kernel parameters of the GP are optimized using marginal likelihood estimation. The CVaR calculation is improved by using a more stable numerical approximation and clipping the variance to avoid numerical instability. The quantile level is also adapted based on the optimization progress.

# Justification
The changes aim to improve the GP model fitting and the CVaR estimation.
1.  **Kernel Optimization**: Optimizing the kernel parameters of the GP model using marginal likelihood estimation allows the model to better adapt to the characteristics of the objective function. This can lead to more accurate predictions and improved performance.
2.  **Robust CVaR Estimation**: The original CVaR estimation can be unstable, especially when the quantile level is close to 1 or the variance is very small. The improved CVaR calculation uses a more stable numerical approximation and clips the variance to avoid numerical instability.
3. **Quantile Level Adaptation**: The quantile level is adapted based on the optimization progress. This allows the algorithm to dynamically adjust the risk aversion during the search. The quantile level is increased when the algorithm is making progress and decreased when the algorithm is stagnating.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.optimize import minimize
from scipy.linalg import solve_triangular

class AdaptiveQuantileRegretBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
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
        self.noise_level = 1e-6  # Add a small noise level to the GP to avoid overfitting
        self.exploration_weight = 0.1 # Weight for exploration in acquisition function

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Optimize kernel parameters using marginal likelihood
        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(noise_level=self.noise_level, noise_level_bounds="fixed")
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        if self.gp is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-6, np.inf) # Clip sigma to avoid numerical instability

        regret = mu - self.best_y
        alpha = self.quantile_level

        # Improved CVaR approximation
        VaR = regret + sigma * norm.ppf(alpha)
        CVaR = regret - sigma * norm.pdf(norm.ppf(alpha)) / (1 - alpha)

        # Add exploration bonus
        exploration_bonus = self.exploration_weight * sigma

        return (CVaR - exploration_bonus).reshape(-1, 1)

    def _select_next_points(self, batch_size):
        x_tries = self._sample_points(batch_size * 10)
        acq_values = self._acquisition_function(x_tries)
        indices = np.argsort(acq_values.flatten())[:batch_size]
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

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        while self.n_evals < self.budget:
            self.gp = self._fit_model(self.X, self.y)

            batch_size = min(10, self.budget - self.n_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Adaptive quantile level
            if len(self.y) > self.n_init:
                improvement = self.best_y - np.min(self.y[-batch_size:])
                if improvement > 0:
                    self.quantile_level = min(0.99, self.quantile_level + 0.05) # Increase quantile level if improving
                else:
                    self.quantile_level *= self.quantile_decay # Decay quantile level if not improving
                    self.quantile_level = max(self.quantile_level, self.min_quantile_level)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AdaptiveQuantileRegretBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1622 with standard deviation 0.1011.

took 22.90 seconds to run.