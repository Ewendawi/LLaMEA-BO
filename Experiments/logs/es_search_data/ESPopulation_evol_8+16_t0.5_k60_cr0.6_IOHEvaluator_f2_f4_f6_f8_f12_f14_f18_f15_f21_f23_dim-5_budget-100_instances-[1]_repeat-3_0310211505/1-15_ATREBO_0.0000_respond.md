# Description
Adaptive Trust Region with Ensemble Acquisition Bayesian Optimization (ATREBO) is a Bayesian optimization algorithm that combines the adaptive trust region approach from ATRBO with an ensemble of acquisition functions. It uses a Gaussian Process Regression (GPR) model as a surrogate and adaptively adjusts the trust region based on the success of previous iterations. The acquisition function is an ensemble of Expected Improvement (EI) and Upper Confidence Bound (UCB), dynamically weighted based on their recent performance within the trust region. To avoid the `NaN` error from EHBBO, the K-means clustering is replaced with a top-k selection based on acquisition function values. A robust EI calculation is implemented.

# Justification
This algorithm combines the strengths of ATRBO (adaptive trust region) and EHBBO (ensemble acquisition). The adaptive trust region helps to focus the search in promising areas, while the ensemble acquisition function balances exploration and exploitation. Dynamically weighting EI and UCB allows the algorithm to adapt to the characteristics of the objective function. The top-k selection avoids the `NaN` errors associated with K-means clustering when acquisition values are very close, as seen in the EHBBO error log. The trust region width is also dynamically adjusted based on success.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

class ATREBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(10 * dim, self.budget // 5)
        self.trust_region_width = 2.0
        self.success_threshold = 0.1
        self.best_y = np.inf
        self.ei_weight = 0.5  # Initial weight for EI, UCB is 1 - ei_weight

    def _sample_points(self, n_points, center=None, width=None):
        if center is None:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            lower_bound = np.maximum(self.bounds[0], center - width / 2)
            upper_bound = np.minimum(self.bounds[1], center + width / 2)
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            scaled_sample = qmc.scale(sample, lower_bound, upper_bound)
            return scaled_sample

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement
        best = np.min(self.y)
        imp = best - mu
        Z = imp / (sigma + 1e-9)
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Upper Confidence Bound
        ucb = mu + 2 * sigma

        # Weighted acquisition function
        acq = self.ei_weight * ei + (1 - self.ei_weight) * ucb
        return acq

    def _select_next_points(self, batch_size):
        n_candidates = max(1000, batch_size * 100)
        X_cand = self._sample_points(n_candidates, center=self.best_x, width=self.trust_region_width)
        acq_values = self._acquisition_function(X_cand)
        top_indices = np.argsort(acq_values.flatten())[::-1][:batch_size]
        return X_cand[top_indices]

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.concatenate((self.X, new_X), axis=0)
            self.y = np.concatenate((self.y, new_y), axis=0)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_x = X_init[np.argmin(y_init)]
        self.best_y = np.min(y_init)

        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            self.model = self._fit_model(self.X, self.y)
            X_next = self._select_next_points(batch_size)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

            new_best_y = np.min(self.y)
            new_best_x = self.X[np.argmin(self.y)]

            # Adjust trust region width
            if (self.best_y - new_best_y) / (abs(self.best_y) + 1e-9) > self.success_threshold:
                self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)
            else:
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)

            # Adjust EI/UCB weight based on performance
            ei_values = self._acquisition_function(self.X) # Recalculate acquisition values
            ei_improvement = -np.inf
            ucb_improvement = -np.inf

            if len(ei_values) > batch_size:
                ei_improvement = np.mean(ei_values[-batch_size:])
                ucb_values = mu + 2*sigma # Calculate UCB values
                ucb_improvement = np.mean(ucb_values[-batch_size:])

            if ei_improvement > ucb_improvement:
                self.ei_weight = min(1.0, self.ei_weight + 0.05)
            else:
                self.ei_weight = max(0.0, self.ei_weight - 0.05)

            self.best_y = new_best_y
            self.best_x = new_best_x

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<ATREBO>", line 113, in __call__
 111 |             if len(ei_values) > batch_size:
 112 |                 ei_improvement = np.mean(ei_values[-batch_size:])
 113->                 ucb_values = mu + 2*sigma # Calculate UCB values
 114 |                 ucb_improvement = np.mean(ucb_values[-batch_size:])
 115 | 
NameError: name 'mu' is not defined
