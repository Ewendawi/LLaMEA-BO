# Description
**ATRDABOUE_v2: Adaptive Trust Region Bayesian Optimization with Dynamic Acquisition Function Balancing, Uncertainty-Aware Exploration, and Adaptive Reward Shaping**

This algorithm builds upon ATRDABOUE by introducing an adaptive reward shaping mechanism in the acquisition function selection process. The core idea is to dynamically adjust the reward signal used to update the alpha and beta parameters of the Beta distributions for Thompson Sampling. The reward is shaped based on the model uncertainty (sigma) at the selected point, encouraging exploration in uncertain regions. This aims to improve the efficiency of acquisition function selection and accelerate convergence.

# Justification
The key improvements are:

1.  **Adaptive Reward Shaping:** The reward signal used to update the Beta distribution parameters is modulated by the model uncertainty (sigma) at the selected point. This encourages exploration in regions of high uncertainty, which can be beneficial for escaping local optima and finding the global optimum. The intuition is that if a selected acquisition function leads to an improvement in a region with high uncertainty, it should be rewarded more strongly.

2.  **Uncertainty-Aware Reward:** The reward is calculated as `reward = (self.best_y - new_y.min()) * (1 + self.gamma * sigma_next.mean())`. This means that the reward is increased proportionally to the average predicted standard deviation (`sigma_next.mean()`) of the points selected in the current batch. The `gamma` parameter controls the strength of this uncertainty-aware reward.

3. **Refined Exploration Weight:** The exploration weight is decayed over time, allowing for more exploitation as the algorithm progresses.

These changes aim to improve the exploration-exploitation trade-off and accelerate convergence by dynamically adapting the reward signal based on model uncertainty.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import beta


class ATRDABOUE_v2:
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
        self.best_x = None
        self.acq_funcs = ['ei', 'ucb']
        self.n_acq = len(self.acq_funcs)
        self.weights = np.ones(self.n_acq) / self.n_acq
        self.gamma = 0.1
        self.exploration_weight = 0.1
        self.exploration_decay = 0.995
        self.alpha = np.ones(self.n_acq)  # Initialize alpha for Beta distribution
        self.beta = np.ones(self.n_acq)   # Initialize beta for Beta distribution
        self.temperature = 1.0  # Initial temperature for Thompson Sampling
        self.temperature_decay = 0.95 # Decay rate for temperature


    def _sample_points(self, n_points, center=None, width=None):
        if center is None:
            sampler = qmc.Sobol(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            lower_bound = np.maximum(self.bounds[0], center - width / 2)
            upper_bound = np.minimum(self.bounds[1], center + width / 2)
            sampler = qmc.Sobol(d=self.dim)
            sample = sampler.random(n=n_points)
            scaled_sample = qmc.scale(sample, lower_bound, upper_bound)
            return scaled_sample

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X, acq_func='ei'):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if acq_func == 'ei':
            best = np.min(self.y)
            imp = best - mu
            Z = imp / (sigma + 1e-9)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0
            acq_values = ei + self.exploration_weight * sigma
            return acq_values

        elif acq_func == 'ucb':
            ucb = mu + 2 * sigma
            acq_values = ucb + self.exploration_weight * sigma
            return acq_values

        else:
            raise ValueError(f"Unknown acquisition function: {acq_func}")

    def _select_next_points(self, batch_size):
        n_candidates = max(2000, batch_size * 200)
        X_cand = self._sample_points(n_candidates, center=self.best_x, width=self.trust_region_width)

        acq_values = np.zeros((self.n_acq, n_candidates))
        for i, acq_func in enumerate(self.acq_funcs):
            acq_values[i, :] = self._acquisition_function(X_cand, acq_func).flatten()

        # Thompson Sampling
        sampled_values = []
        for i in range(self.n_acq):
            sampled_values.append(beta.rvs(self.alpha[i], self.beta[i], size=1)[0])

        # Select acquisition function with highest sampled value
        idx = np.argmax(sampled_values)

        top_indices = np.argsort(acq_values[idx, :])[::-1][:batch_size]
        return X_cand[top_indices], idx, X_cand[top_indices]

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

    def _update_weights(self, idx, new_y, X_next):
        # Reward is based on the improvement
        mu_next, sigma_next = self.model.predict(X_next, return_std=True)
        sigma_next = sigma_next.reshape(-1, 1)

        reward = (self.best_y - new_y.min()) * (1 + self.gamma * sigma_next.mean())


        # Update alpha and beta parameters for the selected acquisition function
        if reward > 0:
            self.alpha[idx] += 1
        else:
            self.beta[idx] += 1

        # Decay temperature
        self.temperature *= self.temperature_decay


    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_x = X_init[np.argmin(y_init)]
        self.best_y = np.min(y_init)

        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            self.model = self._fit_model(self.X, self.y)

            X_next, idx = self._select_next_points(batch_size)
            X_next_eval = X_next
            y_next = self._evaluate_points(func, X_next_eval)

            self._update_eval_points(X_next_eval, y_next)

            self._update_weights(idx, y_next, X_next)

            new_best_y = np.min(self.y)
            new_best_x = self.X[np.argmin(self.y)]

            if (self.best_y - new_best_y) / self.best_y > self.success_threshold:
                self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)
            else:
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)

            self.best_y = new_best_y
            self.best_x = new_best_x
            self.exploration_weight *= self.exploration_decay

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<ATRDABOUE_v2>", line 140, in __call__
 138 |             self.model = self._fit_model(self.X, self.y)
 139 | 
 140->             X_next, idx = self._select_next_points(batch_size)
 141 |             X_next_eval = X_next
 142 |             y_next = self._evaluate_points(func, X_next_eval)
ValueError: too many values to unpack (expected 2)
