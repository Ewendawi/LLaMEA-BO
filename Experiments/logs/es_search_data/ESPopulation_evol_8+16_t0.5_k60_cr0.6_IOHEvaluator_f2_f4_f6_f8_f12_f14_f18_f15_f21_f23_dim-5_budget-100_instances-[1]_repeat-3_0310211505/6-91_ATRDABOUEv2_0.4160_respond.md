# Description
**ATRDABOUE-v2: Adaptive Trust Region Bayesian Optimization with Dynamic Acquisition Function Balancing, Uncertainty-Aware Exploration, and Adaptive Exploration Weight.** This algorithm builds upon ATRDABOUE by introducing a dynamic adjustment of the exploration weight based on the trust region size and the model uncertainty. The exploration weight in the acquisition function is adapted to encourage more exploration when the trust region is large or the model uncertainty is high, and more exploitation when the trust region is small and the model is confident. This aims to improve the balance between exploration and exploitation, especially in different stages of the optimization process.

# Justification
The key improvement in this version is the adaptive exploration weight.
1.  **Adaptive Exploration Weight:** The exploration weight is no longer fixed but is dynamically adjusted based on the trust region width and the average predicted standard deviation of the Gaussian Process model. When the trust region is large, exploration should be encouraged to search for promising regions. Similarly, when the model uncertainty is high, exploration is beneficial to refine the model. When the trust region is small and the model is confident, exploitation should be favored to fine-tune the solution.

This adaptive strategy allows the algorithm to automatically adjust its behavior based on the current state of the optimization process, leading to more efficient exploration and exploitation.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import beta


class ATRDABOUEv2:
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

        # Adaptive Exploration Weight
        exploration_weight = self.exploration_weight * (self.trust_region_width / 2.0) * np.mean(sigma)

        if acq_func == 'ei':
            best = np.min(self.y)
            imp = best - mu
            Z = imp / (sigma + 1e-9)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0
            acq_values = ei + exploration_weight * sigma
            return acq_values

        elif acq_func == 'ucb':
            ucb = mu + 2 * sigma
            acq_values = ucb + exploration_weight * sigma
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
        return X_cand[top_indices], idx

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

    def _update_weights(self, idx, new_y):
        # Reward is based on the improvement
        reward = self.best_y - new_y.min()

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

            y_next = self._evaluate_points(func, X_next)

            self._update_eval_points(X_next, y_next)

            self._update_weights(idx, y_next)

            new_best_y = np.min(self.y)
            new_best_x = self.X[np.argmin(self.y)]

            if (self.best_y - new_best_y) / self.best_y > self.success_threshold:
                self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)
            else:
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)

            self.best_y = new_best_y
            self.best_x = new_best_x

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Feedback
 The algorithm ATRDABOUEv2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1913 with standard deviation 0.1058.

took 962.04 seconds to run.