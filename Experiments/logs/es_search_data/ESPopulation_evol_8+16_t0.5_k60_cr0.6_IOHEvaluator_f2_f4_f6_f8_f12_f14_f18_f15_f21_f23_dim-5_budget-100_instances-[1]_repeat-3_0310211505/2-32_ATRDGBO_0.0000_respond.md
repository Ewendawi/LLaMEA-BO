# Description
Adaptive Trust Region with Dynamic Acquisition Balancing and Gradient Estimation Bayesian Optimization (ATRDGBO) combines the adaptive trust region strategy of ATRBO, the dueling acquisition function selection of ATRDABO, and the gradient estimation of AGETRBO. It adaptively adjusts the trust region based on the success of previous iterations, uses a dueling mechanism to select between Expected Improvement (EI) and Upper Confidence Bound (UCB) acquisition functions, and incorporates gradient information, estimated using finite differences, into the Gaussian Process Regression (GPR) model. A dynamic acquisition balancing mechanism adjusts the weights of the EI and UCB acquisition functions based on their recent performance.

# Justification
This algorithm aims to improve upon ATRBO and ATRDABO by incorporating gradient information and dynamically balancing exploration and exploitation.
- Adaptive Trust Region: The adaptive trust region strategy from ATRBO is retained to focus the search in promising regions while ensuring exploration.
- Dueling Acquisition Functions: The dueling acquisition function selection from ATRDABO is used to dynamically balance exploration and exploitation.
- Gradient Estimation: Gradient information, estimated using finite differences, is incorporated into the GPR model to improve the accuracy of the surrogate model, especially in high-dimensional spaces. This is inspired by AGETRBO, but implemented in a more computationally efficient way by only estimating gradients periodically.
- Dynamic Acquisition Balancing: The weights of the EI and UCB acquisition functions are dynamically adjusted based on their recent performance, allowing the algorithm to adapt to the landscape of the optimization problem. This is implemented using an exponentially weighted moving average of the rewards obtained by each acquisition function.
- Computational Efficiency: The gradient estimation is performed periodically, rather than at every iteration, to reduce the computational cost. The batch size is dynamically adjusted based on the dimensionality of the problem to balance exploration and exploitation.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

class ATRDGBO:
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
        self.gradient_estimation_frequency = 5 # Estimate gradients every 5 iterations
        self.finite_difference_epsilon = 1e-3
        self.reward_history = {func: [] for func in self.acq_funcs}
        self.ewma_alpha = 0.1 # Exponentially weighted moving average factor

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

    def _estimate_gradient(self, func, x):
        # Estimate gradient using finite differences
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.finite_difference_epsilon
            x_minus[i] -= self.finite_difference_epsilon
            gradient[i] = (func(x_plus) - func(x_minus)) / (2 * self.finite_difference_epsilon)
        return gradient

    def _acquisition_function(self, X, acq_func='ei', gradient=None):
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

            if gradient is not None:
                ei += 0.01 * np.abs(np.dot(X - self.best_x, gradient.T)) # Encourage alignment with gradient

            return ei

        elif acq_func == 'ucb':
            ucb = mu + 2 * sigma
            return ucb

        else:
            raise ValueError(f"Unknown acquisition function: {acq_func}")

    def _select_next_points(self, func, batch_size, iteration):
        n_candidates = max(2000, batch_size * 200)
        X_cand = self._sample_points(n_candidates, center=self.best_x, width=self.trust_region_width)

        # Estimate gradient periodically
        if iteration % self.gradient_estimation_frequency == 0:
            gradient = self._estimate_gradient(func, self.best_x)
        else:
            gradient = None

        acq_values = np.zeros((self.n_acq, n_candidates))
        for i, acq_func in enumerate(self.acq_funcs):
            acq_values[i, :] = self._acquisition_function(X_cand, acq_func, gradient).flatten()

        # Dueling acquisition function selection
        idx = np.random.choice(self.n_acq, 1, p=self.weights)[0]

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
        reward = self.y.min() - new_y.min()
        acq_func = self.acq_funcs[idx]
        self.reward_history[acq_func].append(reward)

        # Update EWMA reward
        if len(self.reward_history[acq_func]) > 1:
            ewma_reward = self.ewma_alpha * reward + (1 - self.ewma_alpha) * self.reward_history[acq_func][-2]
        else:
            ewma_reward = reward

        # Update weights based on EWMA reward
        for i in range(self.n_acq):
            if i != idx:
                prob = 1 / (1 + np.exp((ewma_reward) / self.gamma))
                self.weights[i] *= prob
                self.weights[idx] *= (1 - prob)

        self.weights += 1e-9  # Add a small constant to avoid NaN errors
        self.weights /= self.weights.sum()

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_x = X_init[np.argmin(y_init)]
        self.best_y = np.min(y_init)

        batch_size = max(1, self.dim // 2)
        iteration = 0
        while self.n_evals < self.budget:
            self.model = self._fit_model(self.X, self.y)

            X_next, idx = self._select_next_points(func, batch_size, iteration)

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
            iteration += 1

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<ATRDGBO>", line 154, in __call__
 154->             X_next, idx = self._select_next_points(func, batch_size, iteration)
  File "<ATRDGBO>", line 100, in _select_next_points
 100->             acq_values[i, :] = self._acquisition_function(X_cand, acq_func, gradient).flatten()
  File "<ATRDGBO>", line 77, in _acquisition_function
  75 | 
  76 |             if gradient is not None:
  77->                 ei += 0.01 * np.abs(np.dot(X - self.best_x, gradient.T)) # Encourage alignment with gradient
  78 | 
  79 |             return ei
ValueError: non-broadcastable output operand with shape (2000,1) doesn't match the broadcast shape (2000,2000)
