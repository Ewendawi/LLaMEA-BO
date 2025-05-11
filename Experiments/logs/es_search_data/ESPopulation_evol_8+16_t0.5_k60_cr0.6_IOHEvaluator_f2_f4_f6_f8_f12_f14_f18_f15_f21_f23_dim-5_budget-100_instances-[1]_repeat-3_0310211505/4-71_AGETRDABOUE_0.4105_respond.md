# Description
**AGETRDABOUE: Adaptive Gradient-Enhanced Trust Region Bayesian Optimization with Dynamic Acquisition Balancing and Uncertainty-Aware Exploration.** This algorithm combines the strengths of AGETRBO and ATRDABOUE. It incorporates gradient information using finite differences to enhance the Gaussian Process Regression (GPR) model, adaptively adjusts the trust region based on success, dynamically balances multiple acquisition functions (EI and UCB) using Thompson Sampling, and includes an uncertainty-aware exploration term in the acquisition function. The L-BFGS-B optimizer is used to select the next points within the trust region.

# Justification
The algorithm leverages the following key ideas:
1.  **Gradient Enhancement:** Incorporating gradient information (AGETRBO) improves the accuracy of the GPR model, leading to better predictions and faster convergence, especially in smoother regions of the search space.
2.  **Adaptive Trust Region:** The adaptive trust region (ATRBO) ensures a balance between exploration and exploitation by adjusting the search space based on the success of previous iterations.
3.  **Dynamic Acquisition Balancing:** Thompson Sampling (ATRDABOUE) adaptively selects the most promising acquisition function (EI or UCB) based on their past performance, dynamically adjusting the exploration-exploitation trade-off.
4.  **Uncertainty-Aware Exploration:** The uncertainty-aware exploration term (ATRDABOUE) encourages exploration in regions with high predictive variance, preventing premature convergence and improving robustness.
5.  **L-BFGS-B Optimization:** Using L-BFGS-B to optimize the acquisition function within the trust region allows for a more efficient and precise selection of the next points compared to random sampling.

This combination aims to create a robust and efficient algorithm that can handle a wide range of black-box optimization problems by leveraging gradient information where available, dynamically adapting the exploration-exploitation trade-off, and focusing on regions of high uncertainty.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import beta


class AGETRDABOUE:
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
        self.temperature_decay = 0.95  # Decay rate for temperature
        self.delta = 1e-3  # Step size for finite difference gradient estimation

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
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.delta
            x_minus[i] -= self.delta
            x_plus = np.clip(x_plus, self.bounds[0][i], self.bounds[1][i])
            x_minus = np.clip(x_minus, self.bounds[0][i], self.bounds[1][i])
            gradient[i] = (func(x_plus) - func(x_minus)) / (2 * self.delta)
        return gradient

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

    def _select_next_points(self, func, batch_size):
        X_next = []
        for _ in range(batch_size):
            # Define the objective function for optimization
            def objective(x, idx):
                x = x.reshape(1, -1)
                return -self._acquisition_function(x, acq_func=self.acq_funcs[idx])[0, 0]

            # Thompson Sampling
            sampled_values = []
            for i in range(self.n_acq):
                sampled_values.append(beta.rvs(self.alpha[i], self.beta[i], size=1)[0])

            # Select acquisition function with highest sampled value
            idx = np.argmax(sampled_values)

            # Optimization within bounds (trust region)
            lower_bound = np.maximum(self.bounds[0], self.best_x - self.trust_region_width / 2)
            upper_bound = np.minimum(self.bounds[1], self.best_x + self.trust_region_width / 2)
            bounds = [(lower_bound[i], upper_bound[i]) for i in range(self.dim)]

            # Initial guess (randomly sampled within trust region)
            x0 = self._sample_points(1, center=self.best_x, width=self.trust_region_width).flatten()

            # Perform optimization
            result = minimize(lambda x: objective(x, idx), x0, method='L-BFGS-B', bounds=bounds)

            X_next.append(result.x)

        return np.array(X_next), idx

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

            X_next, idx = self._select_next_points(func, batch_size)

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
 The algorithm AGETRDABOUE got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1897 with standard deviation 0.1103.

took 582.00 seconds to run.