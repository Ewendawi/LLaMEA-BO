# Description
**Adaptive Trust Region with Dueling Acquisition and Bayesian Model Averaging (ATRDABOMA)**

This algorithm enhances the ATRDABO framework by incorporating Bayesian Model Averaging (BMA) over multiple Gaussian Process Regression (GPR) models with different kernels. This improves the robustness and accuracy of the surrogate model, leading to a better exploration-exploitation balance. Additionally, the acquisition function weights are updated using a more stable and informative method based on the actual improvement achieved by each acquisition function.

# Justification
*   **Bayesian Model Averaging:** Using an ensemble of GPR models with different kernels (RBF and Matern) allows the algorithm to capture different characteristics of the objective function landscape. Averaging the predictions from these models provides a more robust and accurate surrogate model, reducing the risk of overfitting to a specific kernel.

*   **Improved Weight Update:** The weight update mechanism is refined to consider the actual improvement achieved by each acquisition function. This is done by calculating the average improvement obtained when using each acquisition function and updating the weights proportionally. This provides a more direct and informative way to adjust the weights compared to the previous approach, leading to a better selection of acquisition functions.

*   **Adaptive Trust Region:** The adaptive trust region strategy from ATRBO is retained, allowing the algorithm to focus the search within a dynamically adjusted region based on the success of previous iterations.

*   **Computational Efficiency:** The algorithm maintains computational efficiency by using a relatively small ensemble of GPR models and a simple weight update mechanism. The candidate points are still sampled efficiently using Latin Hypercube Sampling.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

class ATRDABOMA:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(10*dim, self.budget//5)
        self.trust_region_width = 2.0
        self.success_threshold = 0.1
        self.best_y = np.inf
        self.best_x = None
        self.acq_funcs = ['ei', 'ucb']
        self.n_acq = len(self.acq_funcs)
        self.weights = np.ones(self.n_acq) / self.n_acq
        self.gamma = 0.1
        self.n_models = 2  # Number of GPR models in the ensemble
        self.models = []
        for _ in range(self.n_models):
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))  # RBF Kernel
            self.models.append(GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5))

        self.improvement_history = {acq_func: [] for acq_func in self.acq_funcs}


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
        for model in self.models:
            model.fit(X, y)

    def _predict(self, X):
        mu_list = []
        sigma_list = []
        for model in self.models:
            mu, sigma = model.predict(X, return_std=True)
            mu_list.append(mu.reshape(-1, 1))
            sigma_list.append(sigma.reshape(-1, 1))

        mu = np.mean(np.concatenate(mu_list, axis=1), axis=1).reshape(-1, 1)
        sigma = np.mean(np.concatenate(sigma_list, axis=1), axis=1).reshape(-1, 1)
        return mu, sigma


    def _acquisition_function(self, X, acq_func='ei'):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu, sigma = self._predict(X)

        if acq_func == 'ei':
            best = np.min(self.y)
            imp = best - mu
            Z = imp / (sigma + 1e-9)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0
            return ei

        elif acq_func == 'ucb':
            ucb = mu + 2 * sigma
            return ucb

        else:
            raise ValueError(f"Unknown acquisition function: {acq_func}")

    def _select_next_points(self, batch_size):
        n_candidates = max(2000, batch_size * 200)
        X_cand = self._sample_points(n_candidates, center=self.best_x, width=self.trust_region_width)

        acq_values = np.zeros((self.n_acq, n_candidates))
        for i, acq_func in enumerate(self.acq_funcs):
            acq_values[i, :] = self._acquisition_function(X_cand, acq_func).flatten()

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
        acq_func = self.acq_funcs[idx]
        improvement = self.y.min() - new_y.min()
        self.improvement_history[acq_func].append(improvement)

        # Calculate average improvement for each acquisition function
        avg_improvements = {}
        for acq_func in self.acq_funcs:
            if self.improvement_history[acq_func]:
                avg_improvements[acq_func] = np.mean(self.improvement_history[acq_func])
            else:
                avg_improvements[acq_func] = 0.0  # Avoid division by zero

        # Update weights based on average improvement
        total_improvement = sum(avg_improvements.values())
        if total_improvement > 0:
            for i, acq_func in enumerate(self.acq_funcs):
                self.weights[i] = avg_improvements[acq_func] / total_improvement
        else:
            # If no improvement, keep weights uniform
            self.weights = np.ones(self.n_acq) / self.n_acq


        self.weights += 1e-9  # Add a small constant to avoid NaN errors
        self.weights /= self.weights.sum()

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_x = X_init[np.argmin(y_init)]
        self.best_y = np.min(y_init)

        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            self._fit_model(self.X, self.y)

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
 The algorithm ATRDABOMA got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1873 with standard deviation 0.1177.

took 1946.80 seconds to run.