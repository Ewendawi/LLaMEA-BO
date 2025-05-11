# Description
Adaptive Ensemble Bayesian Optimization with Distance-Based Exploration (AEBO-DDE) is a novel metaheuristic algorithm that combines the strengths of surrogate model averaging and diversity-enhanced exploration. It uses an ensemble of Gaussian Process Regression (GPR) models with different kernels, similar to SMABO, to improve the robustness of predictions. To promote diversity, it incorporates a distance-based exploration term in the acquisition function, inspired by DEBO. The algorithm adaptively adjusts the weights of the ensemble models based on their recent performance. Instead of K-means clustering, it uses a simpler top-k selection with a diversity promoting term to select the next points, avoiding the NaN errors encountered in EHBBO.

# Justification
The algorithm builds upon SMABO by incorporating adaptive model weighting and distance-based exploration. Adaptive model weighting allows the algorithm to dynamically adjust the contribution of each GPR model in the ensemble based on its predictive accuracy, leading to better overall performance. The distance-based exploration encourages the algorithm to explore regions of the search space that are far from existing points, promoting diversity and preventing premature convergence. The top-k selection strategy is computationally efficient and avoids the potential issues associated with K-means clustering. The combination of these features results in a robust and efficient Bayesian optimization algorithm that can handle a wide range of black-box optimization problems.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from sklearn.metrics.pairwise import euclidean_distances


class AEBO_DDE:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(10 * dim, self.budget // 5)
        self.model_weights = np.array([0.5, 0.5])  # Initial weights for RBF and Matern kernels
        self.weight_decay = 0.95
        self.exploration_weight = 0.1  # Weight for distance-based exploration

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        kernel_rbf = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        kernel_matern = ConstantKernel(1.0) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)

        gp_rbf = GaussianProcessRegressor(kernel=kernel_rbf, n_restarts_optimizer=5)
        gp_matern = GaussianProcessRegressor(kernel=kernel_matern, n_restarts_optimizer=5)

        gp_rbf.fit(X, y)
        gp_matern.fit(X, y)

        return gp_rbf, gp_matern

    def _acquisition_function(self, X):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu_rbf, sigma_rbf = self.model_rbf.predict(X, return_std=True)
        mu_matern, sigma_matern = self.model_matern.predict(X, return_std=True)

        mu_rbf = mu_rbf.reshape(-1, 1)
        sigma_rbf = sigma_rbf.reshape(-1, 1)
        mu_matern = mu_matern.reshape(-1, 1)
        sigma_matern = sigma_matern.reshape(-1, 1)

        # Adaptive model averaging
        mu = self.model_weights[0] * mu_rbf + self.model_weights[1] * mu_matern
        sigma = self.model_weights[0] * sigma_rbf + self.model_weights[1] * sigma_matern

        # Expected Improvement
        best = np.min(self.y)
        imp = best - mu
        Z = imp / (sigma + 1e-9)
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Distance-based exploration
        if self.X is not None:
            distances = np.min(euclidean_distances(X, self.X), axis=1, keepdims=True)
            diversity_term = distances / np.max(distances)  # Normalize distances
            acq = ei + self.exploration_weight * diversity_term
        else:
            acq = ei

        return acq

    def _select_next_points(self, batch_size):
        n_candidates = max(2000, batch_size * 200)
        X_cand = self._sample_points(n_candidates)
        acq_values = self._acquisition_function(X_cand)

        # Select top-k points
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

        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            self.model_rbf, self.model_matern = self._fit_model(self.X, self.y)

            # Update model weights based on recent performance
            if len(self.y) > self.n_init:
                n_recent = min(10, len(self.y) - self.n_init)
                y_recent = self.y[-n_recent:]
                X_recent = self.X[-n_recent:]

                mu_rbf, _ = self.model_rbf.predict(X_recent, return_std=True)
                mu_matern, _ = self.model_matern.predict(X_recent, return_std=True)

                rbf_error = np.mean((mu_rbf.flatten() - y_recent.flatten())**2)
                matern_error = np.mean((mu_matern.flatten() - y_recent.flatten())**2)

                if rbf_error < matern_error:
                    self.model_weights[0] *= (1 + self.weight_decay)
                    self.model_weights[1] *= (1 - self.weight_decay)
                else:
                    self.model_weights[0] *= (1 - self.weight_decay)
                    self.model_weights[1] *= (1 + self.weight_decay)

                self.model_weights /= np.sum(self.model_weights)  # Normalize weights

            X_next = self._select_next_points(batch_size)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Feedback
 The algorithm AEBO_DDE got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1618 with standard deviation 0.0963.

took 345.56 seconds to run.