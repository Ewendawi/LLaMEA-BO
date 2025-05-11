# Description
LAGBO: Landscape-Aware Gaussian Bayesian Optimization. This algorithm combines the landscape analysis from LGBO with the surrogate model averaging from SMABO. It uses two Gaussian Process Regression (GPR) models with different kernels (RBF and Matern) and averages their predictions. The exploration-exploitation trade-off is dynamically adjusted based on landscape analysis, similar to LGBO. A dynamic temperature parameter is used in the Expected Improvement (EI) acquisition function to control exploration.

# Justification
This algorithm aims to leverage the strengths of both SMABO and LGBO. SMABO uses model averaging to improve the robustness of the surrogate model, while LGBO adapts the exploration-exploitation trade-off based on the landscape. Combining these two approaches should lead to a more robust and efficient algorithm. The landscape analysis helps to guide the search, while model averaging reduces the risk of overfitting to a single model. The temperature parameter in EI allows for further control over exploration. The clipping of EI values, as in LGBO, prevents potential NaN issues.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances

class LAGBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(10*dim, self.budget//5)
        self.temperature = 1.0
        self.landscape_correlation = 0.0
        self.smoothness_threshold = 0.5

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

    def _analyze_landscape(self):
        if self.X is None or self.y is None or len(self.X) < 2:
            return 0.0

        distances = pairwise_distances(self.X)
        value_differences = np.abs(self.y - self.y.T)

        distances = distances.flatten()
        value_differences = value_differences.flatten()
        indices = np.arange(len(distances))
        distances = distances[indices % (len(self.X) + 1) != 0]
        value_differences = value_differences[indices % (len(self.X) + 1) != 0]

        correlation, _ = pearsonr(distances, value_differences)
        return correlation if not np.isnan(correlation) else 0.0

    def _acquisition_function(self, X):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu_rbf, sigma_rbf = self.model_rbf.predict(X, return_std=True)
        mu_matern, sigma_matern = self.model_matern.predict(X, return_std=True)

        mu_rbf = mu_rbf.reshape(-1, 1)
        sigma_rbf = sigma_rbf.reshape(-1, 1)
        mu_matern = mu_matern.reshape(-1, 1)
        sigma_matern = sigma_matern.reshape(-1, 1)
        
        mu = (mu_rbf + mu_matern) / 2.0
        sigma = (sigma_rbf + sigma_matern) / 2.0

        best = np.min(self.y)
        imp = best - mu
        Z = imp / (self.temperature * sigma + 1e-9)
        ei = imp * norm.cdf(Z) + self.temperature * sigma * norm.pdf(Z)
        ei = np.clip(ei, 0, 1e10)
        ei[sigma <= 1e-6] = 0.0

        return ei

    def _select_next_points(self, batch_size):
        n_candidates = max(2000, batch_size * 200)
        X_cand = self._sample_points(n_candidates)

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

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            self.model_rbf, self.model_matern = self._fit_model(self.X, self.y)

            self.landscape_correlation = self._analyze_landscape()

            if self.landscape_correlation > self.smoothness_threshold:
                self.temperature = max(0.1, self.temperature * 0.9)
            else:
                self.temperature = min(2.0, self.temperature * 1.1)

            X_next = self._select_next_points(batch_size)

            y_next = self._evaluate_points(func, X_next)

            self._update_eval_points(X_next, y_next)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Feedback
 The algorithm LAGBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1620 with standard deviation 0.0978.

took 315.00 seconds to run.