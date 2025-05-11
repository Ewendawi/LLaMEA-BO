# Description
DEDSPBO: Diversity Enhanced Dynamic Stochastic Patch Bayesian Optimization. This algorithm integrates the diversity-enhancing strategy from DEHBBO with the stochastic patch dimension sampling from SPBO. It aims to balance exploration and exploitation by focusing on promising regions while maintaining diversity and addressing the dimensionality curse. It corrects the error in SPBO by ensuring the GP model is used with the correct input dimensions. It incorporates dynamic patch size and Hall of Fame diversity maintenance.

# Justification
This algorithm builds upon DEHBBO's diversity mechanism and SPBO's dimension reduction technique. The SPBO error was due to inconsistent dimensions between the input `X` to the `predict` function of the GP model and the dimensions the GP model was trained on. This is corrected by applying the patch indices *before* sampling candidate points.

The Hall of Fame, inherited from DEHBBO, encourages diversity by penalizing points close to existing members. The dynamic batch size and patch size strategies adapt exploration/exploitation based on the remaining budget. L-BFGS-B optimization is used for local search within the selected patch to efficiently leverage the acquisition function. Combining these aspects aims to create a robust and efficient BO algorithm, especially suitable for high-dimensional problems with limited budgets.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class DEDSPBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(20 * dim, self.budget // 5)
        self.best_y = float('inf')
        self.best_x = None
        self.hall_of_fame_X = []
        self.hall_of_fame_y = []
        self.hall_of_fame_size = max(5, dim // 2)
        self.diversity_threshold = 0.5

    def _sample_points(self, n_points, center=None, radius=None):
        if center is None:
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            points = sampler.random(n=n_points)
            return qmc.scale(points, self.bounds[0], self.bounds[1])
        else:
            points = np.random.normal(loc=center, scale=radius / 3, size=(n_points, self.dim))
            points = np.clip(points, self.bounds[0], self.bounds[1])
            return points

    def _fit_model(self, X, y):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp, patch_indices):
        mu, sigma = gp.predict(X[:, patch_indices], return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        thompson_samples = np.random.normal(mu, sigma)
        return thompson_samples

    def _select_next_points(self, batch_size, gp):
        next_X = []
        for _ in range(batch_size):
            remaining_evals = self.budget - self.n_evals
            patch_size = max(1, min(self.dim, int(self.dim * remaining_evals / self.budget) + 1))
            patch_indices = np.random.choice(self.dim, patch_size, replace=False)

            # Sample candidate point within the *reduced* dimension space
            candidate_x_reduced = self._sample_points(1)[:, patch_indices]

            # Optimization of acquisition function within the patch using L-BFGS-B
            res = minimize(lambda x: -self._acquisition_function(x.reshape(1, -1), gp, patch_indices),
                           candidate_x_reduced.flatten(),
                           bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in patch_indices],
                           method='L-BFGS-B')

            candidate_x_patched = np.zeros(self.dim)
            candidate_x_patched[patch_indices] = res.x
            next_X.append(candidate_x_patched)

        return np.array(next_X)

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

        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

            if not self.hall_of_fame_X:
                self.hall_of_fame_X.append(self.best_x)
                self.hall_of_fame_y.append(self.best_y)
            else:
                distances = np.array([np.linalg.norm(self.best_x - hof_x) for hof_x in self.hall_of_fame_X])
                if np.min(distances) > self.diversity_threshold:
                    self.hall_of_fame_X.append(self.best_x)
                    self.hall_of_fame_y.append(self.best_y)
                    if len(self.hall_of_fame_X) > self.hall_of_fame_size:
                        worst_idx = np.argmax(self.hall_of_fame_y)
                        self.hall_of_fame_X.pop(worst_idx)
                        self.hall_of_fame_y.pop(worst_idx)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        while self.n_evals < self.budget:
            gp = self._fit_model(self.X, self.y)

            remaining_evals = self.budget - self.n_evals
            batch_size = min(max(1, int(remaining_evals / (self.dim * 0.1))), 20)

            next_X = self._select_next_points(batch_size, gp)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<DEDSPBO>", line 114, in __call__
 114->             next_X = self._select_next_points(batch_size, gp)
  File "<DEDSPBO>", line 61, in _select_next_points
  61->             res = minimize(lambda x: -self._acquisition_function(x.reshape(1, -1), gp, patch_indices),
  File "<DEDSPBO>", line 61, in <lambda>
  61->             res = minimize(lambda x: -self._acquisition_function(x.reshape(1, -1), gp, patch_indices),
  File "<DEDSPBO>", line 43, in _acquisition_function
  41 | 
  42 |     def _acquisition_function(self, X, gp, patch_indices):
  43->         mu, sigma = gp.predict(X[:, patch_indices], return_std=True)
  44 |         mu = mu.reshape(-1, 1)
  45 |         sigma = sigma.reshape(-1, 1)
IndexError: index 4 is out of bounds for axis 1 with size 4
