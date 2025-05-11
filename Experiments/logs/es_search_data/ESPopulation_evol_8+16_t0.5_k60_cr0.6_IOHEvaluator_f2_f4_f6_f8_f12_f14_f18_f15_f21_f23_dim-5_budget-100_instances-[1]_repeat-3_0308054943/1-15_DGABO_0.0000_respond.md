# Description
**DGA-BO: Diversity and Gradient Aware Bayesian Optimization:** This algorithm combines the diversity-enhancing features of DEBO with the gradient-aware exploration of AGABO, while addressing the computational inefficiency of AGABO and potential over-exploration of DEBO. It uses a Gaussian Process Regression (GPR) model with a Matern kernel for surrogate modeling. The acquisition function is a weighted combination of Expected Improvement (EI), a distance-based diversity term, and a gradient-based exploration term. To improve computational efficiency, the gradient is estimated using a more efficient finite difference method. The diversity term is adaptively weighted based on the iteration number to prioritize exploration early on and exploitation later. A Sobol sequence is used for initial exploration, and a local search refinement step is added to enhance exploitation.

# Justification
The algorithm is designed to address the following:

1.  **Exploration-Exploitation Balance:** Combines Expected Improvement (EI) for exploitation, a diversity term for exploration of unexplored regions, and a gradient-based term for exploration of promising regions with steep descent.
2.  **Computational Efficiency:** AGABO was computationally expensive due to the gradient calculation. This version uses a more efficient gradient approximation.
3.  **Adaptive Diversity:** The diversity weight is adjusted over time, starting high for initial exploration and decreasing as the algorithm converges to focus on exploitation.
4.  **Kernel Choice:** Matern kernel is chosen for GPR as it is more flexible than RBF and can better capture the characteristics of various objective functions.
5.  **Local Refinement:** A local search step is added to fine-tune the solution after the Bayesian Optimization loop, improving the final result.
6.  **Initial Exploration:** Sobol sequence is used for initial exploration to ensure good coverage of the search space.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import minimize
import warnings

class DGABO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.diversity_weight = 0.1
        self.gradient_weight = 0.01
        self.kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        self.best_x = None
        self.best_y = float('inf')
        self.iteration = 0  # Add iteration counter

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, seed=42)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        try:
            model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5, alpha=1e-5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)
            self.kernel = model.kernel_
            return model
        except Exception as e:
            print(f"GP fitting failed: {e}. Returning None.")
            return None

    def _acquisition_function(self, X, model):
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if self.best_y is None:
            ei = np.zeros_like(mu)
        else:
            imp = self.best_y - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0

        # Adaptive diversity weight
        adaptive_diversity_weight = self.diversity_weight * np.exp(-self.iteration / 50)

        # Diversity term
        if self.X is not None:
            distances = np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2)
            min_distances = np.min(distances, axis=1).reshape(-1, 1)
            ei = ei + adaptive_diversity_weight * min_distances

        # Gradient-based exploration term
        if self.X is not None:
            dmu_dx = self._approximate_gradient(X, model)
            gradient_norm = np.linalg.norm(dmu_dx, axis=1, keepdims=True)
            ei = ei + self.gradient_weight * gradient_norm

        return ei

    def _approximate_gradient(self, X, model):
        # Efficient gradient approximation using finite differences
        delta = 1e-4
        X_prime = X + np.diag([delta] * self.dim)
        X_prime = np.clip(X_prime, self.bounds[0], self.bounds[1])  # Clip to bounds
        mu_prime, _ = model.predict(X_prime, return_std=True)
        mu, _ = model.predict(X, return_std=True)
        dmu_dx = (mu_prime - mu) / delta
        return dmu_dx

    def _select_next_points(self, batch_size, model):
        candidate_points = self._sample_points(100 * batch_size)
        acquisition_values = self._acquisition_function(candidate_points, model)
        indices = np.argsort(-acquisition_values.flatten())[:batch_size]
        selected_points = candidate_points[indices]
        return selected_points

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

    def _local_search(self, func, x0):
        # Local search refinement
        def obj(x):
            return func(x)

        res = minimize(obj, x0, method='L-BFGS-B', bounds=[(-5, 5)] * self.dim)
        return res.fun, res.x
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        batch_size = 5
        while self.n_evals < self.budget:
            self.iteration += 1
            model = self._fit_model(self.X, self.y)
            if model is None:
                print("GP model fitting failed. Returning current best.")
                return self.best_y, self.best_x

            next_X = self._select_next_points(batch_size, model)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
        
        # Local search refinement
        best_y, best_x = self._local_search(func, self.best_x)
        
        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<DGABO>", line 131, in __call__
 131->             next_X = self._select_next_points(batch_size, model)
  File "<DGABO>", line 87, in _select_next_points
  87->         acquisition_values = self._acquisition_function(candidate_points, model)
  File "<DGABO>", line 69, in _acquisition_function
  69->             dmu_dx = self._approximate_gradient(X, model)
  File "<DGABO>", line 78, in _approximate_gradient
  76 |         # Efficient gradient approximation using finite differences
  77 |         delta = 1e-4
  78->         X_prime = X + np.diag([delta] * self.dim)
  79 |         X_prime = np.clip(X_prime, self.bounds[0], self.bounds[1])  # Clip to bounds
  80 |         mu_prime, _ = model.predict(X_prime, return_std=True)
ValueError: operands could not be broadcast together with shapes (500,5) (5,5) 
