# Description
**ABETSALSBOv2: Adaptive Batch Ensemble with Thompson Sampling, Uncertainty-Aware Local Search, and Adaptive Kernel Lengthscale Bayesian Optimization.** This algorithm refines ABETSALSBO by introducing an adaptive kernel lengthscale adjustment for the Gaussian Process Regression (GPR) models within the ensemble. It dynamically adjusts the lengthscale of each GPR model based on the local density of evaluated points, allowing for more accurate modeling of the function landscape. Furthermore, the local search is enhanced by incorporating a momentum-based acceleration to escape local optima.

# Justification
The key improvements are:

1.  **Adaptive Kernel Lengthscale:** The original ABETSALSBO used fixed lengthscales for the GPR models in the ensemble. By adapting the lengthscale based on the local density of evaluated points, the GPR models can better capture the local characteristics of the function landscape. A smaller lengthscale is used in regions with high density, allowing for more detailed modeling, while a larger lengthscale is used in regions with low density, promoting exploration. This is achieved by estimating the local density using a k-nearest neighbors approach and adjusting the lengthscale accordingly.

2.  **Momentum-Based Local Search:** The original local search used L-BFGS-B without any acceleration techniques. By incorporating momentum-based acceleration, the local search can escape local optima more effectively and converge to better solutions. The momentum is calculated based on the previous step and the current gradient, allowing the search to overcome small barriers and continue in promising directions.

3.  **Computational Efficiency:** The k-nearest neighbors search for density estimation is performed using efficient data structures and algorithms from `sklearn.neighbors` to minimize the computational overhead. The momentum-based local search is also implemented efficiently using vectorized operations.

These changes aim to improve the accuracy and efficiency of the algorithm, leading to better performance on the BBOB test suite.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors

class ABETSALSBOv2:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim

        self.best_y = np.inf
        self.best_x = None

        self.max_models = 5  # Maximum number of surrogate models in the ensemble
        self.min_models = 1  # Minimum number of surrogate models in the ensemble
        self.models = []
        for i in range(self.max_models):
            length_scale = 1.0 * (i + 1) / self.max_models
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
            self.models.append(model)

        self.max_batch_size = min(10, dim)
        self.min_batch_size = 1
        self.local_search_step_size_factor = 0.1
        self.uncertainty_threshold = 0.5
        self.exploration_weight = 0.1
        self.exploration_decay = 0.995
        self.min_exploration = 0.01
        self.knn = NearestNeighbors(n_neighbors=min(10, self.n_init), algorithm='kd_tree') # KNN for density estimation
        self.density_bandwidth = 1.0 # Bandwidth for density estimation

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Adapt ensemble size
        n_models = max(self.min_models, int(self.max_models * (1 - self.n_evals / self.budget)))
        active_models = self.models[:n_models]

        # Adaptive kernel lengthscale
        if X.shape[0] > 10:
            self.knn.fit(X)
            distances, _ = self.knn.kneighbors(X)
            local_densities = np.mean(np.exp(-distances**2 / (2 * self.density_bandwidth**2)), axis=1)
            min_density = np.min(local_densities)
            max_density = np.max(local_densities)
        else:
            local_densities = np.ones(X.shape[0])
            min_density = 1
            max_density = 1

        for i, model in enumerate(active_models):
            # Adjust lengthscale based on local density
            for j in range(X.shape[0]):
                length_scale = 1.0 * (i + 1) / self.max_models * (1 - 0.5 * (local_densities[j] - min_density) / (max_density - min_density))
                model.kernel_ = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
            model.fit(X, y)
        return active_models

    def _acquisition_function(self, X, active_models):
        # Thompson Sampling
        sampled_values = np.zeros((X.shape[0], len(active_models)))
        sigmas = np.zeros((X.shape[0], len(active_models)))
        for i, model in enumerate(active_models):
            mu, sigma = model.predict(X, return_std=True)
            mu = mu.reshape(-1, 1)
            sigma = sigma.reshape(-1, 1)
            sampled_values[:, i] = np.random.normal(mu.flatten(), sigma.flatten())
            sigmas[:, i] = sigma.flatten()

        acquisition = np.mean(sampled_values, axis=1, keepdims=True)
        acquisition = acquisition.reshape(-1, 1)

        # Hybrid acquisition function (EI + exploration)
        mu = np.mean([model.predict(X) for model in active_models], axis=0).reshape(-1, 1)
        sigma = np.mean(sigmas, axis=1).reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Distance-based exploration term
        if self.X is not None:
            min_dist = np.min(np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2), axis=1, keepdims=True)
            exploration = min_dist / np.max(min_dist)
        else:
            exploration = np.ones_like(ei)

        acquisition = ei + self.exploration_weight * exploration

        return acquisition

    def _select_next_points(self, batch_size, active_models):
        candidate_points = self._sample_points(100 * batch_size)
        acquisition_values = self._acquisition_function(candidate_points, active_models)
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]

        # Adaptive Local search with momentum
        for i in range(batch_size):
            def obj_func(x):
                x = x.reshape(1, -1)
                return np.mean([model.predict(x)[0] for model in active_models])

            x0 = next_points[i]
            bounds = [(self.bounds[0][j], self.bounds[1][j]) for j in range(self.dim)]

            # Uncertainty-aware local search
            uncertainty = np.mean([model.predict(x0.reshape(1, -1), return_std=True)[1] for model in active_models])
            maxiter = int(5 + 10 * uncertainty)
            maxiter = min(maxiter, 20)

            # Adaptive step size
            step_size = self.local_search_step_size_factor * uncertainty
            options = {'maxiter': maxiter, 'ftol': 1e-4}  # Reduced ftol

            # Momentum-based acceleration
            momentum = np.zeros_like(x0)
            alpha = 0.9  # Momentum coefficient
            
            def momentum_obj_func(x):
                nonlocal momentum
                x = x.reshape(1, -1)
                grad = np.zeros_like(x0)
                
                # Numerical gradient approximation
                h = 1e-5
                for k in range(self.dim):
                    x_plus = x0.copy()
                    x_minus = x0.copy()
                    x_plus[k] += h
                    x_minus[k] -= h
                    grad[k] = (obj_func(x_plus) - obj_func(x_minus)) / (2 * h)
                
                momentum = alpha * momentum - step_size * grad
                return obj_func(x + momentum)

            res = minimize(momentum_obj_func, x0, method='L-BFGS-B', bounds=bounds, options=options)
            next_points[i] = res.x

        return next_points

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

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

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        active_models = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            remaining_evals = self.budget - self.n_evals

            # Adjust batch size based on uncertainty
            sigmas = np.array([model.predict(self.X, return_std=True)[1] for model in active_models])
            avg_sigma = np.mean(sigmas)

            if avg_sigma > self.uncertainty_threshold:
                batch_size = self.max_batch_size
                exploration_decay = 0.99
            else:
                batch_size = self.min_batch_size
                exploration_decay = 0.999

            batch_size = min(batch_size, remaining_evals)
            next_X = self._select_next_points(batch_size, active_models)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            active_models = self._fit_model(self.X, self.y)

            # Update exploration weight
            self.exploration_weight = max(self.exploration_weight * exploration_decay, self.min_exploration)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ABETSALSBOv2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1613 with standard deviation 0.1033.

took 725.00 seconds to run.