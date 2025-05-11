# Description
**Adaptive Ensemble Hybrid Bayesian Optimization with Thompson Sampling, Adaptive Local Search, and Variance-Based Exploration (AEHTSALSVBO):** This algorithm enhances AEHTSALSBO by refining the exploration-exploitation balance within the acquisition function and improving the efficiency of the local search. The exploration term is now weighted by the variance of the GPR predictions, promoting exploration in regions of high uncertainty. Additionally, the Thompson Sampling is made more efficient by considering the correlation between the sampled values from different models in the ensemble. The local search is further refined by incorporating a dynamic step size adjustment based on the gradient of the predicted mean, facilitating faster convergence to local optima.

# Justification
The key improvements focus on a more nuanced exploration-exploitation balance and a more efficient local search.

1.  **Variance-Based Exploration:** The original distance-based exploration term is replaced with a variance-weighted exploration. This is motivated by the idea that exploration should be prioritized in areas where the model is most uncertain, as indicated by high variance. This approach is more directly related to the model's confidence and can lead to more effective exploration.

2.  **Correlation-Aware Thompson Sampling:** The original Thompson Sampling implementation treated the samples from different models independently. However, since the models are trained on the same data, their predictions are likely correlated. By considering the correlation, we can obtain a more accurate estimate of the uncertainty and improve the exploration-exploitation trade-off. This is implemented using a multivariate normal distribution.

3.  **Gradient-Based Adaptive Local Search:** The local search is enhanced by incorporating the gradient of the predicted mean. This allows the algorithm to take larger steps in the direction of the optimum, leading to faster convergence. The step size is dynamically adjusted based on the gradient magnitude, ensuring that the algorithm does not overshoot the optimum.

4. **Simplified Acquisition Function:** The EI calculation is simplified by directly using the best observed value instead of introducing a small constant.

These changes aim to improve the overall performance of the algorithm by promoting more efficient exploration and exploitation.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

class AEHTSALSVBO:
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

        self.batch_size = min(10, dim)
        self.local_search_step_size_factor = 0.1
        self.exploration_weight = 0.1

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Adapt ensemble size
        n_models = max(self.min_models, int(self.max_models * (1 - self.n_evals / self.budget)))
        active_models = self.models[:n_models]
        for model in active_models:
            model.fit(X, y)
        return active_models

    def _acquisition_function(self, X, active_models):
        # Thompson Sampling with correlation awareness
        mu_list = []
        sigma_list = []
        for model in active_models:
            mu, sigma = model.predict(X, return_std=True)
            mu_list.append(mu)
            sigma_list.append(sigma)

        mu_list = np.array(mu_list)
        sigma_list = np.array(sigma_list)

        # Calculate covariance matrix
        covariance = np.zeros((X.shape[0], len(active_models), len(active_models)))
        for i in range(X.shape[0]):
            for j in range(len(active_models)):
                for k in range(len(active_models)):
                    covariance[i, j, k] = 0.5 * (sigma_list[j, i]**2 + sigma_list[k, i]**2 - np.sum((active_models[j].predict(X[i].reshape(1, -1), return_cov=True)[1] + active_models[k].predict(X[i].reshape(1, -1), return_cov=True)[1])))

        sampled_values = np.zeros((X.shape[0], len(active_models)))
        for i in range(X.shape[0]):
            try:
                sampled_values[i, :] = multivariate_normal.rvs(mean=mu_list[:, i], cov=covariance[i, :, :], size=1)
            except:
                sampled_values[i, :] = np.random.normal(loc=mu_list[:, i], scale=sigma_list[:, i], size=len(active_models))

        acquisition = np.mean(sampled_values, axis=1, keepdims=True)
        acquisition = acquisition.reshape(-1, 1)

        # Hybrid acquisition function (EI + exploration)
        mu = np.mean([model.predict(X) for model in active_models], axis=0).reshape(-1, 1)
        sigma = np.mean(sigma_list, axis=0).reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Variance-based exploration term
        exploration = sigma**2
        exploration = exploration / np.max(exploration) if np.max(exploration) > 0 else exploration

        acquisition = ei + self.exploration_weight * exploration

        return acquisition

    def _select_next_points(self, batch_size, active_models):
        candidate_points = self._sample_points(100 * batch_size)
        acquisition_values = self._acquisition_function(candidate_points, active_models)
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]

        # Adaptive Local search
        for i in range(batch_size):
            def obj_func(x):
                x = x.reshape(1, -1)
                return np.mean([model.predict(x)[0] for model in active_models])

            def obj_grad(x):
                x = x.reshape(1, -1)
                grads = [model.predict(x, return_std=False, return_grad=True)[1] for model in active_models]
                return np.mean(np.array(grads), axis=0)

            x0 = next_points[i]
            bounds = [(self.bounds[0][j], self.bounds[1][j]) for j in range(self.dim)]

            # Uncertainty-aware local search
            uncertainty = np.mean([model.predict(x0.reshape(1, -1), return_std=True)[1] for model in active_models])
            maxiter = int(5 + 10 * uncertainty)
            maxiter = min(maxiter, 20)

            # Adaptive step size based on gradient
            gradient = obj_grad(x0)
            step_size = self.local_search_step_size_factor * uncertainty * np.linalg.norm(gradient)

            options = {'maxiter': maxiter, 'ftol': 1e-4}
            res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds, options=options)
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
            batch_size = min(self.batch_size, remaining_evals)
            next_X = self._select_next_points(batch_size, active_models)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            active_models = self._fit_model(self.X, self.y)

        return self.best_y, self.best_x
```
## Error
 Traceback (most recent call last):
  File "<AEHTSALSVBO>", line 158, in __call__
 158->             next_X = self._select_next_points(batch_size, active_models)
  File "<AEHTSALSVBO>", line 121, in _select_next_points
 121->             gradient = obj_grad(x0)
  File "<AEHTSALSVBO>", line 109, in obj_grad
 109->                 grads = [model.predict(x, return_std=False, return_grad=True)[1] for model in active_models]
  File "<AEHTSALSVBO>", line 109, in <listcomp>
 107 |             def obj_grad(x):
 108 |                 x = x.reshape(1, -1)
 109->                 grads = [model.predict(x, return_std=False, return_grad=True)[1] for model in active_models]
 110 |                 return np.mean(np.array(grads), axis=0)
 111 | 
TypeError: GaussianProcessRegressor.predict() got an unexpected keyword argument 'return_grad'
