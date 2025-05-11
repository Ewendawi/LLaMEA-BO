# Description
**Adaptive Ensemble with Thompson Sampling and Dynamic Local Search Bayesian Optimization (AETSALSBO):** This algorithm combines the strengths of AHBBO and EHTSALSBO. It uses an ensemble of Gaussian Process Regression (GPR) models with different kernels, Thompson Sampling for efficient acquisition, and a hybrid acquisition function that incorporates both Expected Improvement (EI) and a distance-based exploration term. It also features an adaptive local search strategy, similar to EHTSALSBO, but with dynamically adjusted intensity based on both the uncertainty estimates from the GPR models and the optimization progress. Furthermore, the exploration weight in the acquisition function is dynamically adjusted based on the optimization progress, similar to AHBBO.

# Justification
The AETSALSBO algorithm aims to improve upon existing Bayesian optimization methods by integrating several key features:

1.  **Ensemble of Surrogate Models:** Using an ensemble of GPR models with different kernels, as in EHTSALSBO, allows for a more robust and accurate representation of the objective function landscape. This helps to mitigate the risk of relying on a single model that may be biased or inaccurate.

2.  **Thompson Sampling:** Thompson Sampling provides an efficient way to balance exploration and exploitation. By sampling from the posterior distribution of each model in the ensemble, the algorithm can effectively explore the search space while also focusing on promising regions.

3.  **Hybrid Acquisition Function:** Combining Expected Improvement (EI) with a distance-based exploration term helps to further balance exploration and exploitation. EI encourages the algorithm to focus on regions where the expected improvement is high, while the distance-based exploration term encourages the algorithm to explore regions that are far away from previously evaluated points.

4.  **Adaptive Local Search:** The adaptive local search strategy refines the search around promising regions identified by Thompson Sampling. By dynamically adjusting the intensity of the local search based on the uncertainty estimates from the GPR models and the optimization progress, the algorithm can efficiently exploit promising regions while avoiding premature convergence.

5. **Adaptive Exploration Weight:** Dynamically adjusting the exploration weight, as in AHBBO, allows the algorithm to shift its focus from exploration to exploitation as the optimization progresses. This helps to improve the overall efficiency of the optimization process.

By combining these features, AETSALSBO aims to provide a robust and efficient Bayesian optimization algorithm that can effectively handle a wide range of black-box optimization problems.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class AETSALSBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim

        self.best_y = np.inf
        self.best_x = None

        self.n_models = 3  # Number of surrogate models in the ensemble
        self.models = []
        for i in range(self.n_models):
            length_scale = 1.0 * (i + 1) / self.n_models  # Varying length scales
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
            self.models.append(model)

        self.batch_size = min(10, dim)  # Batch size for selecting points
        self.exploration_weight = 0.1  # Initial exploration weight
        self.exploration_decay = 0.995  # Decay factor for exploration weight
        self.min_exploration = 0.01  # Minimum exploration weight

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        for model in self.models:
            model.fit(X, y)

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)

        # Thompson Sampling: Sample from the posterior distribution of each model
        sampled_values = np.zeros((X.shape[0], self.n_models))
        sigmas = np.zeros((X.shape[0], self.n_models))
        for i, model in enumerate(self.models):
            mu, sigma = model.predict(X, return_std=True)
            mu = mu.reshape(-1, 1)
            sigma = sigma.reshape(-1, 1)
            sampled_values[:, i] = np.random.normal(mu.flatten(), sigma.flatten())
            sigmas[:, i] = sigma.flatten()

        # Average the sampled values across all models
        acquisition = np.mean(sampled_values, axis=1, keepdims=True)
        acquisition = acquisition.reshape(-1, 1)

        # Hybrid acquisition function (EI + exploration)
        mu = np.mean([model.predict(X) for model in self.models], axis=0).reshape(-1, 1)
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

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Generate candidate points
        candidate_points = self._sample_points(100 * batch_size)

        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_points)

        # Select the top batch_size points with the highest acquisition values
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]

        # Adaptive Local search to improve the selected points
        for i in range(batch_size):
            def obj_func(x):
                x = x.reshape(1, -1)
                return np.mean([model.predict(x)[0] for model in self.models])  # Minimize the average predicted value

            x0 = next_points[i]
            bounds = [(self.bounds[0][j], self.bounds[1][j]) for j in range(self.dim)]

            # Adaptive local search iterations based on uncertainty and optimization progress
            uncertainty = np.mean([model.predict(x0.reshape(1, -1), return_std=True)[1] for model in self.models])
            progress = self.n_evals / self.budget  # Optimization progress (0 to 1)
            maxiter = int(5 + 10 * uncertainty * (1 - progress))  # Fewer iterations as optimization progresses
            maxiter = min(maxiter, 20) # Cap the iterations

            res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': maxiter})  # Limited iterations
            next_points[i] = res.x

        return next_points

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        # Do not change the function signature
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

        # Update best seen value
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Optimization
            # select points by acquisition function
            remaining_evals = self.budget - self.n_evals
            batch_size = min(self.batch_size, remaining_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self._fit_model(self.X, self.y)

            # Update exploration weight
            self.exploration_weight = max(self.exploration_weight * self.exploration_decay, self.min_exploration)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AETSALSBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1570 with standard deviation 0.1026.

took 376.35 seconds to run.