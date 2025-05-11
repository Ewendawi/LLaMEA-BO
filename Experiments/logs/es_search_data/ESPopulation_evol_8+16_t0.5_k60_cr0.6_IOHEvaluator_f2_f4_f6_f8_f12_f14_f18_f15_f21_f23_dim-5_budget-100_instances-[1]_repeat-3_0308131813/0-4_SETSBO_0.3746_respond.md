# Description
**Surrogate Ensemble with Thompson Sampling and Local Search (SETSBO):** This algorithm leverages an ensemble of surrogate models (Gaussian Process Regression with different kernels) to improve the robustness and accuracy of the Bayesian optimization process. It uses Thompson Sampling for acquisition, which naturally balances exploration and exploitation. Furthermore, it integrates a local search strategy (L-BFGS-B) to refine the search around promising regions identified by Thompson Sampling. The ensemble of surrogates provides a more reliable estimate of the function landscape, while Thompson Sampling offers a computationally efficient way to select the next points. Local search further enhances the exploitation of promising regions.

# Justification
The key components of SETSBO are justified as follows:

*   **Ensemble of Surrogates:** Using an ensemble of GPR models with different kernels (RBF with varying length scales) helps to capture different characteristics of the objective function. This approach is more robust than relying on a single surrogate model, especially when the function landscape is complex or unknown.

*   **Thompson Sampling:** Thompson Sampling is a computationally efficient acquisition strategy that naturally balances exploration and exploitation. It samples from the posterior distribution of each surrogate model and selects the point with the highest sampled value. This approach avoids the need to explicitly define exploration-exploitation parameters, as in the case of EI or UCB.

*   **Local Search:** Integrating a local search strategy (L-BFGS-B) allows for the refinement of the search around promising regions identified by Thompson Sampling. This is particularly useful when the function landscape has local optima or is noisy. By performing a limited number of local search iterations, the algorithm can quickly converge to a local optimum within a promising region.

*   **Diversity from Previous Algorithms:** This algorithm differs from the previous ones by using an ensemble of surrogate models, Thompson Sampling for acquisition, and a local search strategy. EHBBO uses a hybrid acquisition function, ATRBO uses a trust region, BONGIBO focuses on noise handling and gradient information, and DensiTreeBO uses density-based clustering. SETSBO combines an ensemble approach, Thompson Sampling, and local search to achieve a different balance between exploration and exploitation.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class SETSBO:
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
        for i, model in enumerate(self.models):
            mu, sigma = model.predict(X, return_std=True)
            mu = mu.reshape(-1, 1)
            sigma = sigma.reshape(-1, 1)
            sampled_values[:, i] = np.random.normal(mu.flatten(), sigma.flatten())

        # Average the sampled values across all models
        acquisition = np.mean(sampled_values, axis=1, keepdims=True)
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

        # Local search to improve the selected points
        for i in range(batch_size):
            def obj_func(x):
                x = x.reshape(1, -1)
                return np.mean([model.predict(x)[0] for model in self.models])  # Minimize the average predicted value

            x0 = next_points[i]
            bounds = [(self.bounds[0][j], self.bounds[1][j]) for j in range(self.dim)]
            res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5})  # Limited iterations
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
            batch_size = min(self.n_init, remaining_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self._fit_model(self.X, self.y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm SETSBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1544 with standard deviation 0.0986.

took 165.78 seconds to run.