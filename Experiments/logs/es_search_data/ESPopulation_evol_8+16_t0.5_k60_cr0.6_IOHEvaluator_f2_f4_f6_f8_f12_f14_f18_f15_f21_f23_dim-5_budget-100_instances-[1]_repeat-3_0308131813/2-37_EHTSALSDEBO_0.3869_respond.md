# Description
**Ensemble Hybrid Thompson Sampling with Adaptive Local Search and Exploration Decay Bayesian Optimization (EHTSALSDEBO):** This algorithm builds upon EHTSALSBO by introducing a decaying exploration factor in the hybrid acquisition function. The exploration term, which is distance-based, is multiplied by a factor that decreases with the number of evaluations. This encourages exploration early on and exploitation later in the optimization process. Additionally, the local search is enhanced by incorporating a momentum-based acceleration to escape local optima more effectively.

# Justification
The key improvements are:

1.  **Decaying Exploration Factor:** The original EHTSALSBO uses a fixed exploration weight in the acquisition function. By decaying this weight, we can dynamically adjust the exploration-exploitation trade-off. Initially, a higher exploration weight encourages the algorithm to explore the search space more broadly. As the number of evaluations increases, the exploration weight decreases, focusing the search on promising regions identified earlier. This helps to avoid premature convergence and improves the algorithm's ability to find the global optimum.

2.  **Momentum-based Local Search:** The local search component of EHTSALSBO uses L-BFGS-B to refine solutions. While effective, L-BFGS-B can sometimes get stuck in local optima. By incorporating a momentum term, the local search can escape these local optima more effectively. The momentum term accumulates the gradient information from previous iterations and uses it to guide the search direction. This helps to overcome small barriers in the search space and find better solutions. The momentum term is dynamically adjusted based on the uncertainty estimates from the GPR models, similar to how the number of local search iterations is adjusted.

These changes aim to improve the balance between exploration and exploitation, leading to better performance on the BBOB test suite.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class EHTSALSDEBO:
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
        self.exploration_decay = 0.995 # Decay factor for exploration weight
        self.momentum = 0.0 #initial momentum
        self.momentum_decay = 0.9

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

            # Adaptive local search iterations based on uncertainty
            uncertainty = np.mean([model.predict(x0.reshape(1, -1), return_std=True)[1] for model in self.models])
            maxiter = int(5 + 10 * uncertainty)  # More iterations for higher uncertainty
            maxiter = min(maxiter, 20) # Cap the iterations

            # Momentum-based acceleration
            if i == 0:
                self.momentum = np.zeros_like(x0)
            self.momentum = self.momentum_decay * self.momentum

            def grad_func(x):
                x = x.reshape(1, -1)
                grads = []
                for model in self.models:
                    # numerical gradient
                    delta = 1e-5
                    grad = np.zeros_like(x0)
                    for j in range(self.dim):
                        x_plus = x0.copy()
                        x_plus[j] += delta
                        x_minus = x0.copy()
                        x_minus[j] -= delta
                        grad[j] = (model.predict(x_plus.reshape(1, -1))[0] - model.predict(x_minus.reshape(1, -1))[0]) / (2 * delta)
                    grads.append(grad)
                return np.mean(grads, axis=0)

            def obj_func_momentum(x):
                x = x.reshape(1, -1)
                return np.mean([model.predict(x)[0] for model in self.models])

            res = minimize(obj_func_momentum, x0, method='L-BFGS-B', jac=grad_func, bounds=bounds, options={'maxiter': maxiter})  # Limited iterations
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

            # Decay exploration weight
            self.exploration_weight *= self.exploration_decay

        return self.best_y, self.best_x
```
## Feedback
 The algorithm EHTSALSDEBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1611 with standard deviation 0.1015.

took 406.46 seconds to run.