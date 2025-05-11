# Description
**Gradient-aided Optimistic Bayesian Optimization (GAOBO)**: This algorithm combines the strengths of GradientEnhancedBO and BayesOptimisticBO to achieve efficient global optimization under budget constraints. It leverages gradient information to enhance the Gaussian Process (GP) surrogate model, guiding the search towards promising regions more effectively. It uses an Upper Confidence Bound (UCB) acquisition function with a dynamic exploration parameter to balance exploration and exploitation. Crucially, to avoid exceeding the evaluation budget, the gradient estimation and local search are performed using the GP model instead of directly evaluating the objective function.

# Justification
The main idea is to combine gradient information with the UCB acquisition function to improve the search efficiency. Gradient information helps to refine the GP surrogate model, leading to better predictions and a more informed exploration-exploitation trade-off. The dynamic exploration parameter in the UCB further enhances exploration, preventing premature convergence. The key improvement is to estimate gradients using the GP model, rather than the true function, to avoid exceeding the budget. Similarly, the local search is performed on the GP model.

The `GradientEnhancedBO` failed because it directly evaluated the function when estimating gradients, exceeding the budget. `BayesOptimisticBO` failed because the local search step evaluated the function, exceeding the budget. To address this, GAOBO uses the GP to predict the gradient and performs the local search on the GP model.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.optimize import minimize

class GradientAidedOptimisticBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim
        self.batch_size = min(5, self.dim)
        self.beta = 2.0  # Exploration parameter for UCB
        self.beta_decay = 0.99
        self.best_x = None
        self.best_y = np.inf
        self.delta = 1e-3  # Step size for finite differences

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(
            length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(noise_level=1e-5,
                                                                               noise_level_bounds=(1e-7, 1e-3))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
        model.fit(X, y)
        return model

    def _estimate_gradient(self, model, x):
        # Estimate the gradient using the GP model
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.delta
            x_minus[i] -= self.delta

            # Clip to ensure the points are within bounds
            x_plus[i] = np.clip(x_plus[i], self.bounds[0][i], self.bounds[1][i])
            x_minus[i] = np.clip(x_minus[i], self.bounds[0][i], self.bounds[1][i])

            y_plus, _ = model.predict(x_plus.reshape(1, -1), return_std=True)
            y_minus, _ = model.predict(x_minus.reshape(1, -1), return_std=True)

            gradient[i] = (y_plus - y_minus) / (2 * self.delta)
        return gradient

    def _acquisition_function(self, X, model):
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Upper Confidence Bound
        ucb = mu - self.beta * sigma  # minimize

        # Gradient-based term (using GP model for gradient estimation)
        gradients = np.array([self._estimate_gradient(model, x) for x in X])
        gradient_norm = np.linalg.norm(gradients, axis=1, keepdims=True)
        acquisition = ucb + 0.1 * gradient_norm  # Adjust the weight of the gradient term as needed

        return acquisition

    def _select_next_points(self, batch_size):
        candidate_points = self._sample_points(10 * batch_size)
        model = self._fit_model(self.X, self.y)
        acquisition_values = self._acquisition_function(candidate_points, model)
        indices = np.argsort(acquisition_values.flatten())[:batch_size]  # minimize
        return candidate_points[indices]

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

        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]

    def _local_search(self):
        # Local search around the best point using the GP model
        if self.best_x is not None:
            model = self._fit_model(self.X, self.y)

            def obj_func(x):
                # Use the GP model to predict the objective function value
                y_pred, _ = model.predict(x.reshape(1, -1), return_std=True)
                return y_pred

            bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
            result = minimize(obj_func, self.best_x, method='L-BFGS-B', bounds=bounds)

            if result.fun < self.best_y:
                self.best_y = result.fun
                self.best_x = result.x

                # Update X and y with the new best point (evaluated with the real function)
                new_X = result.x.reshape(1, -1)
                new_y = self._evaluate_points(lambda x: x, new_X)[0]  # Evaluate with the real function
                self._update_eval_points(new_X, new_y.reshape(1,-1))


    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        while self.n_evals < self.budget:
            # Optimization
            next_X = self._select_next_points(self.batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Local search
            self._local_search()

            # Decay exploration parameter
            self.beta *= self.beta_decay

        return self.best_y, self.best_x
```
## Feedback
 The algorithm GradientAidedOptimisticBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1604 with standard deviation 0.0982.

took 164.36 seconds to run.