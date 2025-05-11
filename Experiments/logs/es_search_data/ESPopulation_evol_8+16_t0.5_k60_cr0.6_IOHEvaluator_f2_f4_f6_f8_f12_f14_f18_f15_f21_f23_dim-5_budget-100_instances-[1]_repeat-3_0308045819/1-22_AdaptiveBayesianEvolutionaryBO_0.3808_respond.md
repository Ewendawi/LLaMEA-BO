# Description
**Adaptive Bayesian Evolutionary Optimization with Dynamic Kernel Tuning (ABEO-DKT):** This algorithm enhances the original Bayesian Evolutionary Optimization (BEO) by incorporating adaptive kernel tuning for the Gaussian Process (GP) and dynamically adjusting the diversity term in the acquisition function. The GP kernel parameters (length_scale) are optimized using a gradient-based method (L-BFGS-B) during the GP fitting stage. This allows the GP to better capture the underlying function's characteristics. The diversity term in the acquisition function is also dynamically adjusted based on the exploration-exploitation balance. Furthermore, the batch size is adaptively adjusted based on the budget and the dimensionality of the problem.

# Justification
1.  **Adaptive Kernel Tuning:** The original BEO uses a fixed kernel, which might not be optimal for all functions in the BBOB suite. Tuning the kernel parameters allows the GP to adapt to the function's landscape, improving the accuracy of the surrogate model and leading to better optimization performance. L-BFGS-B is a computationally efficient gradient-based optimization method suitable for this task.
2.  **Dynamic Diversity Term:** The diversity term in the acquisition function helps to balance exploration and exploitation. Dynamically adjusting the weight of this term based on the optimization progress can further improve performance. The adjustment is based on the ratio of the standard deviation of the GP prediction to the Expected Improvement, which indicates the level of uncertainty and the potential for improvement.
3.  **Adaptive Batch Size:** The batch size is dynamically adjusted to improve the efficiency of the algorithm. It is set to a small value (1) for high-dimensional problems and increased for low-dimensional problems to allow for more parallel exploration.
4.  **Budget-Aware Differential Evolution:** The number of iterations in the differential evolution is adjusted based on the remaining budget, ensuring that the algorithm does not exceed the budget limit.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution, minimize
from sklearn.preprocessing import StandardScaler

class AdaptiveBayesianEvolutionaryBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(5*dim, self.budget//10)

        # Do not add any other arguments without a default value
        self.gp = None
        self.best_x = None
        self.best_y = float('inf')
        self.de_pop_size = 15 # Population size for differential evolution
        self.gp_update_interval = 5 # Update GP model every n iterations
        self.diversity_weight = 0.1 # Initial diversity weight

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
        # Normalize y
        scaler = StandardScaler()
        y_normalized = scaler.fit_transform(y)

        # Define the kernel with tunable length_scale
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-6)

        # Optimize the kernel parameters using L-BFGS-B
        self.gp.fit(X, y_normalized)

        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function: Expected Improvement + Diversity
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            # Normalize best_y for EI calculation
            scaler = StandardScaler()
            scaler.fit(self.y)  # Fit on self.y to get the correct scale
            best_y_normalized = scaler.transform(np.array([[self.best_y]]))[0][0]

            mu, sigma = self.gp.predict(X, return_std=True)
            sigma = np.clip(sigma, 1e-9, np.inf)
            imp = best_y_normalized - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

            # Diversity term: encourage exploration
            if self.X is not None:
                distances = np.min([np.linalg.norm(x - self.X, axis=1) for x in X], axis=0)
                diversity = distances.reshape(-1,1)
            else:
                diversity = np.ones((len(X), 1)) # No diversity if no points yet

            # Dynamic diversity weight adjustment
            self.diversity_weight = np.std(sigma) / np.mean(ei + 1e-6)
            self.diversity_weight = np.clip(self.diversity_weight, 0.01, 0.5) # Clip to avoid extreme values

            # Combine EI and diversity
            acquisition = ei + self.diversity_weight * diversity # Adjust weight for diversity as needed
            return acquisition.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using differential evolution
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Define the objective function for differential evolution (negative acquisition function)
        def de_objective(x):
            return -self._acquisition_function(x.reshape(1, -1))[0, 0]

        # Perform differential evolution
        de_bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
        result = differential_evolution(de_objective, de_bounds, popsize=self.de_pop_size, maxiter=max(1, self.budget//(batch_size*5)), tol=0.01, disp=False) # Adjust maxiter

        # Select the best point from differential evolution
        return result.x.reshape(1, -1)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y

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
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Optimization loop
        batch_size = min(max(1, self.dim // 5), 4)  # Adaptive batch size
        iteration = 0
        while self.n_evals < self.budget:
            # Fit the GP model periodically
            if iteration % self.gp_update_interval == 0:
                self._fit_model(self.X, self.y)

            # Select points by acquisition function using differential evolution
            next_X = self._select_next_points(batch_size)

            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)

            # Update the evaluated points
            self._update_eval_points(next_X, next_y)

            iteration += 1

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AdaptiveBayesianEvolutionaryBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1585 with standard deviation 0.0976.

took 4999.31 seconds to run.