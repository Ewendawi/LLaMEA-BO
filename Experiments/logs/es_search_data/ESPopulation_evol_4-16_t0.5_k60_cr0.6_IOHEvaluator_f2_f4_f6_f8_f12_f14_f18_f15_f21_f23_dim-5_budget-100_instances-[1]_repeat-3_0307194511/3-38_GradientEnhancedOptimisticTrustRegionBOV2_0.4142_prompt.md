You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- AdaptiveTrustRegionOptimisticHybridBO: 0.2043, 69.81 seconds, **AdaptiveTrustRegionOptimisticHybridBO**: This algorithm merges the strengths of AdaptiveTrustRegionHybridBO and TrustRegionOptimisticBO, incorporating adaptive trust region management, efficient lengthscale estimation using nearest neighbors, Expected Improvement (EI) for acquisition, and Upper Confidence Bound (UCB) with a dynamic exploration parameter for enhanced global exploration. The local search within the trust region leverages the GP model. The trust region size and UCB's exploration parameter are adaptively adjusted based on the success of local search, balancing exploration and exploitation. A global search step is probabilistically introduced to escape local optima.


- GradientEnhancedOptimisticTrustRegionBO: 0.1971, 119.11 seconds, **GradientEnhancedOptimisticTrustRegionBO (GEOTRBO)**: This algorithm synergistically integrates gradient estimation, optimistic exploration, and a trust region framework. It enhances local search by incorporating gradient information, promotes global exploration using an Upper Confidence Bound (UCB) acquisition function with dynamic exploration parameter, and manages the exploration-exploitation trade-off with an adaptive trust region. The gradient is estimated efficiently using a small number of evaluations, and the UCB acquisition function balances exploration and exploitation. An adaptive trust region ensures focused local search while allowing for occasional global jumps. To further improve performance, a momentum-based trust region adaptation is introduced.


- AdaptiveTrustRegionHybridV2BO: 0.1883, 158.17 seconds, **AdaptiveTrustRegionHybridV2BO**: This algorithm enhances the `AdaptiveTrustRegionHybridBO` by incorporating an adaptive mechanism for the local search's exploration-exploitation balance, drawing inspiration from `AdaptiveTrustRegionBO`. It dynamically adjusts the weight given to the GP model's predicted mean versus the acquisition function during local search. Additionally, it introduces a probability-based global search step to escape local optima, as seen in `AdaptiveTrustRegionBO`, but uses a more refined global search strategy. The lengthscale of the GP kernel is efficiently estimated using nearest neighbors. The trust region size is adaptively adjusted based on the success of local search.


- AdaptiveTrustRegionOptimisticHybridBO: 0.1876, 65.97 seconds, **AdaptiveTrustRegionOptimisticHybridBO**: This algorithm synergizes the strengths of AdaptiveTrustRegionHybridBO and TrustRegionOptimisticBO. It employs an initial Latin Hypercube sampling for exploration, followed by a Gaussian Process (GP) surrogate model. It uses Expected Improvement (EI) for global acquisition and Upper Confidence Bound (UCB) for local search within the trust region. The lengthscale of the GP kernel is efficiently estimated using nearest neighbors. A trust region approach is incorporated to balance exploration and exploitation, where the trust region size is adaptively adjusted based on the success of local search. The local search is performed using a combination of the GP model and the UCB criterion within the trust region, and the best point is evaluated using the actual function. A dynamic exploration parameter for UCB is introduced. This hybrid approach aims to improve both the efficiency and effectiveness of Bayesian Optimization, mitigating premature convergence and enhancing global exploration.




The selected solution to update is:
**GradientEnhancedOptimisticTrustRegionBO (GEOTRBO)**: This algorithm synergistically integrates gradient estimation, optimistic exploration, and a trust region framework. It enhances local search by incorporating gradient information, promotes global exploration using an Upper Confidence Bound (UCB) acquisition function with dynamic exploration parameter, and manages the exploration-exploitation trade-off with an adaptive trust region. The gradient is estimated efficiently using a small number of evaluations, and the UCB acquisition function balances exploration and exploitation. An adaptive trust region ensures focused local search while allowing for occasional global jumps. To further improve performance, a momentum-based trust region adaptation is introduced.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

class GradientEnhancedOptimisticTrustRegionBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * self.dim # initial number of samples
        self.batch_size = min(5, self.dim)
        self.trust_region_size = 2.0
        self.trust_region_shrink = 0.5
        self.trust_region_expand = 2.0
        self.beta = 2.0  # Exploration parameter for UCB
        self.beta_decay = 0.99 # Decay rate for beta
        self.best_x = None
        self.best_y = np.inf
        self.delta = 1e-3 # step size for finite differences in gradient estimation
        self.trust_region_momentum = 0.5 # Momentum for trust region size update
        self.prev_trust_region_change = 0.0

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
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        model = self._fit_model(self.X, self.y)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Upper Confidence Bound
        ucb = mu - self.beta * sigma # minimize
        return ucb

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        candidate_points = self._sample_points(10 * batch_size)
        acquisition_values = self._acquisition_function(candidate_points)
        indices = np.argsort(acquisition_values.flatten())[:batch_size] # minimize
        return candidate_points[indices]

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

        # Update best seen point
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]

    def _estimate_gradient(self, model, x):
        # Estimate the gradient of the function at point x using finite differences
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.delta
            x_minus[i] -= self.delta
            
            # Clip to ensure the points are within bounds
            x_plus[i] = np.clip(x_plus[i], self.bounds[0][i], self.bounds[1][i])
            x_minus[i] = np.clip(x_minus[i], self.bounds[0][i], self.bounds[1][i])

            # Use the GP model to predict function values
            y_plus, _ = model.predict(x_plus.reshape(1, -1), return_std=True)
            y_minus, _ = model.predict(x_minus.reshape(1, -1), return_std=True)

            gradient[i] = (y_plus - y_minus) / (2 * self.delta)
        return gradient

    def _local_search(self, model, center, gradient, n_points=50):
        # Perform local search within the trust region using the GP model and gradient information
        # Generate candidate points within the trust region
        candidate_points = center + self.trust_region_size * np.random.uniform(-1, 1, size=(n_points, self.dim))

        # Clip the candidate points to stay within the bounds
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])

        # Predict the mean values using the GP model
        mu, _ = model.predict(candidate_points, return_std=True)

        # Incorporate gradient information into the prediction
        mu = mu.reshape(-1) - 0.1 * np.sum(gradient * (candidate_points - center), axis=1)

        # Select the point with the minimum predicted mean value
        best_index = np.argmin(mu)
        best_point = candidate_points[best_index]

        return best_point

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_index = np.argmin(self.y)
        best_x = self.X[best_index]
        best_y = self.y[best_index][0]

        while self.n_evals < self.budget:
            # Fit the GP model
            model = self._fit_model(self.X, self.y)

            # Estimate gradient at the best point
            gradient = self._estimate_gradient(model, best_x)

            # Perform local search within the trust region
            next_x = self._local_search(model, best_x.copy(), gradient)
            next_x = np.clip(next_x, self.bounds[0], self.bounds[1]) # Ensure it's within bounds

            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0] # Evaluate the actual function
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            # Check if the new point is better than the current best
            if next_y < best_y:
                best_x = next_x
                best_y = next_y
                # Adjust trust region size
                trust_region_change = self.trust_region_expand - 1.0
                self.trust_region_size *= self.trust_region_expand
            else:
                # Shrink trust region if no improvement
                trust_region_change = 1.0 - self.trust_region_shrink
                self.trust_region_size *= self.trust_region_shrink

            # Apply momentum to trust region size update
            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0) # Keep trust region within reasonable bounds
            
            # Decay exploration parameter
            self.beta *= self.beta_decay

        return best_y, best_x

```
The algorithm GradientEnhancedOptimisticTrustRegionBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1971 with standard deviation 0.1125.

took 119.11 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

