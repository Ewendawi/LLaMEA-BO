You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- AdaptiveTrustRegionHybridBO: 0.1919, 67.74 seconds, **AdaptiveTrustRegionHybridBO**: This algorithm combines the strengths of TrustRegionBO and EfficientHybridBO. It uses an initial Latin Hypercube sampling for exploration, followed by a Gaussian Process (GP) surrogate model with Expected Improvement (EI) acquisition. The lengthscale of the GP kernel is efficiently estimated using nearest neighbors. A trust region approach is incorporated to balance exploration and exploitation, where the trust region size is adaptively adjusted based on the success of local search. The local search is performed using the GP model within the trust region, and the best point is evaluated using the actual function. This hybrid approach aims to improve both the efficiency and effectiveness of Bayesian Optimization.


- TREGEBO: 0.1861, 118.77 seconds, **Trust Region with Efficient Gradient Estimation (TREGEBO)**: This algorithm combines the Trust Region approach from `TrustRegionBO` with an efficient gradient estimation strategy to enhance local search and improve the Gaussian Process (GP) surrogate model. Instead of directly evaluating the function to estimate the gradient for multiple candidate points like in `GradientEnhancedBO`, it estimates the gradient only at the current best point using a reduced number of function evaluations. The gradient information is then used to refine the local search within the trust region. To avoid exceeding the budget during gradient estimation, a surrogate model is used to approximate function values for points used in finite difference calculations.


- AdaptiveTrustRegionBO: 0.1842, 206.63 seconds, **AdaptiveTrustRegionBO**: This algorithm refines the TrustRegionBO by introducing adaptive mechanisms for trust region size adjustment and local search point selection. It uses a more informed approach to update the trust region size based on the actual improvement observed, rather than a fixed expansion/shrinkage factor. It also incorporates the acquisition function into the local search, balancing exploration and exploitation within the trust region. Furthermore, to avoid getting stuck in local optima, it introduces a probability of performing a global search step.


- TrustRegionOptimisticBO: 0.1823, 112.70 seconds, **TrustRegionOptimisticBO**: This algorithm combines the strengths of TrustRegionBO and BayesOptimisticBO. It uses a Gaussian Process (GP) surrogate model with a Trust Region approach for local exploitation and an Upper Confidence Bound (UCB) acquisition function with dynamic exploration parameter for global exploration. The local search within the trust region is performed using the GP model to avoid exceeding the budget. The UCB acquisition function is used to select the next points to evaluate, balancing exploration and exploitation. The trust region size is adapted based on the success of the local search. The algorithm also monitors the number of function evaluations and stops when the budget is exhausted.




The selected solution to update is:
**TrustRegionOptimisticBO**: This algorithm combines the strengths of TrustRegionBO and BayesOptimisticBO. It uses a Gaussian Process (GP) surrogate model with a Trust Region approach for local exploitation and an Upper Confidence Bound (UCB) acquisition function with dynamic exploration parameter for global exploration. The local search within the trust region is performed using the GP model to avoid exceeding the budget. The UCB acquisition function is used to select the next points to evaluate, balancing exploration and exploitation. The trust region size is adapted based on the success of the local search. The algorithm also monitors the number of function evaluations and stops when the budget is exhausted.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

class TrustRegionOptimisticBO:
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

    def _local_search(self, model, center, n_points=50):
        # Perform local search within the trust region using the GP model
        # Generate candidate points within the trust region
        candidate_points = center + self.trust_region_size * np.random.uniform(-1, 1, size=(n_points, self.dim))

        # Clip the candidate points to stay within the bounds
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])

        # Predict the mean values using the GP model
        mu, _ = model.predict(candidate_points, return_std=True)

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

            # Perform local search within the trust region
            next_x = self._local_search(model, best_x.copy())
            next_x = np.clip(next_x, self.bounds[0], self.bounds[1]) # Ensure it's within bounds

            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0] # Evaluate the actual function
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            # Check if the new point is better than the current best
            if next_y < best_y:
                best_x = next_x
                best_y = next_y
                # Adjust trust region size
                self.trust_region_size *= self.trust_region_expand
            else:
                # Shrink trust region if no improvement
                self.trust_region_size *= self.trust_region_shrink

            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0) # Keep trust region within reasonable bounds

            # Decay exploration parameter
            self.beta *= self.beta_decay

        return best_y, best_x

```
The algorithm TrustRegionOptimisticBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1823 with standard deviation 0.1072.

took 112.70 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

