You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- GradientEnhancedAdaptiveTrustRegionBO: 0.1977, 151.94 seconds, **GradientEnhancedAdaptiveTrustRegionBO (GEATRBO)**: This algorithm combines gradient estimation with an adaptive trust region strategy, enhanced by nearest neighbor-based lengthscale estimation for the GP kernel and dynamic adjustment of the UCB exploration parameter. It incorporates a momentum-based trust region adaptation for smoother transitions and a probability-based global search to escape local optima. The local search leverages both the GP model and gradient information, and the trust region size is adaptively adjusted based on the success of the local search.


- GradientEnhancedAdaptiveTrustRegionOptimisticHybridBO: 0.1975, 77.57 seconds, **GradientEnhancedAdaptiveTrustRegionOptimisticHybridBO (GEATROHBO)**: This algorithm fuses gradient-enhanced local search, adaptive trust region management, and a hybrid acquisition strategy, including both Expected Improvement (EI) and Upper Confidence Bound (UCB). It estimates gradients using finite differences to accelerate local search and employs an adaptive trust region to balance exploration and exploitation. The acquisition function combines EI for global exploration and UCB for local refinement within the trust region. The lengthscale is estimated using nearest neighbors. A probabilistic global search step is included to escape local optima. The trust region size and UCB's exploration parameter are adaptively adjusted based on the success of local search.


- AdaptiveTrustRegionMExBO: 0.1931, 191.64 seconds, **Adaptive Trust Region with Momentum and Exploitation-Exploration Balancing (ATRMExBO)**: This algorithm combines adaptive trust region management with momentum-based updates and dynamic adjustment of exploitation-exploration balance in local search. It leverages nearest neighbors for efficient lengthscale estimation in the Gaussian Process (GP) model. It uses Expected Improvement (EI) for global acquisition and a weighted combination of GP prediction and EI within the trust region for local search. The trust region size is adaptively adjusted based on the success of local search, incorporating momentum to smooth the adjustments. The exploitation-exploration balance in local search is dynamically adjusted based on the success rate of local searches. A global search step, optimized using L-BFGS-B, is probabilistically introduced to escape local optima.


- GradientEnhancedAdaptiveTrustRegionOptimisticHybridBO: 0.1919, 78.44 seconds, **GradientEnhancedAdaptiveTrustRegionOptimisticHybridBO (GEATR-OHBO)**: This algorithm combines the strengths of GradientEnhancedOptimisticTrustRegionBO (GEOTRBO) and AdaptiveTrustRegionOptimisticHybridBO. It incorporates gradient estimation for enhanced local search, adaptive trust region management, efficient lengthscale estimation using nearest neighbors, Expected Improvement (EI) for global acquisition, and Upper Confidence Bound (UCB) with a dynamic exploration parameter for enhanced global exploration. The local search within the trust region leverages the GP model and gradient information. The trust region size and UCB's exploration parameter are adaptively adjusted based on the success of local search, balancing exploration and exploitation. A global search step is probabilistically introduced to escape local optima. The gradient estimation is performed using a small number of evaluations and is incorporated into the local search process. The lengthscale is estimated using nearest neighbors for better GP model fitting.




The selected solution to update is:
**GradientEnhancedAdaptiveTrustRegionOptimisticHybridBO (GEATROHBO)**: This algorithm fuses gradient-enhanced local search, adaptive trust region management, and a hybrid acquisition strategy, including both Expected Improvement (EI) and Upper Confidence Bound (UCB). It estimates gradients using finite differences to accelerate local search and employs an adaptive trust region to balance exploration and exploitation. The acquisition function combines EI for global exploration and UCB for local refinement within the trust region. The lengthscale is estimated using nearest neighbors. A probabilistic global search step is included to escape local optima. The trust region size and UCB's exploration parameter are adaptively adjusted based on the success of local search.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors

class GradientEnhancedAdaptiveTrustRegionOptimisticHybridBO:
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
        self.success_threshold = 0.7
        self.beta = 2.0  # Exploration parameter for UCB
        self.beta_decay = 0.99 # Decay rate for beta
        self.global_search_prob = 0.05 # Probability of performing a global search step
        self.delta = 1e-3 # step size for finite differences in gradient estimation

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

        # Efficient lengthscale estimation using nearest neighbors
        nn = NearestNeighbors(n_neighbors=min(len(X), 10)).fit(X)
        distances, _ = nn.kneighbors(X)
        median_distance = np.median(distances[:, 1])  # Exclude the point itself

        # Define the kernel with the estimated lengthscale
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=median_distance, length_scale_bounds=(1e-3, 1e3))

        # Gaussian Process Regressor
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))  # Return zeros if the model hasn't been fit yet

        model = self._fit_model(self.X, self.y)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement
        y_best = np.min(self.y)
        imp = y_best - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # avoid division by zero

        # Upper Confidence Bound
        ucb = mu - self.beta * sigma # minimize

        # Combine EI and UCB (weighted average)
        acquisition = 0.5 * ei + 0.5 * ucb
        return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        candidate_points = self._sample_points(10 * batch_size)  # Generate more candidates
        acquisition_values = self._acquisition_function(candidate_points)

        # Sort by acquisition function value and select top batch_size points
        indices = np.argsort(acquisition_values.flatten())[::-1][:batch_size]
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
            if np.random.rand() < self.global_search_prob:
                # Global search step
                next_x = self._select_next_points(1)[0]
            else:
                next_x = self._local_search(model, best_x.copy(), gradient)

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
The algorithm GradientEnhancedAdaptiveTrustRegionOptimisticHybridBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1975 with standard deviation 0.1076.

took 77.57 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

