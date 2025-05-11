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
**Adaptive Trust Region with Momentum and Exploitation-Exploration Balancing (ATRMExBO)**: This algorithm combines adaptive trust region management with momentum-based updates and dynamic adjustment of exploitation-exploration balance in local search. It leverages nearest neighbors for efficient lengthscale estimation in the Gaussian Process (GP) model. It uses Expected Improvement (EI) for global acquisition and a weighted combination of GP prediction and EI within the trust region for local search. The trust region size is adaptively adjusted based on the success of local search, incorporating momentum to smooth the adjustments. The exploitation-exploration balance in local search is dynamically adjusted based on the success rate of local searches. A global search step, optimized using L-BFGS-B, is probabilistically introduced to escape local optima.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize

class AdaptiveTrustRegionMExBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim
        self.batch_size = min(5, self.dim)
        self.trust_region_size = 2.0
        self.trust_region_shrink = 0.5
        self.trust_region_expand = 2.0
        self.local_search_exploitation = 0.8  # Initial weight for exploitation in local search
        self.local_search_exploitation_adjust = 0.05
        self.global_search_prob = 0.05
        self.success_threshold = 0.7
        self.trust_region_momentum = 0.5 # Momentum for trust region size update
        self.prev_trust_region_change = 0.0
        self.successful_local_searches = 0
        self.success_window = 5 # Window size to track successful local searches


    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        nn = NearestNeighbors(n_neighbors=min(len(X), 10)).fit(X)
        distances, _ = nn.kneighbors(X)
        median_distance = np.median(distances[:, 1])

        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(
            length_scale=median_distance, length_scale_bounds=(1e-3, 1e3)
        )

        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        model = self._fit_model(self.X, self.y)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        y_best = np.min(self.y)
        imp = y_best - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0
        return ei

    def _select_next_points(self, batch_size):
        candidate_points = self._sample_points(10 * batch_size)
        acquisition_values = self._acquisition_function(candidate_points)
        indices = np.argsort(acquisition_values.flatten())[::-1][:batch_size]
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

    def _local_search(self, model, center, n_points=50):
        candidate_points = center + self.trust_region_size * np.random.uniform(-1, 1, size=(n_points, self.dim))
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])

        mu, _ = model.predict(candidate_points, return_std=True)
        mu = mu.reshape(-1, 1)
        ei = self._acquisition_function(candidate_points)

        weighted_values = self.local_search_exploitation * mu + (1 - self.local_search_exploitation) * (-ei)
        best_index = np.argmin(weighted_values)
        best_point = candidate_points[best_index]

        return best_point

    def _global_search(self):
        # Refined global search using optimization of the acquisition function
        def objective(x):
            return -self._acquisition_function(x.reshape(1, -1))[0, 0]

        x0 = self._sample_points(1)[0]  # Start from a random point
        bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5})  # Limited iterations
        return result.x

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_index = np.argmin(self.y)
        best_x = self.X[best_index]
        best_y = self.y[best_index][0]

        success_history = []

        while self.n_evals < self.budget:
            model = self._fit_model(self.X, self.y)

            if np.random.rand() < self.global_search_prob:
                next_x = self._global_search()  # Use refined global search
            else:
                next_x = self._local_search(model, best_x.copy())
            next_x = np.clip(next_x, self.bounds[0], self.bounds[1])

            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0]
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            if next_y < best_y:
                best_x = next_x
                best_y = next_y
                trust_region_change = self.trust_region_expand - 1 # Change relative to current size
                success_history.append(1)
            else:
                trust_region_change = self.trust_region_shrink - 1 # Change relative to current size
                success_history.append(0)

            # Apply momentum to trust region update
            self.prev_trust_region_change = self.trust_region_momentum * self.prev_trust_region_change + (1 - self.trust_region_momentum) * trust_region_change
            self.trust_region_size *= (1 + self.prev_trust_region_change)

            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0)

            # Adjust exploitation-exploration balance based on success rate
            if len(success_history) > self.success_window:
                success_rate = np.mean(success_history[-self.success_window:])
                if success_rate > self.success_threshold:
                    self.local_search_exploitation = min(1.0, self.local_search_exploitation + self.local_search_exploitation_adjust)
                else:
                    self.local_search_exploitation = max(0.0, self.local_search_exploitation - self.local_search_exploitation_adjust)

        return best_y, best_x

```
The algorithm AdaptiveTrustRegionMExBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1931 with standard deviation 0.1095.

took 191.64 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

