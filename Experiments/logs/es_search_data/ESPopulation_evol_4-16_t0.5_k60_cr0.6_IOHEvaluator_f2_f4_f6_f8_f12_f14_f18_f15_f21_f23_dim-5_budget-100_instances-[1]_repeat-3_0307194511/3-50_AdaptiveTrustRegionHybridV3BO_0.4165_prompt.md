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
**AdaptiveTrustRegionHybridV2BO**: This algorithm enhances the `AdaptiveTrustRegionHybridBO` by incorporating an adaptive mechanism for the local search's exploration-exploitation balance, drawing inspiration from `AdaptiveTrustRegionBO`. It dynamically adjusts the weight given to the GP model's predicted mean versus the acquisition function during local search. Additionally, it introduces a probability-based global search step to escape local optima, as seen in `AdaptiveTrustRegionBO`, but uses a more refined global search strategy. The lengthscale of the GP kernel is efficiently estimated using nearest neighbors. The trust region size is adaptively adjusted based on the success of local search.


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

class AdaptiveTrustRegionHybridV2BO:
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

        successful_local_searches = 0
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
                self.trust_region_size *= self.trust_region_expand
                successful_local_searches += 1
                # Increase exploitation
                self.local_search_exploitation = min(1.0, self.local_search_exploitation + self.local_search_exploitation_adjust)
            else:
                self.trust_region_size *= self.trust_region_shrink
                successful_local_searches = 0
                # Increase exploration
                self.local_search_exploitation = max(0.0, self.local_search_exploitation - self.local_search_exploitation_adjust)

            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0)

        return best_y, best_x

```
The algorithm AdaptiveTrustRegionHybridV2BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1883 with standard deviation 0.1035.

took 158.17 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

