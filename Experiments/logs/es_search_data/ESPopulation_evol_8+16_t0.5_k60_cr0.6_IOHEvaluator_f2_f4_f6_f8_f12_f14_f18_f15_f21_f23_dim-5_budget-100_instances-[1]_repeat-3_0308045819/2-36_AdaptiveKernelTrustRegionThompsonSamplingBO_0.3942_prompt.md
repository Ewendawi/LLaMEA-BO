You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- AdaptiveParetoTrustRegionBO: 0.1768, 12.67 seconds, **Adaptive Pareto Trust Region Bayesian Optimization (APTRBO):** This algorithm combines the strengths of Adaptive Trust Region Bayesian Optimization (ATBO) and Pareto Active Bayesian Optimization (PABO). It uses a trust region approach to focus the search around promising regions while employing a Pareto-based multi-objective approach to balance exploration (diversity) and exploitation (expected improvement) within the trust region. The trust region radius is adapted based on the success of finding better solutions. Active learning, by querying points with high GP variance, is used to reduce uncertainty, particularly on the Pareto front.


- AdaptiveEvolutionaryTrustRegionBO: 0.1747, 160.25 seconds, **Adaptive Evolutionary Trust Region Bayesian Optimization (AETRBO):** This algorithm combines the strengths of Adaptive Trust Region Bayesian Optimization (ATBO) and Bayesian Evolutionary Optimization (BEO) for improved performance and robustness. It employs a Gaussian Process (GP) surrogate model within a trust region framework, similar to ATBO. However, instead of randomly sampling points within the trust region, it uses differential evolution (DE), inspired by BEO, to select candidate points that optimize the acquisition function (Lower Confidence Bound - LCB). The trust region radius is adaptively adjusted based on the success of the DE search. This allows for efficient exploration and exploitation of the search space. Gradient estimation is avoided to maintain computational efficiency and avoid potential budget overruns.


- AdaptiveContextualTrustRegionBO: 0.1729, 13.87 seconds, **Adaptive Contextual Trust Region Bayesian Optimization (ACTRBO):** This algorithm combines the strengths of Adaptive Trust Region Bayesian Optimization (ATBO) and Context-Aware Bayesian Optimization (CABO) while addressing their limitations. It uses a Gaussian Process (GP) surrogate model, a trust region approach that dynamically adjusts its size based on success, and a context-aware acquisition function that penalizes points close to existing points in regions of high confidence. To avoid the `IndexError` observed in `ContextAwareBO`, the point selection is modified to ensure indices are within bounds. Furthermore, the exploration-exploitation balance is dynamically adjusted based on the trust region's success.


- AdaptiveTrustRegionBO: 0.1722, 6.36 seconds, **AdaptiveTrustRegionBO (ATBO):** This algorithm implements a trust region approach within a Bayesian Optimization framework. It uses a Gaussian Process (GP) to model the objective function and dynamically adjusts the size of the trust region based on the GP's predictive performance. The acquisition function is the lower confidence bound (LCB), which balances exploration and exploitation. To further enhance exploration, especially in high-dimensional spaces, the algorithm incorporates a random restart mechanism. The trust region is centered around the best point found so far.


- AdaptiveTrustRegionBO: 0.1682, 191.18 seconds, **AdaptiveTrustRegionBO (ATBO) - Enhanced Exploration and Exploitation:** This enhanced version of ATBO focuses on improving the balance between exploration and exploitation within the trust region framework. It introduces a dynamic kappa parameter in the Lower Confidence Bound (LCB) acquisition function, which adjusts based on the optimization progress. This allows the algorithm to prioritize exploration in the early stages and gradually shift towards exploitation as it converges. Additionally, the trust region sampling strategy is refined to generate more diverse candidate points, and the Gaussian Process (GP) model is updated with a more flexible kernel.


- TrustRegionThompsonSamplingBO: 0.1660, 6.21 seconds, **Trust Region Thompson Sampling Bayesian Optimization (TRTSBO):** This algorithm combines the trust region approach from AdaptiveTrustRegionBO (ATBO) with the Thompson Sampling acquisition function and batch selection strategy from Efficient Hybrid Bayesian Optimization (EHBBO). Specifically, it uses a Gaussian Process (GP) to model the objective function, a trust region to constrain the search space around the current best point, and Thompson Sampling to select candidate points within the trust region. A k-means clustering strategy is then used to select a diverse batch of points from the candidates. The trust region radius is dynamically adjusted based on the success of previous iterations. This approach aims to balance exploration and exploitation effectively while maintaining computational efficiency.


- TrustRegionEnsembleBO: 0.1629, 23.44 seconds, **Trust Region Ensemble Bayesian Optimization (TREBO):** This algorithm combines the strengths of Adaptive Trust Region BO (ATBO) and Surrogate Ensemble BO (SEBO). It uses an ensemble of Gaussian Process (GP) models within a trust region framework. The trust region adapts based on the success of previous iterations, and the ensemble provides robustness by averaging predictions from multiple GPs with different kernels. A key innovation is the use of Thompson Sampling with the ensemble to balance exploration and exploitation within the trust region. The trust region is centered around the best point found so far. By using Thompson Sampling on the ensemble, we aim to improve exploration and avoid premature convergence.


- AdaptiveBayesianEvolutionaryBO: 0.1585, 4999.31 seconds, **Adaptive Bayesian Evolutionary Optimization with Dynamic Kernel Tuning (ABEO-DKT):** This algorithm enhances the original Bayesian Evolutionary Optimization (BEO) by incorporating adaptive kernel tuning for the Gaussian Process (GP) and dynamically adjusting the diversity term in the acquisition function. The GP kernel parameters (length_scale) are optimized using a gradient-based method (L-BFGS-B) during the GP fitting stage. This allows the GP to better capture the underlying function's characteristics. The diversity term in the acquisition function is also dynamically adjusted based on the exploration-exploitation balance. Furthermore, the batch size is adaptively adjusted based on the budget and the dimensionality of the problem.




The selected solution to update is:
**Trust Region Thompson Sampling Bayesian Optimization (TRTSBO):** This algorithm combines the trust region approach from AdaptiveTrustRegionBO (ATBO) with the Thompson Sampling acquisition function and batch selection strategy from Efficient Hybrid Bayesian Optimization (EHBBO). Specifically, it uses a Gaussian Process (GP) to model the objective function, a trust region to constrain the search space around the current best point, and Thompson Sampling to select candidate points within the trust region. A k-means clustering strategy is then used to select a diverse batch of points from the candidates. The trust region radius is dynamically adjusted based on the success of previous iterations. This approach aims to balance exploration and exploitation effectively while maintaining computational efficiency.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.cluster import KMeans

class TrustRegionThompsonSamplingBO:
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
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.radius_decay = 0.95
        self.radius_increase = 1.1
        self.min_radius = 0.1
        self.success_ratio = 0.0 #ratio to track the success of trust region
        self.random_restart_prob = 0.05 #Probability of random restart

    def _sample_points(self, n_points):
        # sample points within the trust region or randomly
        # return array of shape (n_points, n_dims)
        if self.best_x is None or np.random.rand() < self.random_restart_prob:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            points = []
            while len(points) < n_points:
                sample = np.random.uniform(-1, 1, size=(self.dim,))
                sample = self.best_x + self.trust_region_radius * sample
                # Clip to bounds
                sample = np.clip(sample, self.bounds[0], self.bounds[1])
                points.append(sample)
            return np.array(points)

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function: Thompson Sampling
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))  # Return random values if GP is not fitted yet
        else:
            y_samples = self.gp.sample_y(X, n_samples=1)
            return y_samples

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        n_candidates = min(100 * self.dim, self.budget - self.n_evals)
        candidates = self._sample_points(n_candidates)
        acquisition_values = self._acquisition_function(candidates)

        # Cluster the candidates using k-means
        n_clusters = min(batch_size, n_candidates)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(candidates)
        cluster_ids = kmeans.labels_

        # Select the best candidate from each cluster
        next_points = []
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_ids == i)[0]
            best_index = cluster_indices[np.argmin(acquisition_values[cluster_indices])]
            next_points.append(candidates[best_index])

        return np.array(next_points)

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
            self.success_ratio = 1.0 #reset success ratio
        else:
            self.success_ratio *= 0.75 #reduce success ratio if not improving

    def _adjust_trust_region(self):
        # Adjust the trust region size based on the success
        if self.success_ratio > 0.5:
            self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)

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
        batch_size = min(10, self.dim)
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Select points by acquisition function
            next_X = self._select_next_points(batch_size)

            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)

            # Update the evaluated points
            self._update_eval_points(next_X, next_y)

            # Adjust trust region radius
            self._adjust_trust_region()

        return self.best_y, self.best_x

```
The algorithm TrustRegionThompsonSamplingBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1660 with standard deviation 0.0991.

took 6.21 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

