You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- AEEHBBO: 0.1698, 78.37 seconds, **Adaptive Exploration-Exploitation EHBBO (AEEHBBO):** This algorithm refines the EHBBO algorithm by introducing an adaptive exploration-exploitation balance in the acquisition function and using a more efficient point selection strategy. The exploration weight in the hybrid acquisition function is dynamically adjusted based on the optimization progress. Specifically, the exploration weight decreases as the number of evaluations increases, shifting the focus from exploration to exploitation. Additionally, the point selection strategy is refined by using a combination of random sampling and local search around the best point found so far.


- AHBBO_ABS: 0.1651, 198.40 seconds, **Adaptive Hybrid Bayesian Optimization with Dynamic Exploration and Adaptive Batch Size (AHBBO-ABS):** This algorithm builds upon AHBBO by introducing an adaptive batch size strategy and refining the exploration-exploitation balance. The batch size is dynamically adjusted based on the optimization progress and the uncertainty estimates from the Gaussian Process Regression (GPR) model. Specifically, the batch size increases when the model uncertainty is high, promoting exploration, and decreases when the model is confident, focusing on exploitation. The exploration weight decay is also adjusted based on the batch size, ensuring a more robust exploration-exploitation trade-off.


- AEHTSALSBO: 0.1642, 223.77 seconds, **Adaptive Ensemble Hybrid Bayesian Optimization with Thompson Sampling and Uncertainty-Aware Local Search (AEHTSALSBO):** This algorithm builds upon EHTSALSBO by introducing an adaptive strategy for managing the ensemble of Gaussian Process Regression (GPR) models. It dynamically adjusts the number of models in the ensemble based on the optimization progress, favoring a larger ensemble in the early stages for better exploration and reducing it later for more focused exploitation. Furthermore, it refines the local search by incorporating a more sophisticated uncertainty-aware mechanism, using the variance predictions from the GPR models to guide the local search iterations and step size. The acquisition function combines Expected Improvement (EI) and a distance-based exploration term, while Thompson Sampling is used for efficient acquisition. Latin Hypercube Sampling (LHS) is used for initial sampling.


- EDAHBO: 0.1635, 60.46 seconds, **Efficient Density-Adaptive Hybrid Bayesian Optimization (EDAHBO):** This algorithm combines the strengths of EHBBO and AdaptiveBandwidthDensiTreeBO to achieve efficient and robust black-box optimization. It leverages a Gaussian Process Regression (GPR) model with a hybrid acquisition function (Expected Improvement + Distance-based Exploration) from EHBBO to balance exploration and exploitation. It integrates an adaptive Kernel Density Estimation (KDE) from AdaptiveBandwidthDensiTreeBO to focus the search on high-density regions of promising solutions, with the bandwidth dynamically adjusted based on the local density of evaluated points. A key innovation is a dynamic weighting strategy that adaptively adjusts the influence of the KDE and the distance-based exploration term in the acquisition function based on the optimization progress. This allows the algorithm to initially prioritize exploration and then gradually shift towards exploitation as more information about the function landscape is gathered.


- AHBBO: 0.1619, 57.29 seconds, **Adaptive Hybrid Bayesian Optimization with Dynamic Exploration (AHBBO):** This algorithm refines the EHBBO algorithm by introducing an adaptive exploration strategy. The exploration term in the acquisition function is dynamically adjusted based on the optimization progress. Specifically, the exploration weight decreases as the number of evaluations increases, shifting the focus from exploration to exploitation. Additionally, a lower bound on the exploration weight is introduced to prevent premature convergence.


- AEDDSBO: 0.1619, 58.58 seconds, **Adaptive Exploration with Dynamic Distance Scaling Bayesian Optimization (AEDDSBO):** This algorithm builds upon AHBBO by introducing dynamic scaling of the distance-based exploration term in the acquisition function. It aims to improve the balance between global exploration and local exploitation by adaptively adjusting the influence of the distance term based on the distribution of evaluated points. Additionally, it incorporates a more robust mechanism for exploration weight decay, considering the variance of the GPR predictions.


- EHTSALSDEBO: 0.1611, 406.46 seconds, **Ensemble Hybrid Thompson Sampling with Adaptive Local Search and Exploration Decay Bayesian Optimization (EHTSALSDEBO):** This algorithm builds upon EHTSALSBO by introducing a decaying exploration factor in the hybrid acquisition function. The exploration term, which is distance-based, is multiplied by a factor that decreases with the number of evaluations. This encourages exploration early on and exploitation later in the optimization process. Additionally, the local search is enhanced by incorporating a momentum-based acceleration to escape local optima more effectively.


- EHBBO: 0.1605, 56.90 seconds, **Efficient Hybrid Bayesian Optimization (EHBBO):** This algorithm combines an initial space-filling design using Latin Hypercube Sampling (LHS), a Gaussian Process Regression (GPR) surrogate model, and a hybrid acquisition function that balances exploration and exploitation. The acquisition function combines Expected Improvement (EI) and a distance-based exploration term. A simple but effective batch selection strategy is used to select multiple points for evaluation in each iteration.




The selected solution to update is:
**Efficient Density-Adaptive Hybrid Bayesian Optimization (EDAHBO):** This algorithm combines the strengths of EHBBO and AdaptiveBandwidthDensiTreeBO to achieve efficient and robust black-box optimization. It leverages a Gaussian Process Regression (GPR) model with a hybrid acquisition function (Expected Improvement + Distance-based Exploration) from EHBBO to balance exploration and exploitation. It integrates an adaptive Kernel Density Estimation (KDE) from AdaptiveBandwidthDensiTreeBO to focus the search on high-density regions of promising solutions, with the bandwidth dynamically adjusted based on the local density of evaluated points. A key innovation is a dynamic weighting strategy that adaptively adjusts the influence of the KDE and the distance-based exploration term in the acquisition function based on the optimization progress. This allows the algorithm to initially prioritize exploration and then gradually shift towards exploitation as more information about the function landscape is gathered.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KernelDensity, NearestNeighbors

class EDAHBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.k_neighbors = min(10, 2 * dim)
        self.best_y = np.inf
        self.best_x = None
        self.kde_bandwidth = 0.5
        self.batch_size = min(10, dim)  # Batch size for selecting points

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=5, alpha=1e-5
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Distance-based exploration term
        if self.X is not None and len(self.X) > 0:
            min_dist = np.min(
                np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2),
                axis=1,
                keepdims=True,
            )
            exploration = min_dist / np.max(min_dist)
        else:
            exploration = np.ones(X.shape[0]).reshape(-1, 1)

        # KDE-based exploration term
        if self.X is not None and len(self.X) > self.dim + 1:
            bandwidth = self._adaptive_bandwidth()
            kde = KernelDensity(bandwidth=bandwidth).fit(self.X)
            kde_scores = kde.score_samples(X)
            kde_scores = np.exp(kde_scores).reshape(-1, 1)  # Convert to density
            kde_exploration = kde_scores / np.max(kde_scores)  # Normalize
        else:
            kde_exploration = np.zeros(X.shape[0]).reshape(-1, 1)

        # Dynamic weighting
        exploration_weight = np.clip(1.0 - self.n_evals / self.budget, 0.1, 1.0)
        kde_weight = 1.0 - exploration_weight

        # Hybrid acquisition function
        acquisition = (
            ei + exploration_weight * exploration + kde_weight * kde_exploration
        )
        return acquisition

    def _adaptive_bandwidth(self):
        if self.X is None or len(self.X) < self.k_neighbors:
            return self.kde_bandwidth

        nbrs = NearestNeighbors(
            n_neighbors=self.k_neighbors, algorithm="ball_tree"
        ).fit(self.X)
        distances, _ = nbrs.kneighbors(self.X)
        k_distances = distances[:, -1]
        bandwidth = np.median(k_distances)
        return bandwidth

    def _select_next_points(self, batch_size):
        candidate_points = self._sample_points(100 * batch_size)
        acquisition_values = self._acquisition_function(candidate_points)
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]
        return next_points

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

        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[
        np.float64, np.array
    ]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            remaining_evals = self.budget - self.n_evals
            batch_size = min(self.batch_size, remaining_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self.model = self._fit_model(self.X, self.y)

        return self.best_y, self.best_x

```
The algorithm EDAHBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1635 with standard deviation 0.0962.

took 60.46 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

