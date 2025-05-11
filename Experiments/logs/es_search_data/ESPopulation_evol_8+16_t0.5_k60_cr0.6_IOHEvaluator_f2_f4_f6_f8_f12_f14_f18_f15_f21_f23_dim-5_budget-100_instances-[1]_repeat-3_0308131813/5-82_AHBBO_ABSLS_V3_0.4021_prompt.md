You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- ABETSALSDEBO: 0.1804, 582.71 seconds, **Adaptive Batch Ensemble with Thompson Sampling, Density-Aware Exploration, and Uncertainty-Aware Local Search Bayesian Optimization (ABETSALSDEBO):** This algorithm combines the strengths of ABDAHBO and AEHTSALSBO. It uses an adaptive ensemble of Gaussian Process Regression (GPR) models, similar to AEHTSALSBO, dynamically adjusting the ensemble size based on optimization progress. It employs Thompson Sampling for efficient acquisition within the ensemble. The batch size is adaptively adjusted based on the average uncertainty of the GPR predictions, as in ABDAHBO. Furthermore, it incorporates an uncertainty-aware local search, using the variance predictions from the GPR models to guide the local search iterations and step size, similar to AEHTSALSBO. It also includes a Kernel Density Estimation (KDE) to focus the search on high-density regions of promising solutions, with the bandwidth dynamically adjusted based on the local density of evaluated points, as in ABDAHBO. The acquisition function is a hybrid of Expected Improvement (EI), a distance-based exploration term, and a KDE-based exploration term. The exploration weight decay is also adjusted based on the batch size, ensuring a more robust exploration-exploitation trade-off.


- AETSALS_KDEBO: 0.1766, 525.90 seconds, **Adaptive Ensemble with Thompson Sampling, Uncertainty-Aware Local Search, and Kernel Density Estimation (AETSALS-KDEBO):** This algorithm combines the strengths of ABETSALSBO and AEHTSALSBO while incorporating Kernel Density Estimation (KDE) to improve exploration and exploitation. It adaptively manages an ensemble of Gaussian Process Regression (GPR) models, dynamically adjusting the ensemble size based on optimization progress. Thompson Sampling is used for efficient acquisition within the ensemble. The batch size is adaptively adjusted based on the average uncertainty of the GPR predictions. An uncertainty-aware local search, using the variance predictions from the GPR models, guides the local search iterations and step size. The acquisition function is a hybrid of Expected Improvement (EI), a distance-based exploration term, and a KDE-based exploration term. The KDE bandwidth is dynamically adjusted based on the local density of evaluated points.


- ABDAHLSBO: 0.1757, 303.63 seconds, **Adaptive Batch-Size Density-Aware Hybrid Bayesian Optimization with Local Search (ABDAHLSBO):** This algorithm combines the strengths of ABDAHBO and AHBBO_ABSLS, incorporating adaptive batch size based on GPR uncertainty, density-aware exploration using KDE, and local search around the best solution found so far. It dynamically adjusts the KDE bandwidth and exploration weight. The adaptive batch size helps to balance exploration and exploitation, while the KDE focuses the search on promising regions. The local search refines the solutions found by the global search.


- AETSALSDEBO: 0.1722, 798.91 seconds, **Adaptive Ensemble with Thompson Sampling, Uncertainty-Aware Local Search, and Dynamic Exploration Weight Bayesian Optimization (AETSALSDEBO):** This algorithm combines the strengths of ABETSALSBO and AEEHBBO, focusing on adaptive exploration-exploitation and efficient local search within an ensemble framework. It uses an ensemble of Gaussian Process Regression (GPR) models with varying length scales to capture different aspects of the function landscape. Thompson Sampling is employed for efficient acquisition, balancing exploration and exploitation within the ensemble. An adaptive local search, guided by the uncertainty estimates from the GPR models, refines the solutions. The exploration weight in the hybrid acquisition function is dynamically adjusted based on both the optimization progress and the GPR model uncertainty, allowing for a robust and efficient exploration-exploitation trade-off. It also incorporates a dynamic ensemble size adjustment based on the optimization progress. Furthermore, a decaying step size is used in the local search to avoid overshooting.


- AHBBO_ABSLS_V2: 0.1721, 299.58 seconds, **Adaptive Hybrid Bayesian Optimization with Dynamic Exploration, Adaptive Batch Size, and Improved Local Search (AHBBO_ABSLS_V2):** This algorithm refines the AHBBO-ABSLS algorithm by introducing improvements to the local search strategy and the exploration-exploitation balance. The local search is enhanced by adaptively adjusting the local search radius based on the optimization progress and the GPR model's uncertainty. Additionally, a more sophisticated exploration weight decay schedule is implemented, incorporating both the number of evaluations and the GPR model uncertainty to ensure a more robust and efficient exploration-exploitation trade-off.


- ABETSALSBO: 0.1719, 575.47 seconds, **Adaptive Batch Ensemble with Thompson Sampling and Uncertainty-Aware Local Search Bayesian Optimization (ABETSALSBO):** This algorithm combines the strengths of AHBBO_ABS and AEHTSALSBO. It adaptively manages an ensemble of Gaussian Process Regression (GPR) models, dynamically adjusting the ensemble size based on optimization progress. It utilizes Thompson Sampling for efficient acquisition within the ensemble. The batch size is adaptively adjusted based on the average uncertainty of the GPR predictions. Furthermore, it incorporates an uncertainty-aware local search, using the variance predictions from the GPR models to guide the local search iterations and step size. A hybrid acquisition function combines Expected Improvement (EI) and a distance-based exploration term.


- ABDAHBO: 0.1711, 208.94 seconds, **Adaptive Batch-Size Density-Aware Hybrid Bayesian Optimization (ABDAHBO):** This algorithm combines the adaptive batch size strategy from AHBBO_ABS with the density-aware exploration from EDAHBO. It dynamically adjusts the batch size based on the uncertainty estimates from the Gaussian Process Regression (GPR) model. It also incorporates a Kernel Density Estimation (KDE) to focus the search on high-density regions of promising solutions, with the bandwidth dynamically adjusted based on the local density of evaluated points. The acquisition function is a hybrid of Expected Improvement (EI), distance-based exploration, and KDE-based exploration, with dynamically adjusted weights. A novel aspect is the adaptive adjustment of the KDE bandwidth based on the local distribution of the evaluated points, improving the accuracy of density estimation. Finally, the exploration weight decay is also adjusted based on the batch size, ensuring a more robust exploration-exploitation trade-off.


- ABETSALSDEBO: 0.1705, 985.75 seconds, **Adaptive Batch Ensemble with Thompson Sampling, Density-Aware Exploration, and Uncertainty-Aware Local Search Bayesian Optimization (ABETSALSDEBO):** This algorithm combines the strengths of ABDAHBO and AETSALSBO. It uses an ensemble of Gaussian Process Regression (GPR) models with varying length scales and Thompson Sampling for efficient acquisition. It incorporates both density-aware exploration using Kernel Density Estimation (KDE) and uncertainty-aware adaptive local search with momentum. The batch size and exploration weight are dynamically adjusted based on the model uncertainty. The KDE bandwidth is also adaptively adjusted based on the local distribution of evaluated points.




The selected solution to update is:
**Adaptive Hybrid Bayesian Optimization with Dynamic Exploration, Adaptive Batch Size, and Improved Local Search (AHBBO_ABSLS_V2):** This algorithm refines the AHBBO-ABSLS algorithm by introducing improvements to the local search strategy and the exploration-exploitation balance. The local search is enhanced by adaptively adjusting the local search radius based on the optimization progress and the GPR model's uncertainty. Additionally, a more sophisticated exploration weight decay schedule is implemented, incorporating both the number of evaluations and the GPR model uncertainty to ensure a more robust and efficient exploration-exploitation trade-off.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class AHBBO_ABSLS_V2:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim # Initial number of points

        self.best_y = np.inf
        self.best_x = None

        self.max_batch_size = min(10, dim) # Maximum batch size for selecting points
        self.min_batch_size = 1
        self.exploration_weight = 0.2 # Initial exploration weight
        self.exploration_weight_min = 0.01 # Minimum exploration weight
        self.uncertainty_threshold = 0.5  # Threshold for adjusting batch size
        self.initial_local_search_radius = 0.2
        self.local_search_radius = self.initial_local_search_radius
        self.local_search_radius_min = 0.01

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
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Distance-based exploration term
        if self.X is not None:
            min_dist = np.min(np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2), axis=1, keepdims=True)
            exploration = min_dist / np.max(min_dist)
        else:
            exploration = np.ones(X.shape[0])[:,None]

        # Hybrid acquisition function
        acquisition = ei + self.exploration_weight * exploration
        return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Generate candidate points
        candidate_points = self._sample_points(50 * batch_size)  # Generate more candidates

        # Add points around the best solution (local search)
        if self.best_x is not None:
            local_points = np.random.normal(loc=self.best_x, scale=self.local_search_radius, size=(50 * batch_size, self.dim))
            local_points = np.clip(local_points, self.bounds[0], self.bounds[1])
            candidate_points = np.vstack((candidate_points, local_points))

        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_points)

        # Select the top batch_size points with the highest acquisition values
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]

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

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Optimization
            # select points by acquisition function

            # Adjust batch size based on uncertainty
            _, sigma = self.model.predict(self.X, return_std=True)
            avg_sigma = np.mean(sigma)

            if avg_sigma > self.uncertainty_threshold:
                batch_size = self.max_batch_size
            else:
                batch_size = self.min_batch_size

            remaining_evals = self.budget - self.n_evals
            batch_size = min(batch_size, remaining_evals) # Adjust batch size to budget

            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self.model = self._fit_model(self.X, self.y)

            # Update exploration weight
            self.exploration_weight = max(self.exploration_weight_min, self.exploration_weight * (1 - self.n_evals / self.budget) * (1 + avg_sigma))

            # Update local search radius
            self.local_search_radius = max(self.local_search_radius_min, self.initial_local_search_radius * (1 - self.n_evals / self.budget) * (1 + avg_sigma))

        return self.best_y, self.best_x

```
The algorithm AHBBO_ABSLS_V2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1721 with standard deviation 0.0975.

took 299.58 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

