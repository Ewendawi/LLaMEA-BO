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
**Adaptive Batch Ensemble with Thompson Sampling, Density-Aware Exploration, and Uncertainty-Aware Local Search Bayesian Optimization (ABETSALSDEBO):** This algorithm combines the strengths of ABDAHBO and AEHTSALSBO. It uses an adaptive ensemble of Gaussian Process Regression (GPR) models, similar to AEHTSALSBO, dynamically adjusting the ensemble size based on optimization progress. It employs Thompson Sampling for efficient acquisition within the ensemble. The batch size is adaptively adjusted based on the average uncertainty of the GPR predictions, as in ABDAHBO. Furthermore, it incorporates an uncertainty-aware local search, using the variance predictions from the GPR models to guide the local search iterations and step size, similar to AEHTSALSBO. It also includes a Kernel Density Estimation (KDE) to focus the search on high-density regions of promising solutions, with the bandwidth dynamically adjusted based on the local density of evaluated points, as in ABDAHBO. The acquisition function is a hybrid of Expected Improvement (EI), a distance-based exploration term, and a KDE-based exploration term. The exploration weight decay is also adjusted based on the batch size, ensuring a more robust exploration-exploitation trade-off.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KernelDensity, NearestNeighbors
from scipy.optimize import minimize

class ABETSALSDEBO:
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
        self.max_batch_size = min(10, dim)  # Maximum batch size for selecting points
        self.min_batch_size = 1
        self.exploration_weight = 0.1  # Initial exploration weight
        self.exploration_decay = 0.995  # Decay factor for exploration weight
        self.min_exploration = 0.01  # Minimum exploration weight
        self.uncertainty_threshold = 0.5  # Threshold for adjusting batch size
        self.local_search_step_size_factor = 0.1

        self.max_models = 5  # Maximum number of surrogate models in the ensemble
        self.min_models = 1  # Minimum number of surrogate models in the ensemble
        self.models = []
        for i in range(self.max_models):
            length_scale = 1.0 * (i + 1) / self.max_models
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
            model = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=5, alpha=1e-5
            )
            self.models.append(model)

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Adapt ensemble size
        n_models = max(
            self.min_models, int(self.max_models * (1 - self.n_evals / self.budget))
        )
        active_models = self.models[:n_models]
        for model in active_models:
            model.fit(X, y)
        return active_models

    def _acquisition_function(self, X, active_models):
        # Thompson Sampling
        sampled_values = np.zeros((X.shape[0], len(active_models)))
        sigmas = np.zeros((X.shape[0], len(active_models)))
        for i, model in enumerate(active_models):
            mu, sigma = model.predict(X, return_std=True)
            mu = mu.reshape(-1, 1)
            sigma = sigma.reshape(-1, 1)
            sampled_values[:, i] = np.random.normal(mu.flatten(), sigma.flatten())
            sigmas[:, i] = sigma.flatten()

        acquisition_ts = np.mean(sampled_values, axis=1, keepdims=True)
        acquisition_ts = acquisition_ts.reshape(-1, 1)

        # Hybrid acquisition function (EI + exploration + KDE)
        mu = np.mean([model.predict(X) for model in active_models], axis=0).reshape(
            -1, 1
        )
        sigma = np.mean(sigmas, axis=1).reshape(-1, 1)

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
            ei
            + self.exploration_weight * exploration
            + kde_weight * kde_exploration
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

    def _select_next_points(self, batch_size, active_models):
        candidate_points = self._sample_points(100 * batch_size)
        acquisition_values = self._acquisition_function(candidate_points, active_models)
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]

        # Adaptive Local search
        for i in range(batch_size):
            def obj_func(x):
                x = x.reshape(1, -1)
                return np.mean([model.predict(x)[0] for model in active_models])

            x0 = next_points[i]
            bounds = [(self.bounds[0][j], self.bounds[1][j]) for j in range(self.dim)]

            # Uncertainty-aware local search
            uncertainty = np.mean([model.predict(x0.reshape(1, -1), return_std=True)[1] for model in active_models])
            maxiter = int(5 + 10 * uncertainty)
            maxiter = min(maxiter, 20)

            # Adaptive step size
            step_size = self.local_search_step_size_factor * uncertainty
            options = {'maxiter': maxiter, 'ftol': 1e-4}  # Reduced ftol
            res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds, options=options)
            next_points[i] = res.x

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

        active_models = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            remaining_evals = self.budget - self.n_evals

            # Adjust batch size based on uncertainty
            sigmas = []
            for model in active_models:
                _, sigma = model.predict(self.X, return_std=True)
                sigmas.append(np.mean(sigma))
            avg_sigma = np.mean(sigmas)

            if avg_sigma > self.uncertainty_threshold:
                batch_size = self.max_batch_size
                exploration_decay = 0.99
            else:
                batch_size = self.min_batch_size
                exploration_decay = 0.999

            batch_size = min(batch_size, remaining_evals)  # Adjust batch size to budget

            next_X = self._select_next_points(batch_size, active_models)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            active_models = self._fit_model(self.X, self.y)

            # Update exploration weight
            self.exploration_weight = max(
                self.exploration_weight * exploration_decay, self.min_exploration
            )

        return self.best_y, self.best_x

```
The algorithm ABETSALSDEBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1804 with standard deviation 0.1137.

took 582.71 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

