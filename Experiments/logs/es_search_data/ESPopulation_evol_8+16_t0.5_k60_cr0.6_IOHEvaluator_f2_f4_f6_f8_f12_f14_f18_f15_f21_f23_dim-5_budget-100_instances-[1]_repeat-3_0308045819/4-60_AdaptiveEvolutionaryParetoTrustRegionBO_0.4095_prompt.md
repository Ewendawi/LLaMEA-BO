You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- AdaptiveEvolutionaryTrustRegionBO_DKAE: 0.1846, 229.81 seconds, **AdaptiveEvolutionaryTrustRegionBO with Dynamic Kappa, Adaptive DE Iterations, and Enhanced Trust Region Adaptation (AETRBO-DKAE):** This algorithm builds upon AETRBO-DKADI by introducing a more sophisticated trust region adaptation strategy. Instead of relying solely on the success ratio, it incorporates a measure of the GP model's prediction error within the trust region. If the GP model's predictions are consistently inaccurate, the trust region is shrunk more aggressively to encourage exploration. Additionally, the algorithm dynamically adjusts the lower bound of the kappa parameter based on the observed noise level in the objective function. This allows for more robust performance in noisy environments. Finally, the initial trust region radius is made dependent on the dimensionality of the problem, scaling it appropriately for high-dimensional spaces.


- AdaptiveParetoLCBTrustRegionBO: 0.1806, 11.48 seconds, **Adaptive Pareto Lower Confidence Bound Trust Region Bayesian Optimization (APLCB-TRBO):** This algorithm combines the strengths of Adaptive Trust Region Bayesian Optimization (ATBO) with a Pareto-based approach for balancing exploration and exploitation using the Lower Confidence Bound (LCB) acquisition function. Instead of directly optimizing the LCB, it considers both the LCB value and a diversity metric in a Pareto sense. The algorithm samples candidate points within a trust region, evaluates their LCB and diversity, and then selects the point on the Pareto front with the highest LCB value for evaluation. The trust region radius is adapted based on the success of previous iterations. The key difference from AdaptiveParetoTrustRegionBO is the use of LCB instead of Expected Improvement as the primary acquisition function, and a more efficient Pareto selection based on LCB.


- ParetoEnsembleAdaptiveTrustRegionBO: 0.1784, 37.47 seconds, **ParetoEnsembleAdaptiveTrustRegionBO (PEATRBO):** This algorithm combines the strengths of Adaptive Pareto Trust Region Bayesian Optimization (APTRBO) and Trust Region Ensemble Bayesian Optimization (TREBO) while addressing their individual limitations. It uses an ensemble of Gaussian Process (GP) models within a trust region framework, similar to TREBO, to improve robustness and model uncertainty. It also incorporates a Pareto-based approach, similar to APTRBO, to balance exploration (diversity) and exploitation (expected improvement) when selecting candidate points within the trust region. A key innovation is the dynamic adjustment of the Pareto front's influence based on the trust region's success, allowing the algorithm to adapt its exploration-exploitation balance more effectively. Furthermore, instead of Thompson Sampling, it uses Expected Improvement for selecting the next point from the Pareto front, while also considering the variance of the GP ensemble. The trust region is centered around the best point found so far, and its radius is adaptively adjusted based on the success of previous iterations.


- AdaptiveEvolutionaryTrustRegionBO_DKADI: 0.1782, 282.06 seconds, **AdaptiveEvolutionaryTrustRegionBO with Dynamic Kappa and Adaptive DE Iterations (AETRBO-DKADI):** This enhanced version of AETRBO introduces dynamic adjustment of the LCB's kappa parameter and adapts the number of iterations in the differential evolution (DE) based on the optimization progress and remaining budget.  The kappa parameter is annealed over time, promoting exploration initially and exploitation later. The number of DE iterations is dynamically adjusted to balance exploration and exploitation within the trust region, taking into account the remaining budget. A more flexible GP kernel with learnable length scales is also introduced.


- AdaptiveParetoTrustRegionBO_DKAW: 0.1779, 31.71 seconds, **Adaptive Pareto Trust Region BO with Dynamic Kernel and Acquisition Weighting (APTRBO-DKAW):** This algorithm builds upon the Adaptive Pareto Trust Region Bayesian Optimization (APTRBO) framework by incorporating two key enhancements: dynamic kernel selection for the Gaussian Process (GP) and adaptive weighting of the Expected Improvement (EI) and diversity components in the Pareto-based acquisition function. The GP kernel is dynamically tuned using a simple grid search over a set of predefined kernel options (RBF with different length scales). The weights for EI and diversity are adjusted based on the optimization progress and the success of previous iterations in improving the best-found solution. This dynamic weighting allows the algorithm to adapt its exploration-exploitation balance more effectively.


- AdaptiveParetoContextualTrustRegionBO_DKP: 0.1779, 110.60 seconds, **Adaptive Pareto Contextual Trust Region Bayesian Optimization with Dynamic Kernel and Penalty (APCTRBO-DKP):** This algorithm integrates the strengths of AdaptiveParetoLCBTrustRegionBO and AdaptiveContextualTrustRegionBO_DCP. It employs a Pareto-based approach within a trust region to balance exploration (diversity) and exploitation (LCB), while also incorporating a dynamic context penalty to discourage sampling near existing points. Furthermore, it dynamically tunes the GP kernel using L-BFGS-B optimization. The dynamic context penalty and kernel optimization are adapted based on the success of previous iterations, allowing for a more flexible and adaptive search strategy. The LCB is used instead of EI for computational efficiency and is combined with diversity in the Pareto sense.


- ContextualEvolutionaryTrustRegionBO: 0.1769, 377.83 seconds, **Contextual Evolutionary Trust Region Bayesian Optimization (CETRBO):** This algorithm synergistically combines the strengths of AdaptiveEvolutionaryTrustRegionBO (AETRBO) and AdaptiveContextualTrustRegionBO (ACTRBO). It employs a Gaussian Process (GP) surrogate model within an adaptive trust region. Inside the trust region, it leverages differential evolution (DE) to efficiently search for points that optimize a context-aware acquisition function. This acquisition function not only considers the Lower Confidence Bound (LCB) for balancing exploration and exploitation but also incorporates a context penalty based on the distances to the nearest neighbors in the evaluated points. This penalty discourages sampling points too close to existing ones, promoting diversity and preventing premature convergence. The trust region radius is adaptively adjusted based on the success of previous iterations. To improve computational efficiency, the GP kernel parameters are fixed.


- AdaptiveParetoTrustRegionBO: 0.1768, 12.67 seconds, **Adaptive Pareto Trust Region Bayesian Optimization (APTRBO):** This algorithm combines the strengths of Adaptive Trust Region Bayesian Optimization (ATBO) and Pareto Active Bayesian Optimization (PABO). It uses a trust region approach to focus the search around promising regions while employing a Pareto-based multi-objective approach to balance exploration (diversity) and exploitation (expected improvement) within the trust region. The trust region radius is adapted based on the success of finding better solutions. Active learning, by querying points with high GP variance, is used to reduce uncertainty, particularly on the Pareto front.




The selected solutions to update are:
## AdaptiveEvolutionaryTrustRegionBO_DKAE
**AdaptiveEvolutionaryTrustRegionBO with Dynamic Kappa, Adaptive DE Iterations, and Enhanced Trust Region Adaptation (AETRBO-DKAE):** This algorithm builds upon AETRBO-DKADI by introducing a more sophisticated trust region adaptation strategy. Instead of relying solely on the success ratio, it incorporates a measure of the GP model's prediction error within the trust region. If the GP model's predictions are consistently inaccurate, the trust region is shrunk more aggressively to encourage exploration. Additionally, the algorithm dynamically adjusts the lower bound of the kappa parameter based on the observed noise level in the objective function. This allows for more robust performance in noisy environments. Finally, the initial trust region radius is made dependent on the dimensionality of the problem, scaling it appropriately for high-dimensional spaces.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution

class AdaptiveEvolutionaryTrustRegionBO_DKAE:
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
        self.trust_region_radius = 2.0 * np.sqrt(dim) # Initial trust region radius, scaled by dimension
        self.radius_decay = 0.95
        self.radius_increase = 1.1
        self.min_radius = 0.1
        self.success_ratio = 0.0 #ratio to track the success of trust region
        self.random_restart_prob = 0.05
        self.de_pop_size = 10  # Population size for differential evolution
        self.lcb_kappa = 2.0  # Kappa parameter for Lower Confidence Bound
        self.kappa_decay = 0.99 # Decay rate for kappa
        self.min_kappa = 0.1 # Minimum value for kappa
        self.gp_error_threshold = 0.1 # Threshold for GP error
        self.noise_estimate = 1e-4 #initial noise estimate

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
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function: Lower Confidence Bound
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            mu, sigma = self.gp.predict(X, return_std=True)
            LCB = mu - self.lcb_kappa * sigma
            return LCB.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using differential evolution within the trust region
        # return array of shape (batch_size, n_dims)

        if self.best_x is None:
            return self._sample_points(batch_size)

        def de_objective(x):
            return self._acquisition_function(x.reshape(1, -1))[0, 0]

        de_bounds = [(max(self.bounds[0][i], self.best_x[i] - self.trust_region_radius),
                      min(self.bounds[1][i], self.best_x[i] + self.trust_region_radius)) for i in range(self.dim)]

        # Adjust maxiter based on remaining budget and optimization progress
        remaining_evals = self.budget - self.n_evals
        maxiter = max(1, int(remaining_evals / (self.de_pop_size * self.dim * 2)))
        maxiter = min(maxiter, 100) #limit maxiter to prevent excessive computation

        result = differential_evolution(de_objective, de_bounds, popsize=self.de_pop_size, maxiter=maxiter, tol=0.01, disp=False)

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
            self.success_ratio = 1.0 #reset success ratio
        else:
            self.success_ratio *= 0.75 #reduce success ratio if not improving

        #Update noise estimate
        self.noise_estimate = np.var(self.y)


    def _adjust_trust_region(self):
        # Adjust the trust region size based on the success
        if self.best_x is not None:
            #Calculate GP error within the trust region
            distances = np.linalg.norm(self.X - self.best_x, axis=1)
            within_tr = distances < self.trust_region_radius
            if np.any(within_tr):
                X_tr = self.X[within_tr]
                y_tr = self.y[within_tr]
                mu, _ = self.gp.predict(X_tr, return_std=True)
                gp_error = np.mean(np.abs(mu.reshape(-1,1) - y_tr))
            else:
                gp_error = 0.0

            if self.success_ratio > 0.5 and gp_error < self.gp_error_threshold:
                self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
            else:
                self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius) #shrink if no best_x

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
        batch_size = min(4, self.dim) #increase batch size to 4 or dim, whichever is smaller
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

            # Decay kappa, adjust min_kappa based on noise estimate
            self.lcb_kappa = max(self.lcb_kappa * self.kappa_decay, self.min_kappa)
            self.min_kappa = 0.1 + np.sqrt(self.noise_estimate) #dynamic min kappa

        return self.best_y, self.best_x

```
The algorithm AdaptiveEvolutionaryTrustRegionBO_DKAE got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1846 with standard deviation 0.0993.

took 229.81 seconds to run.

## AdaptiveParetoTrustRegionBO_DKAW
**Adaptive Pareto Trust Region BO with Dynamic Kernel and Acquisition Weighting (APTRBO-DKAW):** This algorithm builds upon the Adaptive Pareto Trust Region Bayesian Optimization (APTRBO) framework by incorporating two key enhancements: dynamic kernel selection for the Gaussian Process (GP) and adaptive weighting of the Expected Improvement (EI) and diversity components in the Pareto-based acquisition function. The GP kernel is dynamically tuned using a simple grid search over a set of predefined kernel options (RBF with different length scales). The weights for EI and diversity are adjusted based on the optimization progress and the success of previous iterations in improving the best-found solution. This dynamic weighting allows the algorithm to adapt its exploration-exploitation balance more effectively.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

class AdaptiveParetoTrustRegionBO_DKAW:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(5 * dim, self.budget // 10)
        self.gp = None
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0
        self.radius_decay = 0.95
        self.radius_increase = 1.1
        self.min_radius = 0.1
        self.success_ratio = 0.0
        self.random_restart_prob = 0.05
        self.ei_weight = 0.5  # Initial weight for Expected Improvement
        self.diversity_weight = 0.5  # Initial weight for Diversity

    def _sample_points(self, n_points):
        if self.best_x is None or np.random.rand() < self.random_restart_prob:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            points = []
            while len(points) < n_points:
                sample = np.random.uniform(-1, 1, size=(self.dim,))
                sample = self.best_x + self.trust_region_radius * sample
                sample = np.clip(sample, self.bounds[0], self.bounds[1])
                points.append(sample)
            return np.array(points)

    def _fit_model(self, X, y):
        # Dynamic Kernel Selection
        kernels = [
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=0.1, length_scale_bounds="fixed"),
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed"),
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=10.0, length_scale_bounds="fixed"),
        ]
        best_gp = None
        best_log_likelihood = -np.inf

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for kernel in kernels:
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
            gp.fit(X_train, y_train)
            log_likelihood = gp.log_marginal_likelihood(gp.kernel_.theta)

            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_gp = gp

        self.gp = best_gp
        return self.gp

    def _expected_improvement(self, X):
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            mu, sigma = self.gp.predict(X, return_std=True)
            sigma = np.clip(sigma, 1e-9, np.inf)
            imp = self.best_y - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            return ei.reshape(-1, 1)

    def _diversity_metric(self, X):
        if self.X is None:
            return np.ones((len(X), 1))
        else:
            distances = np.min(cdist(X, self.X), axis=1)
            return distances.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        n_candidates = min(100 * self.dim, self.budget - self.n_evals)
        candidates = self._sample_points(n_candidates)

        ei = self._expected_improvement(candidates)
        diversity = self._diversity_metric(candidates)

        ei_normalized = (ei - np.min(ei)) / (np.max(ei) - np.min(ei)) if np.max(ei) != np.min(ei) else np.zeros_like(ei)
        diversity_normalized = (diversity - np.min(diversity)) / (np.max(diversity) - np.min(diversity)) if np.max(diversity) != np.min(diversity) else np.zeros_like(diversity)

        # Adaptive Weighting
        F = np.hstack([self.ei_weight * ei_normalized, self.diversity_weight * diversity_normalized])

        is_efficient = np.ones(F.shape[0], dtype=bool)
        for i, c in enumerate(F):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(F[is_efficient] >= c, axis=1)
                is_efficient[i] = True

        pareto_front = candidates[is_efficient]

        if self.gp is not None and len(pareto_front) > 0:
            _, sigma = self.gp.predict(pareto_front, return_std=True)
            next_point = pareto_front[np.argmax(sigma)].reshape(1, -1)
        elif len(pareto_front) > 0:
            next_point = pareto_front[np.random.choice(len(pareto_front))].reshape(1, -1)
        else:
            # If Pareto front is empty, sample randomly from trust region
            next_point = self._sample_points(1)

        return next_point

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]
            self.success_ratio = 1.0
        else:
            self.success_ratio *= 0.75

    def _adjust_trust_region(self):
        if self.success_ratio > 0.5:
            self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
            # Increase EI weight, decrease diversity weight
            self.ei_weight = min(self.ei_weight + 0.1, 1.0)
            self.diversity_weight = max(self.diversity_weight - 0.1, 0.0)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)
            # Decrease EI weight, increase diversity weight
            self.ei_weight = max(self.ei_weight - 0.1, 0.0)
            self.diversity_weight = min(self.diversity_weight + 0.1, 1.0)

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        batch_size = min(1, self.dim)
        while self.n_evals < self.budget:
            self._fit_model(self.X, self.y)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self._adjust_trust_region()

        return self.best_y, self.best_x

```
The algorithm AdaptiveParetoTrustRegionBO_DKAW got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1779 with standard deviation 0.1066.

took 31.71 seconds to run.

Combine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it. If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.




Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

