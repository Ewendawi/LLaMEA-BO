You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- AdaptiveEvolutionaryTrustRegionBO_DKAE: 0.1846, 229.81 seconds, **AdaptiveEvolutionaryTrustRegionBO with Dynamic Kappa, Adaptive DE Iterations, and Enhanced Trust Region Adaptation (AETRBO-DKAE):** This algorithm builds upon AETRBO-DKADI by introducing a more sophisticated trust region adaptation strategy. Instead of relying solely on the success ratio, it incorporates a measure of the GP model's prediction error within the trust region. If the GP model's predictions are consistently inaccurate, the trust region is shrunk more aggressively to encourage exploration. Additionally, the algorithm dynamically adjusts the lower bound of the kappa parameter based on the observed noise level in the objective function. This allows for more robust performance in noisy environments. Finally, the initial trust region radius is made dependent on the dimensionality of the problem, scaling it appropriately for high-dimensional spaces.


- AdaptiveEvolutionaryParetoTrustRegionBO: 0.1827, 262.21 seconds, **AdaptiveEvolutionaryParetoTrustRegionBO (AEPTRBO):** This algorithm combines the strengths of AdaptiveEvolutionaryTrustRegionBO_DKAE and AdaptiveParetoTrustRegionBO_DKAW, aiming for improved exploration-exploitation balance and robustness. It uses a trust region framework with adaptive radius adjustment based on success ratio and GP model error. Within the trust region, it employs a Pareto-based approach to balance Expected Improvement (EI) and diversity. Differential Evolution (DE) is used to efficiently search for Pareto optimal points within the trust region. Dynamic kernel selection for the GP model and adaptive weighting of EI and diversity in the Pareto front construction are also incorporated. The LCB kappa parameter is dynamically adjusted based on noise estimation.


- AdaptiveEvolutionaryParetoTrustRegionBO: 0.1814, 205.34 seconds, **AdaptiveEvolutionaryParetoTrustRegionBO (AEPTRBO):** This algorithm combines the strengths of AdaptiveEvolutionaryTrustRegionBO_DKAE (AETRBO-DKAE) and AdaptiveParetoLCBTrustRegionBO (APLCB-TRBO) to achieve a more robust and efficient Bayesian Optimization. It uses a trust region framework with adaptive radius adjustment, similar to AETRBO-DKAE. Within the trust region, it employs a Pareto-based approach, inspired by APLCB-TRBO, to balance exploration (diversity) and exploitation (LCB). However, instead of sampling candidate points and then selecting the Pareto front, it uses differential evolution (DE), like AETRBO-DKAE, to directly optimize the Pareto front, considering both LCB and diversity as objectives. A dynamic weighting scheme is introduced to adjust the importance of LCB and diversity based on the optimization progress and the trust region's success. This dynamic weighting allows the algorithm to adapt its exploration-exploitation balance more effectively.


- AdaptiveEvolutionaryTrustRegionBO_DKAEB: 0.1813, 316.03 seconds, **AdaptiveEvolutionaryTrustRegionBO with Dynamic Kappa, Adaptive DE Iterations, Enhanced Trust Region Adaptation, and Kernel Bandwidth Adjustment (AETRBO-DKAEB):** This algorithm builds upon AETRBO-DKADI by incorporating dynamic adjustment of the Gaussian Process (GP) kernel's bandwidth (length_scale). It adaptively adjusts the bandwidth based on the optimization progress and the estimated landscape complexity. A wider bandwidth is used initially for global exploration, and the bandwidth is reduced as the algorithm converges to exploit local optima. The trust region adaptation is also improved by incorporating a measure of the GP model's prediction error within the trust region, and the initial trust region radius is scaled based on dimensionality.


- AdaptiveEvolutionaryParetoContextualTrustRegionBO_DKAK: 0.1811, 584.75 seconds, **Adaptive Evolutionary Pareto Contextual Trust Region Bayesian Optimization with Dynamic Kappa and Adaptive Kernel (AEPCTRBO-DKAK):** This algorithm combines the strengths of AdaptiveEvolutionaryTrustRegionBO_DKAE and AdaptiveParetoContextualTrustRegionBO_DKP. It uses a trust region approach with adaptive radius. Inside the trust region, it employs a Pareto-based approach to balance exploration (diversity and context penalty) and exploitation (Lower Confidence Bound). Differential Evolution (DE) is used to search for candidate points within the trust region, optimizing the Pareto front. The LCB's kappa parameter is dynamically adjusted, and the GP kernel is dynamically tuned using L-BFGS-B optimization. The context penalty discourages sampling near existing points, promoting diversity. The number of DE iterations is also dynamically adjusted based on the remaining budget.


- AdaptiveContextKernelEvolutionaryTrustRegionBO: 0.1807, 395.81 seconds, **Adaptive Context and Kernel Evolutionary Trust Region Bayesian Optimization (ACKETRBO):** This algorithm builds upon ContextualEvolutionaryTrustRegionBO by incorporating dynamic kernel adaptation for the Gaussian Process (GP) and dynamically adjusting the context penalty based on the optimization progress. A set of predefined kernels (RBF with different length scales) are evaluated periodically, and the best performing kernel based on the marginal log-likelihood is selected. The context penalty is also dynamically adjusted based on the success of previous iterations in reducing the objective function value. If recent steps have not led to significant improvement, the context penalty is reduced to encourage exploration. Additionally, the LCB kappa parameter is annealed over the optimization process, starting with a higher value for exploration and gradually decreasing it to promote exploitation. The batch size for evaluating new points is also dynamically adjusted based on the remaining budget.


- AdaptiveParetoLCBTrustRegionBO: 0.1806, 11.48 seconds, **Adaptive Pareto Lower Confidence Bound Trust Region Bayesian Optimization (APLCB-TRBO):** This algorithm combines the strengths of Adaptive Trust Region Bayesian Optimization (ATBO) with a Pareto-based approach for balancing exploration and exploitation using the Lower Confidence Bound (LCB) acquisition function. Instead of directly optimizing the LCB, it considers both the LCB value and a diversity metric in a Pareto sense. The algorithm samples candidate points within a trust region, evaluates their LCB and diversity, and then selects the point on the Pareto front with the highest LCB value for evaluation. The trust region radius is adapted based on the success of previous iterations. The key difference from AdaptiveParetoTrustRegionBO is the use of LCB instead of Expected Improvement as the primary acquisition function, and a more efficient Pareto selection based on LCB.


- AdaptiveContextualTrustRegionBO_EDP: 0.1798, 741.12 seconds, **Adaptive Contextual Trust Region Bayesian Optimization with Ensemble and Dynamic Penalty (ACTREBO-EDP):** This algorithm builds upon AdaptiveEvolutionaryTrustRegionBO_DKAE and ContextualEvolutionaryTrustRegionBO by integrating an ensemble of Gaussian Process (GP) models, a dynamic context penalty, and an enhanced trust region adaptation strategy. The ensemble of GPs improves the robustness of the surrogate model and provides better uncertainty estimates. The dynamic context penalty encourages exploration by penalizing points close to existing samples, while the trust region is adapted based on GP error, success ratio, and the diversity of sampled points. Differential Evolution (DE) is used within the trust region to efficiently search for points optimizing a context-aware acquisition function based on LCB. The kernel parameters of the GPs are dynamically adjusted using a simple grid search.




The selected solution to update is:
**Adaptive Evolutionary Pareto Contextual Trust Region Bayesian Optimization with Dynamic Kappa and Adaptive Kernel (AEPCTRBO-DKAK):** This algorithm combines the strengths of AdaptiveEvolutionaryTrustRegionBO_DKAE and AdaptiveParetoContextualTrustRegionBO_DKP. It uses a trust region approach with adaptive radius. Inside the trust region, it employs a Pareto-based approach to balance exploration (diversity and context penalty) and exploitation (Lower Confidence Bound). Differential Evolution (DE) is used to search for candidate points within the trust region, optimizing the Pareto front. The LCB's kappa parameter is dynamically adjusted, and the GP kernel is dynamically tuned using L-BFGS-B optimization. The context penalty discourages sampling near existing points, promoting diversity. The number of DE iterations is also dynamically adjusted based on the remaining budget.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize, differential_evolution

class AdaptiveEvolutionaryParetoContextualTrustRegionBO_DKAK:
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
        self.trust_region_radius = 2.0 * np.sqrt(dim)
        self.radius_decay = 0.95
        self.radius_increase = 1.1
        self.min_radius = 0.1
        self.success_ratio = 0.0
        self.random_restart_prob = 0.05

        self.knn = NearestNeighbors(n_neighbors=5)
        self.context_penalty = 0.1
        self.initial_context_penalty = 0.1
        self.context_penalty_decay = 0.95
        self.context_penalty_increase = 1.05
        self.min_context_penalty = 0.01
        self.max_context_penalty = 1.0

        self.de_pop_size = 10
        self.lcb_kappa = 2.0
        self.kappa_decay = 0.99
        self.min_kappa = 0.1
        self.noise_estimate = 1e-4

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
        # Kernel optimization
        def neg_log_likelihood(theta):
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=theta, length_scale_bounds="fixed")
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
            gp.fit(X, y)
            return -gp.log_marginal_likelihood()

        # Initial guess for kernel parameters
        initial_length_scale = 1.0

        # Optimize kernel parameters using L-BFGS-B
        result = minimize(neg_log_likelihood, initial_length_scale, method='L-BFGS-B', bounds=[(1e-5, 10.0)])
        optimized_length_scale = result.x[0]

        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=optimized_length_scale, length_scale_bounds="fixed")
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        self.gp.fit(X, y)
        self.knn.fit(X)
        return self.gp

    def _lower_confidence_bound(self, X):
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            mu, sigma = self.gp.predict(X, return_std=True)
            lcb = mu - self.lcb_kappa * sigma
            return lcb.reshape(-1, 1)

    def _diversity_metric(self, X):
        if self.X is None:
            return np.ones((len(X), 1))
        else:
            distances = np.min(cdist(X, self.X), axis=1)
            return distances.reshape(-1, 1)

    def _context_penalty_metric(self, X):
        if self.X is None:
            return np.zeros((len(X), 1))
        else:
            distances, _ = self.knn.kneighbors(X)
            context_penalty = np.mean(distances, axis=1).reshape(-1, 1)
            return context_penalty

    def _select_next_points(self, batch_size):
        if self.best_x is None:
            return self._sample_points(batch_size)

        def de_objective(x):
            X_candidate = x.reshape(1, -1)
            lcb = self._lower_confidence_bound(X_candidate)
            diversity = self._diversity_metric(X_candidate)
            context_penalty = self._context_penalty_metric(X_candidate) * self.context_penalty

            lcb_normalized = (lcb - np.min(lcb)) / (np.max(lcb) - np.min(lcb)) if np.max(lcb) != np.min(lcb) else np.zeros_like(lcb)
            diversity_normalized = (diversity - np.min(diversity)) / (np.max(diversity) - np.min(diversity)) if np.max(diversity) != np.min(diversity) else np.zeros_like(diversity)
            context_penalty_normalized = (context_penalty - np.min(context_penalty)) / (np.max(context_penalty) - np.min(context_penalty)) if np.max(context_penalty) != np.min(context_penalty) else np.zeros_like(context_penalty)

            F = np.hstack([lcb_normalized, diversity_normalized, context_penalty_normalized])

            # Pareto front calculation (minimization of all objectives)
            is_efficient = np.ones(F.shape[0], dtype=bool)
            for i, c in enumerate(F):
                if is_efficient[i]:
                    is_efficient[is_efficient] = np.any(F[is_efficient] <= c, axis=1)  # Changed to <= for minimization
                    is_efficient[i] = True

            # Return the negative LCB value for DE (DE is a minimizer)
            if np.any(is_efficient):
                lcb_pareto = self._lower_confidence_bound(X_candidate[is_efficient])
                return lcb_pareto[0, 0]  # Return LCB of the first Pareto point
            else:
                return lcb[0, 0] #Return LCB if no pareto point is found

        de_bounds = [(max(self.bounds[0][i], self.best_x[i] - self.trust_region_radius),
                      min(self.bounds[1][i], self.best_x[i] + self.trust_region_radius)) for i in range(self.dim)]

        remaining_evals = self.budget - self.n_evals
        maxiter = max(1, int(remaining_evals / (self.de_pop_size * self.dim * 2)))
        maxiter = min(maxiter, 100)

        result = differential_evolution(de_objective, de_bounds, popsize=self.de_pop_size, maxiter=maxiter, tol=0.01, disp=False)
        return result.x.reshape(1, -1)

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

        self.noise_estimate = np.var(self.y)

    def _adjust_trust_region(self):
        if self.success_ratio > 0.5:
            self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)

    def _adjust_context_penalty(self):
        if self.success_ratio > 0.5:
            self.context_penalty = min(self.context_penalty * self.context_penalty_increase, self.max_context_penalty)
        else:
            self.context_penalty = max(self.context_penalty * self.context_penalty_decay, self.min_context_penalty)

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
            self._adjust_context_penalty()

            self.lcb_kappa = max(self.lcb_kappa * self.kappa_decay, self.min_kappa)
            self.min_kappa = 0.1 + np.sqrt(self.noise_estimate)

        return self.best_y, self.best_x

```
The algorithm AdaptiveEvolutionaryParetoContextualTrustRegionBO_DKAK got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1811 with standard deviation 0.1096.

took 584.75 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

