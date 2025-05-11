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
**AdaptiveEvolutionaryTrustRegionBO with Dynamic Kappa, Adaptive DE Iterations, Enhanced Trust Region Adaptation, and Kernel Bandwidth Adjustment (AETRBO-DKAEB):** This algorithm builds upon AETRBO-DKADI by incorporating dynamic adjustment of the Gaussian Process (GP) kernel's bandwidth (length_scale). It adaptively adjusts the bandwidth based on the optimization progress and the estimated landscape complexity. A wider bandwidth is used initially for global exploration, and the bandwidth is reduced as the algorithm converges to exploit local optima. The trust region adaptation is also improved by incorporating a measure of the GP model's prediction error within the trust region, and the initial trust region radius is scaled based on dimensionality.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution
from numpy.linalg import norm as linalg_norm

class AdaptiveEvolutionaryTrustRegionBO_DKAEB:
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
        self.random_restart_prob = 0.05
        self.de_pop_size = 10  # Population size for differential evolution
        self.lcb_kappa = 2.0  # Kappa parameter for Lower Confidence Bound
        self.kappa_decay = 0.99 # Decay rate for kappa
        self.min_kappa = 0.1 # Minimum value for kappa
        self.initial_length_scale = 1.0
        self.length_scale = self.initial_length_scale
        self.length_scale_decay = 0.95
        self.length_scale_increase = 1.05
        self.min_length_scale = 0.01
        self.max_length_scale = 100.0
        self.initial_trust_region_radius = min(2.0, 5.0 / np.sqrt(dim)) # Scale initial radius with dimension

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
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.length_scale, length_scale_bounds=(1e-2, 1e2))
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

    def _adjust_trust_region(self):
        # Adjust the trust region size based on the success
        if self.gp is not None and self.best_x is not None:
            mu, sigma = self.gp.predict(self.X, return_std=True)
            error = np.mean(np.abs(mu.flatten() - self.y.flatten()))
        else:
            error = 0.0

        if self.success_ratio > 0.5:
            self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)

        # Adjust more aggressively if GP predictions are inaccurate
        if error > 0.5:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)

    def _adjust_kernel_bandwidth(self):
        # Adjust kernel bandwidth based on optimization progress
        if self.gp is not None and self.best_x is not None:
            # Estimate landscape complexity using the gradient norm of the GP mean function
            delta = 1e-4  # Small perturbation for gradient estimation
            gradient = np.zeros_like(self.best_x)
            for i in range(self.dim):
                x_plus = self.best_x.copy()
                x_minus = self.best_x.copy()
                x_plus[i] += delta
                x_minus[i] -= delta
                mu_plus, _ = self.gp.predict(x_plus.reshape(1, -1), return_std=True)
                mu_minus, _ = self.gp.predict(x_minus.reshape(1, -1), return_std=True)
                gradient[i] = (mu_plus - mu_minus) / (2 * delta)
            gradient_norm = linalg_norm(gradient)

            if gradient_norm > 1.0:  # High complexity, reduce bandwidth
                self.length_scale = max(self.length_scale * self.length_scale_decay, self.min_length_scale)
            else:  # Low complexity, increase bandwidth
                self.length_scale = min(self.length_scale * self.length_scale_increase, self.max_length_scale)
        else:
            self.length_scale = self.initial_length_scale

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
        self.trust_region_radius = self.initial_trust_region_radius
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

            # Decay kappa
            self.lcb_kappa = max(self.lcb_kappa * self.kappa_decay, self.min_kappa)

            # Adjust kernel bandwidth
            self._adjust_kernel_bandwidth()

        return self.best_y, self.best_x

```
The algorithm AdaptiveEvolutionaryTrustRegionBO_DKAEB got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1813 with standard deviation 0.1033.

took 316.03 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

