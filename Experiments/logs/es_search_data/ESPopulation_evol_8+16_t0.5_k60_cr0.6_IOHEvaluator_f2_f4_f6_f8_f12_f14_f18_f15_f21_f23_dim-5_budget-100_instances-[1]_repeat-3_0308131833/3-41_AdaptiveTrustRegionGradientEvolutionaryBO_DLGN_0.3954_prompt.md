You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- AdaptiveTrustRegionEvolutionaryBO_DKAB: 0.2038, 576.54 seconds, **AdaptiveTrustRegionEvolutionaryBO with Dynamic Kernel and Acquisition Blending (ATREBO-DKAB):** This algorithm builds upon the original ATREBO by incorporating a dynamic kernel within the Gaussian Process (GP) and blending the Expected Improvement (EI) acquisition function with a variance-based exploration term. The GP kernel dynamically adapts its length scale during the optimization process using an optimization routine, allowing the GP to better capture the function's landscape. Furthermore, the acquisition function is a weighted combination of EI and the GP's predictive variance, with the weights dynamically adjusted based on the optimization progress. This balances exploitation and exploration more effectively. Finally, the differential evolution parameters are adapted using a simplified approach to reduce computational cost.


- AdaptiveTrustRegionGradientEvolutionaryBO: 0.1958, 2111.20 seconds, **Adaptive Trust Region Gradient Evolutionary Bayesian Optimization (ATGRBO):** This algorithm combines the strengths of Adaptive Trust Region Evolutionary Bayesian Optimization (ATREBO) and TrustRegionGradientBO (TRGBO). It employs a Gaussian Process (GP) surrogate model within a trust region framework, adaptively adjusting the trust region size based on the success of the optimization process. To enhance exploration, it incorporates an approximate gradient-based term into the acquisition function, similar to TRGBO, and uses differential evolution (DE) within the trust region to optimize the acquisition function, as in ATREBO. The gradient is approximated using finite differences within the trust region to keep the computational cost low. The mutation rate of the DE is also adapted based on the success rate of recent generations. Furthermore, a dynamic weighting scheme is used to balance the Expected Improvement (EI) and gradient-based exploration in the acquisition function.


- AdaptiveTrustRegionEvolutionaryBO: 0.1929, 431.99 seconds, **Adaptive Trust Region Evolutionary Bayesian Optimization (ATREBO):** This algorithm synergistically combines the strengths of Trust Region Bayesian Optimization (TRBO) and Bayesian Evolutionary Optimization (BEO). It employs a Gaussian Process (GP) surrogate model, adaptively adjusts a trust region, and uses differential evolution (DE) within the trust region to optimize the acquisition function. The trust region size is dynamically adjusted based on the success of the DE search. Furthermore, ATREBO incorporates a mechanism to adapt the mutation rate of the DE based on the success rate of recent generations, enhancing its adaptability. To avoid the error encountered in BEO, the mutation parameter for DE is carefully controlled within valid bounds.


- AdaptiveTrustRegionEvolutionaryLocalSearchBO: 0.1879, 212.81 seconds, **Adaptive Trust Region Evolutionary with Local Search Bayesian Optimization (ATRELSBO):** This algorithm combines the strengths of Adaptive Trust Region Evolutionary Bayesian Optimization (ATREBO) and AdaptiveTrustRegionBO (ATrBO) by integrating differential evolution for global search within the trust region, adaptive scaling of the Expected Improvement (EI) acquisition function based on predictive variance, and a dynamic local search strategy. The trust region is adaptively adjusted based on the success of the optimization process. This approach aims to balance global exploration with efficient local exploitation, enhancing the overall optimization performance. The GP kernel is also dynamically updated to better fit the function landscape.


- AdaptiveTrustRegionBO: 0.1851, 21.23 seconds, **AdaptiveTrustRegionBO (ATrBO):** This algorithm refines the TrustRegionBO by introducing adaptive scaling of the Expected Improvement (EI) acquisition function and dynamic adjustment of local search radius. The EI is scaled by a factor that depends on the GP's predictive variance, promoting exploration in uncertain regions. The local search radius is dynamically adjusted based on the success rate of previous local searches. A higher success rate leads to a smaller search radius, enabling finer exploitation, while a lower success rate increases the radius to explore more broadly. Additionally, the Gaussian Process kernel is updated dynamically by optimizing the length_scale during the fitting process, allowing the GP to adapt to the function's landscape more effectively.


- TrustRegionAdaptiveVarianceBO: 0.1819, 1.41 seconds, **Trust Region Adaptive Variance Bayesian Optimization (TRAVBO):** This algorithm combines the trust region approach of TrustRegionBO with the adaptive variance exploration strategy of AdaptiveVarianceBO. It uses a Gaussian Process (GP) surrogate model within a trust region framework, adaptively adjusting the exploration-exploitation trade-off based on the GP's predictive variance. The acquisition function is modified to incorporate both the trust region constraints and the adaptive variance scaling. The algorithm also includes a mechanism to adapt the trust region size based on the success of the optimization process. The sparse GP approximation is removed, as it caused errors in the previous implementation, and the exploration weight decay is made more robust. A more robust handling of the EI calculation is also included.


- AdaptiveTrustRegionEvolutionaryVarianceBO: 0.1806, 233.94 seconds, **Adaptive Trust Region Evolutionary Variance Bayesian Optimization (ATREVOBO):** This algorithm merges the strengths of AdaptiveTrustRegionEvolutionaryBO (ATREBO) and TrustRegionAdaptiveVarianceBO (TRAVBO). It employs a Gaussian Process (GP) surrogate model within an adaptive trust region. The acquisition function combines Expected Improvement (EI) with adaptive variance scaling, similar to TRAVBO, to balance exploration and exploitation. Differential Evolution (DE), as in ATREBO, is used to optimize the acquisition function within the trust region. Furthermore, the algorithm dynamically adjusts the trust region size and DE mutation rate based on optimization progress. A local search is also performed to refine the search. The adaptation of the exploration weight based on the variance of the GP predictions is used to improve exploration in uncertain regions.


- AdaptiveTrustRegionEvolutionaryKNNBO: 0.1799, 537.95 seconds, **Adaptive Trust Region Evolutionary KNN Bayesian Optimization (ATREKBO):** This algorithm combines the strengths of Adaptive Trust Region Evolutionary BO (ATREBO) and Hybrid Trust Region KNN BO (HTRKBO). It leverages a Gaussian Process (GP) surrogate model within a trust region framework, using differential evolution (DE) to optimize the acquisition function, as in ATREBO. However, it enhances the acquisition function by incorporating a KNN-based exploration term, similar to HTRKBO, to balance exploration and exploitation more effectively. The trust region size, DE mutation rate, and the weights for EI and KNN exploration are all adaptively adjusted based on the optimization progress. A local search is performed to refine the search.




The selected solution to update is:
**Adaptive Trust Region Gradient Evolutionary Bayesian Optimization (ATGRBO):** This algorithm combines the strengths of Adaptive Trust Region Evolutionary Bayesian Optimization (ATREBO) and TrustRegionGradientBO (TRGBO). It employs a Gaussian Process (GP) surrogate model within a trust region framework, adaptively adjusting the trust region size based on the success of the optimization process. To enhance exploration, it incorporates an approximate gradient-based term into the acquisition function, similar to TRGBO, and uses differential evolution (DE) within the trust region to optimize the acquisition function, as in ATREBO. The gradient is approximated using finite differences within the trust region to keep the computational cost low. The mutation rate of the DE is also adapted based on the success rate of recent generations. Furthermore, a dynamic weighting scheme is used to balance the Expected Improvement (EI) and gradient-based exploration in the acquisition function.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution

class AdaptiveTrustRegionGradientEvolutionaryBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.gp = None
        self.trust_region_radius = 2.0
        self.trust_region_shrink = 0.5
        self.trust_region_expand = 1.5
        self.success_threshold = 0.75
        self.failure_threshold = 0.25
        self.trust_region_center = np.zeros(dim)
        self.best_x = None
        self.best_y = np.inf
        self.de_popsize = 10
        self.de_mutation = 0.5
        self.de_crossover = 0.7
        self.mutation_adaptation_rate = 0.1
        self.success_threshold_de = 0.2
        self.gradient_weight = 0.1
        self.finite_difference_step = 0.1
        self.ei_gradient_weight_ratio = 0.5 # initial weight ratio
        self.ei_gradient_weight_adapt_rate = 0.1

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        scaled_sample = qmc.scale(sample, -self.trust_region_radius, self.trust_region_radius)
        return scaled_sample + self.trust_region_center

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)
        gp.fit(X, y)
        return gp

    def _approximate_gradients(self, func, x):
        gradients = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.finite_difference_step
            x_minus[i] -= self.finite_difference_step
            x_plus = np.clip(x_plus, self.bounds[0], self.bounds[1])
            x_minus = np.clip(x_minus, self.bounds[0], self.bounds[1])
            gradients[i] = (func(x_plus) - func(x_minus)) / (2 * self.finite_difference_step)
        return gradients

    def _acquisition_function(self, X):
        if self.gp is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.gp.predict(X, return_std=True)
        improvement = self.best_y - mu
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        gradient_exploration = np.zeros_like(ei)
        for i in range(len(X)):
            gradients = self._approximate_gradients(lambda x: self.gp.predict(x.reshape(1, -1))[0], X[i])
            gradient_exploration[i] = self.gradient_weight * np.linalg.norm(gradients)

        # Dynamic weighting
        ei_weight = self.ei_gradient_weight_ratio
        gradient_weight = 1 - self.ei_gradient_weight_ratio
        
        return (ei_weight * ei + gradient_weight * gradient_exploration).reshape(-1, 1)

    def _select_next_points(self, batch_size):
        def de_objective(x):
            return -self._acquisition_function(x.reshape(1, -1))[0, 0]

        de_bounds = list(zip(np.maximum(self.bounds[0], self.trust_region_center - self.trust_region_radius),
                             np.minimum(self.bounds[1], self.trust_region_center + self.trust_region_radius)))

        de_result = differential_evolution(
            func=de_objective,
            bounds=de_bounds,
            popsize=self.de_popsize,
            mutation=self.de_mutation,
            recombination=self.de_crossover,
            maxiter=5,
            tol=0.01,
            seed=None,
            strategy='rand1bin',
            init='latinhypercube'
        )
        
        next_point = de_result.x.reshape(1, -1)
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
        if self.y[best_index, 0] < self.best_y:
            self.best_x = self.X[best_index]
            self.best_y = self.y[best_index, 0]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_X = np.clip(initial_X, self.bounds[0], self.bounds[1])
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        while self.n_evals < self.budget:
            self.gp = self._fit_model(self.X, self.y)

            batch_size = min(1, self.budget - self.n_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            predicted_improvement = self.best_y - self.gp.predict(self.best_x.reshape(1, -1))[0]
            actual_improvement = self.best_y - min(self.y)[0]

            if abs(predicted_improvement) < 1e-9:
                ratio = 0.0
            else:
                ratio = actual_improvement / predicted_improvement

            if ratio > self.success_threshold:
                self.trust_region_radius *= self.trust_region_expand
            elif ratio < self.failure_threshold:
                self.trust_region_radius *= self.trust_region_shrink
                self.trust_region_radius = max(self.trust_region_radius, 0.1)

            self.trust_region_center = self.best_x
            
            if len(self.y) > self.de_popsize:
                recent_ys = self.y[-self.de_popsize:]
                best_recent_y = np.min(recent_ys)
                if best_recent_y < self.best_y + self.success_threshold_de:
                    self.de_mutation *= (1 + self.mutation_adaptation_rate)
                else:
                    self.de_mutation *= (1 - self.mutation_adaptation_rate)
                self.de_mutation = np.clip(self.de_mutation, 0.1, 1.99)

            # Adapt EI/Gradient weight based on GP's predictive variance
            mean_sigma = np.mean(self.gp.predict(self.X, return_std=True)[1])
            self.ei_gradient_weight_ratio += self.ei_gradient_weight_adapt_rate * (1 - mean_sigma)
            self.ei_gradient_weight_ratio = np.clip(self.ei_gradient_weight_ratio, 0.1, 0.9)

        return self.best_y, self.best_x

```
The algorithm AdaptiveTrustRegionGradientEvolutionaryBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1958 with standard deviation 0.1140.

took 2111.20 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

