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
**Adaptive Trust Region Evolutionary with Local Search Bayesian Optimization (ATRELSBO):** This algorithm combines the strengths of Adaptive Trust Region Evolutionary Bayesian Optimization (ATREBO) and AdaptiveTrustRegionBO (ATrBO) by integrating differential evolution for global search within the trust region, adaptive scaling of the Expected Improvement (EI) acquisition function based on predictive variance, and a dynamic local search strategy. The trust region is adaptively adjusted based on the success of the optimization process. This approach aims to balance global exploration with efficient local exploitation, enhancing the overall optimization performance. The GP kernel is also dynamically updated to better fit the function landscape.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution

class AdaptiveTrustRegionEvolutionaryLocalSearchBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim
        self.gp = None
        self.trust_region_radius = 2.0 # Initial trust region radius
        self.trust_region_shrink = 0.5 # Shrink factor for trust region
        self.trust_region_expand = 1.5 # Expansion factor for trust region
        self.success_threshold = 0.75 # Threshold for trust region expansion
        self.failure_threshold = 0.25 # Threshold for trust region contraction
        self.trust_region_center = np.zeros(dim) # Initial trust region center
        self.best_x = None
        self.best_y = np.inf
        self.de_popsize = 5 # Population size for differential evolution
        self.de_mutation = 0.5 # Initial mutation rate for differential evolution
        self.de_crossover = 0.7 # Crossover rate for differential evolution
        self.mutation_adaptation_rate = 0.1 # Rate to adapt mutation based on success
        self.success_threshold_de = 0.2 # Threshold for considering a generation successful
        self.local_search_radius = 0.1
        self.local_search_success_rate = 0.0
        self.local_search_success_memory = []
        self.local_search_success_window = 5

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points within the trust region
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        scaled_sample = qmc.scale(sample, -self.trust_region_radius, self.trust_region_radius)
        return scaled_sample + self.trust_region_center

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) # Allow length_scale to be optimized
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None:
            return np.zeros((len(X), 1)) # Return zeros if the model is not fitted yet

        mu, sigma = self.gp.predict(X, return_std=True)
        # Expected Improvement acquisition function
        improvement = self.best_y - mu
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # Avoid division by zero

        # Adaptive scaling of EI based on variance
        ei = ei * sigma  # Scale EI by the predictive variance

        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a differential evolution strategy to optimize the acquisition function within the trust region
        # return array of shape (batch_size, n_dims)

        def de_objective(x):
            # Objective function for differential evolution (negative acquisition function)
            return -self._acquisition_function(x.reshape(1, -1))[0, 0]

        # Define bounds for differential evolution within the trust region
        de_bounds = list(zip(np.maximum(self.bounds[0], self.trust_region_center - self.trust_region_radius),
                             np.minimum(self.bounds[1], self.trust_region_center + self.trust_region_radius)))

        # Perform differential evolution
        de_result = differential_evolution(
            func=de_objective,
            bounds=de_bounds,
            popsize=self.de_popsize,
            mutation=self.de_mutation,
            recombination=self.de_crossover,
            maxiter=3, # Reduce maxiter for computational efficiency
            tol=0.01,
            seed=None,
            strategy='rand1bin',
            init='latinhypercube'
        )
        
        # Select the best point from the differential evolution result
        next_point = de_result.x.reshape(1, -1)
        
        return next_point

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
        
        # Update best observed solution
        best_index = np.argmin(self.y)
        if self.y[best_index, 0] < self.best_y:
            self.best_x = self.X[best_index]
            self.best_y = self.y[best_index, 0]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_X = np.clip(initial_X, self.bounds[0], self.bounds[1])
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        while self.n_evals < self.budget:
            # Optimization

            # Fit GP model
            self.gp = self._fit_model(self.X, self.y)

            # Select points by acquisition function using differential evolution
            batch_size = min(1, self.budget - self.n_evals) # Batch size of 1 for evolutionary strategy
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Calculate the ratio of improvement
            predicted_improvement = self.best_y - self.gp.predict(self.best_x.reshape(1, -1))[0]
            actual_improvement = self.best_y - min(self.y)[0]

            if abs(predicted_improvement) < 1e-9:
                ratio = 0.0
            else:
                ratio = actual_improvement / predicted_improvement

            # Adjust trust region size
            if ratio > self.success_threshold:
                self.trust_region_radius *= self.trust_region_expand
            elif ratio < self.failure_threshold:
                self.trust_region_radius *= self.trust_region_shrink
                self.trust_region_radius = max(self.trust_region_radius, 0.1)

            # Update trust region center
            self.trust_region_center = self.best_x
            
            # Adapt mutation rate based on success - Removed mutation adaptation for simplicity and efficiency
            # Local search around the best solution
            if self.best_x is not None and self.n_evals < self.budget:
                local_X = self._sample_points(1) * self.local_search_radius + self.best_x
                local_X = np.clip(local_X, self.bounds[0], self.bounds[1])
                local_y = self._evaluate_points(func, local_X)
                self._update_eval_points(local_X, local_y)

                # Update local search success rate
                if local_y[0, 0] < self.best_y:
                    self.local_search_success_memory.append(1)
                else:
                    self.local_search_success_memory.append(0)

                if len(self.local_search_success_memory) > self.local_search_success_window:
                    self.local_search_success_memory.pop(0)

                self.local_search_success_rate = np.mean(self.local_search_success_memory)

                # Adjust local search radius
                if self.local_search_success_rate > 0.5:
                    self.local_search_radius *= 0.9  # Reduce radius if successful
                else:
                    self.local_search_radius *= 1.1  # Increase radius if unsuccessful
                self.local_search_radius = np.clip(self.local_search_radius, 0.01, 1.0)

        return self.best_y, self.best_x

```
The algorithm AdaptiveTrustRegionEvolutionaryLocalSearchBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1879 with standard deviation 0.1133.

took 212.81 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

