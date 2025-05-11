You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- AdaptiveTrustRegionEvolutionaryBO: 0.1929, 431.99 seconds, **Adaptive Trust Region Evolutionary Bayesian Optimization (ATREBO):** This algorithm synergistically combines the strengths of Trust Region Bayesian Optimization (TRBO) and Bayesian Evolutionary Optimization (BEO). It employs a Gaussian Process (GP) surrogate model, adaptively adjusts a trust region, and uses differential evolution (DE) within the trust region to optimize the acquisition function. The trust region size is dynamically adjusted based on the success of the DE search. Furthermore, ATREBO incorporates a mechanism to adapt the mutation rate of the DE based on the success rate of recent generations, enhancing its adaptability. To avoid the error encountered in BEO, the mutation parameter for DE is carefully controlled within valid bounds.


- AdaptiveTrustRegionBO: 0.1851, 21.23 seconds, **AdaptiveTrustRegionBO (ATrBO):** This algorithm refines the TrustRegionBO by introducing adaptive scaling of the Expected Improvement (EI) acquisition function and dynamic adjustment of local search radius. The EI is scaled by a factor that depends on the GP's predictive variance, promoting exploration in uncertain regions. The local search radius is dynamically adjusted based on the success rate of previous local searches. A higher success rate leads to a smaller search radius, enabling finer exploitation, while a lower success rate increases the radius to explore more broadly. Additionally, the Gaussian Process kernel is updated dynamically by optimizing the length_scale during the fitting process, allowing the GP to adapt to the function's landscape more effectively.


- TrustRegionAdaptiveVarianceBO: 0.1819, 1.41 seconds, **Trust Region Adaptive Variance Bayesian Optimization (TRAVBO):** This algorithm combines the trust region approach of TrustRegionBO with the adaptive variance exploration strategy of AdaptiveVarianceBO. It uses a Gaussian Process (GP) surrogate model within a trust region framework, adaptively adjusting the exploration-exploitation trade-off based on the GP's predictive variance. The acquisition function is modified to incorporate both the trust region constraints and the adaptive variance scaling. The algorithm also includes a mechanism to adapt the trust region size based on the success of the optimization process. The sparse GP approximation is removed, as it caused errors in the previous implementation, and the exploration weight decay is made more robust. A more robust handling of the EI calculation is also included.


- HybridTrustRegionKNNBO: 0.1794, 1.82 seconds, **HybridTrustRegionKNNBO (HTRKBO):** This algorithm combines the strengths of Trust Region Bayesian Optimization (TRBO) and Surrogate Model Free Bayesian Optimization (SMFBO) using a KNN approach. It employs a Gaussian Process (GP) surrogate model within a trust region framework, similar to TRBO, but adaptively adjusts the acquisition function by incorporating information from the nearest neighbors, inspired by SMFBO. Specifically, the acquisition function is a weighted combination of Expected Improvement (EI) from the GP model and a distance-based exploration term from KNN. The weights are dynamically adjusted based on the optimization progress. The trust region size is adaptively adjusted based on the agreement between the GP model's predictions and the actual function evaluations. A local search is performed around the best-observed solution to refine the search.


- AdaptiveTrustRegionQuantileBO: 0.1792, 1.47 seconds, **Adaptive Trust Region Quantile BO (ATRQBO):** This algorithm combines the trust region approach of TRBO with the quantile-based regret minimization of QRBO. It uses a Gaussian Process (GP) surrogate model within a trust region. The acquisition function is based on the Conditional Value at Risk (CVaR) of the regret, similar to QRBO. However, the trust region radius is adaptively adjusted based on the agreement between the GP model's predictions and the actual function evaluations, as in TRBO. Additionally, the quantile level in the CVaR calculation is dynamically adjusted. This hybrid approach aims to leverage the strengths of both TRBO and QRBO, focusing the search on promising regions while being robust to noisy evaluations and outliers. A local search is also performed within the trust region.


- TrustRegionGradientBO: 0.1772, 48.48 seconds, **TrustRegionGradientBO (TRGBO):** This algorithm combines the Trust Region framework from TrustRegionBO with gradient-enhanced exploration inspired by GradientEnhancedBO. It uses a Gaussian Process (GP) surrogate model within a trust region and incorporates an approximate gradient-based term into the acquisition function to encourage exploration in promising directions. To improve efficiency, it employs a dynamically adjusted trust region size and a local search strategy. The gradient is approximated using finite differences within the trust region to keep the computational cost low.


- TrustRegionBO: 0.1770, 1.42 seconds, **TrustRegionBO (TRBO):** This algorithm employs a Gaussian Process (GP) surrogate model within a trust region framework. It starts with an initial trust region size and iteratively refines the GP model within this region. The acquisition function balances exploration and exploitation, and the next point is selected by optimizing this function within the trust region. The trust region size is adaptively adjusted based on the agreement between the GP model's predictions and the actual function evaluations. If the agreement is good, the trust region is expanded; otherwise, it is contracted. This approach focuses the search on promising regions while ensuring sufficient exploration. To enhance computational efficiency, TRBO uses a simple local search within the trust region and periodically re-evaluates the trust region center.


- AdaptiveTrustRegionEnsembleBO: 0.1712, 3.69 seconds, **Adaptive Trust Region Ensemble Bayesian Optimization (ATREBO):** This algorithm combines the strengths of Trust Region Bayesian Optimization (TRBO) and Dynamic Ensemble Bayesian Optimization (DEBO). It uses an ensemble of Gaussian Process (GP) models with different kernels within a trust region framework. The trust region size is adaptively adjusted based on the agreement between the ensemble's predictions and the actual function evaluations. The acquisition function is a weighted combination of Expected Improvement (EI) from each GP model in the ensemble, and a novelty search term to promote exploration. The weights of the GP models are dynamically adjusted based on their predictive performance within the trust region. A local search is performed around the best solution found so far.




The selected solution to update is:
**Trust Region Adaptive Variance Bayesian Optimization (TRAVBO):** This algorithm combines the trust region approach of TrustRegionBO with the adaptive variance exploration strategy of AdaptiveVarianceBO. It uses a Gaussian Process (GP) surrogate model within a trust region framework, adaptively adjusting the exploration-exploitation trade-off based on the GP's predictive variance. The acquisition function is modified to incorporate both the trust region constraints and the adaptive variance scaling. The algorithm also includes a mechanism to adapt the trust region size based on the success of the optimization process. The sparse GP approximation is removed, as it caused errors in the previous implementation, and the exploration weight decay is made more robust. A more robust handling of the EI calculation is also included.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

class TrustRegionAdaptiveVarianceBO:
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
        self.local_search_radius = 0.1
        self.exploration_weight = 0.1 # Initial exploration weight
        self.exploration_decay = 0.95 # Decay factor for exploration weight
        self.min_variance_threshold = 0.01 # Minimum variance threshold for exploration

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
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
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

        # Adaptive exploration weight based on variance
        mean_sigma = np.mean(sigma)
        if mean_sigma < self.min_variance_threshold:
            self.exploration_weight = max(self.exploration_weight * self.exploration_decay, 0.01)  # Reduce exploration, but keep a minimum

        # Expected Improvement acquisition function
        improvement = self.best_y - mu
        Z = improvement / (sigma + 1e-9)  # Add a small constant to avoid division by zero
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # Avoid division by zero
        
        # Incorporate adaptive variance and trust region
        acquisition = ei + self.exploration_weight * sigma
        return acquisition.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Optimize the acquisition function within the trust region
        # return array of shape (batch_size, n_dims)

        # Generate candidate points within the trust region
        x_tries = self._sample_points(batch_size * 10)

        # Clip the points to stay within the bounds
        x_tries = np.clip(x_tries, self.bounds[0], self.bounds[1])

        acq_values = self._acquisition_function(x_tries)

        # Select the top batch_size points based on the acquisition function values
        indices = np.argsort(acq_values.flatten())[::-1][:batch_size]
        return x_tries[indices]

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

            # Select points by acquisition function
            batch_size = min(10, self.budget - self.n_evals)
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

            # Local search around the best solution
            if self.best_x is not None and self.n_evals < self.budget:
                local_X = self._sample_points(1) * self.local_search_radius + self.best_x
                local_X = np.clip(local_X, self.bounds[0], self.bounds[1])
                local_y = self._evaluate_points(func, local_X)
                self._update_eval_points(local_X, local_y)

        return self.best_y, self.best_x

```
The algorithm TrustRegionAdaptiveVarianceBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1819 with standard deviation 0.1114.

took 1.41 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

