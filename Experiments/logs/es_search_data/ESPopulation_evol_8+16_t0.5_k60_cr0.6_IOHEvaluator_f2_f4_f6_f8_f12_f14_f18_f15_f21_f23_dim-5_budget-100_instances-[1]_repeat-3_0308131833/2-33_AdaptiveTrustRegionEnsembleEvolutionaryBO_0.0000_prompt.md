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




The selected solutions to update are:
## AdaptiveTrustRegionEvolutionaryBO
**Adaptive Trust Region Evolutionary Bayesian Optimization (ATREBO):** This algorithm synergistically combines the strengths of Trust Region Bayesian Optimization (TRBO) and Bayesian Evolutionary Optimization (BEO). It employs a Gaussian Process (GP) surrogate model, adaptively adjusts a trust region, and uses differential evolution (DE) within the trust region to optimize the acquisition function. The trust region size is dynamically adjusted based on the success of the DE search. Furthermore, ATREBO incorporates a mechanism to adapt the mutation rate of the DE based on the success rate of recent generations, enhancing its adaptability. To avoid the error encountered in BEO, the mutation parameter for DE is carefully controlled within valid bounds.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution

class AdaptiveTrustRegionEvolutionaryBO:
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
        self.de_popsize = 10 # Population size for differential evolution
        self.de_mutation = 0.5 # Initial mutation rate for differential evolution
        self.de_crossover = 0.7 # Crossover rate for differential evolution
        self.mutation_adaptation_rate = 0.1 # Rate to adapt mutation based on success
        self.success_threshold_de = 0.2 # Threshold for considering a generation successful

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
        # Expected Improvement acquisition function
        improvement = self.best_y - mu
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # Avoid division by zero
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
            maxiter=5, # Reduce maxiter for computational efficiency
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
            
            # Adapt mutation rate based on success
            if len(self.y) > self.de_popsize:
                recent_ys = self.y[-self.de_popsize:]
                best_recent_y = np.min(recent_ys)
                if best_recent_y < self.best_y + self.success_threshold_de:
                    self.de_mutation *= (1 + self.mutation_adaptation_rate)
                else:
                    self.de_mutation *= (1 - self.mutation_adaptation_rate)
                self.de_mutation = np.clip(self.de_mutation, 0.1, 1.99) # Clip mutation to avoid ValueError

        return self.best_y, self.best_x

```
The algorithm AdaptiveTrustRegionEvolutionaryBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1929 with standard deviation 0.0918.

took 431.99 seconds to run.

## AdaptiveTrustRegionEnsembleBO
**Adaptive Trust Region Ensemble Bayesian Optimization (ATREBO):** This algorithm combines the strengths of Trust Region Bayesian Optimization (TRBO) and Dynamic Ensemble Bayesian Optimization (DEBO). It uses an ensemble of Gaussian Process (GP) models with different kernels within a trust region framework. The trust region size is adaptively adjusted based on the agreement between the ensemble's predictions and the actual function evaluations. The acquisition function is a weighted combination of Expected Improvement (EI) from each GP model in the ensemble, and a novelty search term to promote exploration. The weights of the GP models are dynamically adjusted based on their predictive performance within the trust region. A local search is performed around the best solution found so far.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.spatial.distance import cdist

class AdaptiveTrustRegionEnsembleBO:
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
        self.n_models = 3 # Number of GP models in the ensemble
        self.gps = []
        self.weights = np.ones(self.n_models) / self.n_models # Initial weights
        self.weight_decay = 0.95
        self.novelty_weight = 0.1
        self.best_x = None
        self.best_y = np.inf
        self.trust_region_radius = 2.0 # Initial trust region radius
        self.trust_region_shrink = 0.5 # Shrink factor for trust region
        self.trust_region_expand = 1.5 # Expansion factor for trust region
        self.success_threshold = 0.75 # Threshold for trust region expansion
        self.failure_threshold = 0.25 # Threshold for trust region contraction
        self.trust_region_center = np.zeros(dim) # Initial trust region center
        self.local_search_radius = 0.1
        
        # Define different kernels for the ensemble
        kernels = [
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed"),
            ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds="fixed", nu=1.5),
            ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=0.5, length_scale_bounds="fixed") # More local
        ]
        for kernel in kernels:
            self.gps.append(GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6))

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
        for gp in self.gps:
            gp.fit(X, y)

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.X is None:
            return np.zeros((len(X), 1))

        mu = np.zeros((len(X), 1))
        sigma = np.zeros((len(X), 1))
        ei = np.zeros((len(X), 1))
        
        # Weighted average of GP predictions
        for i, gp in enumerate(self.gps):
            m, s = gp.predict(X, return_std=True)
            mu_i = m.reshape(-1, 1)
            sigma_i = s.reshape(-1, 1)

            # Expected Improvement acquisition function for each model
            improvement = self.best_y - mu_i
            Z = improvement / sigma_i
            ei_i = improvement * norm.cdf(Z) + sigma_i * norm.pdf(Z)
            ei_i[sigma_i <= 1e-6] = 0.0  # Avoid division by zero
            ei += self.weights[i] * ei_i
            
            mu += self.weights[i] * mu_i
            sigma += self.weights[i] * sigma_i

        acquisition = ei

        # Novelty search component
        if len(self.X) > 0:
            distances = cdist(X, self.X)
            min_distances = np.min(distances, axis=1)
            acquisition += self.novelty_weight * min_distances.reshape(-1, 1)
        
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
            
        # Update GP model weights based on recent performance within the trust region
        if len(self.X) > self.n_init:
            for i, gp in enumerate(self.gps):
                # Calculate the error on the existing points within the trust region
                distances = np.linalg.norm(self.X - self.trust_region_center, axis=1)
                within_region = distances <= self.trust_region_radius
                
                if np.sum(within_region) > 0:
                    X_region = self.X[within_region]
                    y_region = self.y[within_region]
                    y_pred, sigma = gp.predict(X_region, return_std=True)
                    error = np.mean((y_pred.reshape(-1, 1) - y_region)**2)
                else:
                    error = 0.0 # If no points in the region, default to 0
                
                # Update weights based on the inverse of the error
                self.weights[i] = np.exp(-error)
            
            # Normalize the weights
            self.weights /= np.sum(self.weights)
            
            # Decay the weights to encourage exploration
            self.weights *= self.weight_decay
            self.weights += (1 - self.weight_decay) / self.n_models
            self.weights /= np.sum(self.weights)

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

            # Fit GP models
            self._fit_model(self.X, self.y)

            # Select points by acquisition function
            batch_size = min(10, self.budget - self.n_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Calculate the ratio of improvement
            predicted_improvement = self.best_y - np.mean([gp.predict(self.best_x.reshape(1, -1))[0] for gp in self.gps])
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
The algorithm AdaptiveTrustRegionEnsembleBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1712 with standard deviation 0.1031.

took 3.69 seconds to run.

Combine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it. If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.




Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

