You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- TrustRegionBO: 0.1770, 1.42 seconds, **TrustRegionBO (TRBO):** This algorithm employs a Gaussian Process (GP) surrogate model within a trust region framework. It starts with an initial trust region size and iteratively refines the GP model within this region. The acquisition function balances exploration and exploitation, and the next point is selected by optimizing this function within the trust region. The trust region size is adaptively adjusted based on the agreement between the GP model's predictions and the actual function evaluations. If the agreement is good, the trust region is expanded; otherwise, it is contracted. This approach focuses the search on promising regions while ensuring sufficient exploration. To enhance computational efficiency, TRBO uses a simple local search within the trust region and periodically re-evaluates the trust region center.


- QuantileRegretBO: 0.1578, 1.30 seconds, **QuantileRegretBO (QRBO):** This algorithm focuses on minimizing the quantile of the observed regret. It uses a Gaussian Process (GP) to model the objective function and estimates the distribution of the regret. The acquisition function is based on the Conditional Value at Risk (CVaR) of the regret, which is optimized to select the next points. This approach is robust to noisy evaluations and outliers, as it focuses on the tail of the regret distribution. The algorithm also incorporates a dynamic adjustment of the quantile level based on the optimization progress, adapting the risk aversion during the search.


- EfficientHybridBO: 0.1527, 0.66 seconds, **Efficient Hybrid Bayesian Optimization (EHBBO):** This algorithm combines an efficient space-filling initial design with a Gaussian Process (GP) surrogate model and an acquisition function that balances exploration and exploitation. It uses a batch selection strategy based on Thompson sampling to efficiently select multiple points for evaluation in each iteration. To enhance computational efficiency, the GP model is updated only periodically, and a simple yet effective local search is performed around the best-observed solution to refine the search.


- DynamicEnsembleBO: 0.1494, 2.93 seconds, **DynamicEnsembleBO (DEBO):** This algorithm employs a dynamic ensemble of Gaussian Process (GP) models with varying kernel parameters to capture different characteristics of the objective function. It uses a weighted acquisition function that combines the predictions from each GP model in the ensemble, where the weights are dynamically adjusted based on the models' predictive performance. To enhance exploration, DEBO incorporates a novelty search component that encourages the algorithm to explore regions that are dissimilar to previously evaluated points. This approach aims to improve the robustness and adaptability of the optimization process.


- SurrogateModelFreeBO: 0.1485, 0.95 seconds, **SurrogateModelFreeBO (SMFBO):** This algorithm deviates from traditional Bayesian Optimization by eliminating the explicit surrogate model. Instead, it relies on a combination of space-filling design, direct function evaluations, and a nearest-neighbor-based acquisition function. It maintains a memory of evaluated points and their corresponding function values. The acquisition function selects new points by considering the distance to existing points and their function values, favoring regions that are both unexplored and have the potential for improvement. To enhance exploration, it incorporates a random sampling component. This approach is particularly suitable for problems where fitting a surrogate model is computationally expensive or unreliable.


- GradientEnhancedBO: 0.1407, 1.13 seconds, **GradientEnhancedBO (GEBO):** This algorithm leverages gradient information to enhance the efficiency of Bayesian Optimization. It uses a Gaussian Process (GP) surrogate model and incorporates gradient observations into the GP training data. The acquisition function combines the predicted mean and variance from the GP with a gradient-based exploration term, encouraging the algorithm to explore regions with promising gradients. To further improve performance, GEBO employs a multi-start local optimization strategy to refine the search around promising candidate points identified by the acquisition function.


- AdaptiveVarianceBO: 0.0000, 0.00 seconds, **AdaptiveVarianceBO (AVBO):** This algorithm utilizes a Gaussian Process (GP) surrogate model with an acquisition function that adaptively adjusts its exploration-exploitation trade-off based on the variance of the GP predictions. It employs a dynamic scaling factor for the exploration term in the acquisition function, increasing exploration when the GP's predictive variance is low across the search space and decreasing it when high variance regions are identified. This adaptive strategy aims to efficiently balance global exploration and local refinement. Additionally, to improve computational efficiency, the algorithm uses a sparse GP approximation by selecting a subset of data points for model fitting in each iteration, which reduces the computational cost associated with GP training and prediction.


- BayesianEvolutionaryBO: 0.0000, 0.00 seconds, **Bayesian Evolutionary Optimization (BEO):** This algorithm combines Bayesian Optimization with evolutionary strategies. It uses a Gaussian Process (GP) to model the objective function and an acquisition function based on Expected Improvement (EI). Instead of directly optimizing the acquisition function, it employs an evolutionary algorithm (specifically, a differential evolution strategy) to generate candidate points. This approach leverages the global search capabilities of evolutionary algorithms while still benefiting from the sample efficiency of Bayesian Optimization. The algorithm also incorporates a mechanism to adapt the mutation rate of the differential evolution based on the success rate of recent generations, further enhancing its adaptability.




The selected solution to update is:
**QuantileRegretBO (QRBO):** This algorithm focuses on minimizing the quantile of the observed regret. It uses a Gaussian Process (GP) to model the objective function and estimates the distribution of the regret. The acquisition function is based on the Conditional Value at Risk (CVaR) of the regret, which is optimized to select the next points. This approach is robust to noisy evaluations and outliers, as it focuses on the tail of the regret distribution. The algorithm also incorporates a dynamic adjustment of the quantile level based on the optimization progress, adapting the risk aversion during the search.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

class QuantileRegretBO:
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
        self.quantile_level = 0.9 # Initial quantile level for CVaR
        self.quantile_decay = 0.95 # Decay factor for quantile level
        self.min_quantile_level = 0.5 # Minimum quantile level
        self.best_y = np.inf
        self.best_x = None

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
        
        # Calculate CVaR of the regret
        regret = mu - self.best_y
        alpha = self.quantile_level
        
        # CVaR approximation (using Gaussian quantiles)
        VaR = regret + sigma * norm.ppf(alpha)
        CVaR = regret - (sigma * norm.pdf(norm.ppf(alpha)) / (1 - alpha))

        # If alpha is close to 1, the above calculation can be unstable.
        # In this case, we can approximate CVaR with VaR.
        if alpha > 0.99:
            CVaR = VaR
            
        return CVaR.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Optimize the acquisition function to find the next points
        x_tries = self._sample_points(batch_size * 10) # Generate more candidates
        acq_values = self._acquisition_function(x_tries)

        # Select the top batch_size points based on the acquisition function values
        indices = np.argsort(acq_values.flatten())[:batch_size] # changed from [::-1] to [:] since we want to minimize CVaR
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
            
            # Decay quantile level
            self.quantile_level *= self.quantile_decay
            self.quantile_level = max(self.quantile_level, self.min_quantile_level)

        return self.best_y, self.best_x

```
The algorithm QuantileRegretBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1578 with standard deviation 0.1029.

took 1.30 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

