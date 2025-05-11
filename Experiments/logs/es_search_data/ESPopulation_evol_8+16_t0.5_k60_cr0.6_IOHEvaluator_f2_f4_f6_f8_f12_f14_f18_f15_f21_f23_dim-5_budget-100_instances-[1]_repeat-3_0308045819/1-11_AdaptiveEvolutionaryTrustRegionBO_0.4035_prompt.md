You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- AdaptiveTrustRegionBO: 0.1722, 6.36 seconds, **AdaptiveTrustRegionBO (ATBO):** This algorithm implements a trust region approach within a Bayesian Optimization framework. It uses a Gaussian Process (GP) to model the objective function and dynamically adjusts the size of the trust region based on the GP's predictive performance. The acquisition function is the lower confidence bound (LCB), which balances exploration and exploitation. To further enhance exploration, especially in high-dimensional spaces, the algorithm incorporates a random restart mechanism. The trust region is centered around the best point found so far.


- SurrogateEnsembleBO: 0.1529, 17.21 seconds, **SurrogateEnsembleBO (SEBO):** This algorithm employs an ensemble of surrogate models to improve robustness and handle uncertainty in the black-box function. The ensemble consists of Gaussian Process Regression (GPR) with different kernel parameters. The acquisition function is the Upper Confidence Bound (UCB), and the next points are selected by averaging the predictions of all surrogate models in the ensemble. To encourage exploration, a dynamic weighting scheme is used to adjust the contribution of each surrogate based on its performance.


- EHBBO: 0.1497, 4.99 seconds, **Efficient Hybrid Bayesian Optimization (EHBBO):** This algorithm combines an initial space-filling design using Latin Hypercube Sampling (LHS) with a Gaussian Process (GP) surrogate model and an acquisition function that balances exploration and exploitation. A computationally efficient batch selection strategy based on k-means clustering is used to diversify the search and improve parallelization. The acquisition function is Thompson Sampling, which is known for its efficiency and good performance in high-dimensional spaces.


- ParetoActiveBO: 0.1492, 10.52 seconds, **ParetoActiveBO (PABO):** This algorithm uses a Pareto-based approach to manage multiple acquisition functions simultaneously, promoting a diverse set of candidate solutions. It employs a Gaussian Process (GP) surrogate model and considers two acquisition functions: Expected Improvement (EI) for exploitation and a distance-based metric to encourage exploration of less-sampled regions. The Pareto front of non-dominated solutions is maintained, and new points are selected from this front. Active learning is incorporated by querying the function value at the point that maximizes the variance predicted by the GP, thus reducing uncertainty.


- BayesianEvolutionaryBO: 0.1415, 1149.15 seconds, **Bayesian Evolutionary Optimization (BEO):** This algorithm combines Bayesian Optimization with an evolutionary strategy. It uses a Gaussian Process (GP) surrogate model to estimate the objective function and an evolutionary algorithm (specifically, a differential evolution strategy) to explore the search space and select promising candidate points. The acquisition function is a hybrid of Expected Improvement (EI) and a diversity metric based on the distance to existing points. This helps to balance exploration and exploitation. The error in `GradientEnhancedBO` was due to exceeding the budget during local search. To avoid this, the number of local search steps is limited based on the remaining budget.


- DynamicResourceAllocationBO: 0.1341, 184.69 seconds, **DynamicResourceAllocationBO (DRABO):** This algorithm dynamically allocates the budget between exploration and exploitation phases based on the observed performance. It starts with an exploration phase using Thompson Sampling to quickly identify promising regions. Then, it switches to an exploitation phase using Expected Improvement (EI), focusing on refining the search within the identified regions. The switch between phases is governed by a performance metric based on the rate of improvement. To prevent `IndexError` observed in `ContextAwareBO`, the code now ensures that the selected indices are within the bounds of the candidate set. To address the `OverBudgetException` in `GradientEnhancedBO`, the number of L-BFGS-B iterations is limited based on the remaining budget.


- GradientEnhancedBO: 0.0000, 0.00 seconds, **GradientEnhancedBO (GEBO):** This algorithm leverages gradient information, estimated using finite differences, to enhance the Gaussian Process surrogate model and guide the search. It uses Expected Improvement (EI) as the acquisition function, modified to incorporate gradient information. A simple local search is performed around the best point found so far, using gradient information to accelerate convergence.


- ContextAwareBO: 0.0000, 0.00 seconds, **ContextAwareBO (CABO):** This algorithm introduces a context-aware approach to Bayesian Optimization by incorporating information about the local landscape around previously evaluated points. It uses a Gaussian Process (GP) surrogate model with a modified acquisition function that combines Expected Improvement (EI) with a context-aware term. This term penalizes points that are too close to existing points in regions where the GP model has high confidence (low uncertainty), encouraging exploration of less-explored areas. To estimate the local landscape, the algorithm uses a simple k-nearest neighbors (k-NN) approach. The initial sampling is done using a Sobol sequence to ensure good space coverage.




The selected solutions to update are:
## AdaptiveTrustRegionBO
**AdaptiveTrustRegionBO (ATBO):** This algorithm implements a trust region approach within a Bayesian Optimization framework. It uses a Gaussian Process (GP) to model the objective function and dynamically adjusts the size of the trust region based on the GP's predictive performance. The acquisition function is the lower confidence bound (LCB), which balances exploration and exploitation. To further enhance exploration, especially in high-dimensional spaces, the algorithm incorporates a random restart mechanism. The trust region is centered around the best point found so far.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

class AdaptiveTrustRegionBO:
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
        self.random_restart_prob = 0.05 #Probability of random restart

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
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
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
            #sigma = np.clip(sigma, 1e-9, np.inf)
            #LCB = mu - kappa * sigma #kappa is exploration-exploitation trade-off
            LCB = mu - 2.0 * sigma  # Using a fixed kappa for simplicity
            return LCB.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        candidates = self._sample_points(batch_size)
        acquisition_values = self._acquisition_function(candidates)
        best_index = np.argmin(acquisition_values)
        return candidates[best_index].reshape(1, -1)

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
        if self.success_ratio > 0.5:
            self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)

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
        batch_size = min(1, self.dim)
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

        return self.best_y, self.best_x

```
The algorithm AdaptiveTrustRegionBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1722 with standard deviation 0.1007.

took 6.36 seconds to run.

## BayesianEvolutionaryBO
**Bayesian Evolutionary Optimization (BEO):** This algorithm combines Bayesian Optimization with an evolutionary strategy. It uses a Gaussian Process (GP) surrogate model to estimate the objective function and an evolutionary algorithm (specifically, a differential evolution strategy) to explore the search space and select promising candidate points. The acquisition function is a hybrid of Expected Improvement (EI) and a diversity metric based on the distance to existing points. This helps to balance exploration and exploitation. The error in `GradientEnhancedBO` was due to exceeding the budget during local search. To avoid this, the number of local search steps is limited based on the remaining budget.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution

class BayesianEvolutionaryBO:
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
        self.de_pop_size = 15 # Population size for differential evolution
        self.gp_update_interval = 5 # Update GP model every n iterations

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
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function: Expected Improvement + Diversity
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            mu, sigma = self.gp.predict(X, return_std=True)
            sigma = np.clip(sigma, 1e-9, np.inf)
            imp = self.best_y - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

            # Diversity term: encourage exploration
            if self.X is not None:
                distances = np.min([np.linalg.norm(x - self.X, axis=1) for x in X], axis=0)
                diversity = distances.reshape(-1,1)
            else:
                diversity = np.ones((len(X), 1)) # No diversity if no points yet

            # Combine EI and diversity
            acquisition = ei + 0.1 * diversity # Adjust weight for diversity as needed
            return acquisition.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using differential evolution
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Define the objective function for differential evolution (negative acquisition function)
        def de_objective(x):
            return -self._acquisition_function(x.reshape(1, -1))[0, 0]

        # Perform differential evolution
        de_bounds = [(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)]
        result = differential_evolution(de_objective, de_bounds, popsize=self.de_pop_size, maxiter=max(1, self.budget//(batch_size*5)), tol=0.01, disp=False) # Adjust maxiter

        # Select the best point from differential evolution
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
        batch_size = min(1, self.dim)
        iteration = 0
        while self.n_evals < self.budget:
            # Fit the GP model periodically
            if iteration % self.gp_update_interval == 0:
                self._fit_model(self.X, self.y)

            # Select points by acquisition function using differential evolution
            next_X = self._select_next_points(batch_size)

            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)

            # Update the evaluated points
            self._update_eval_points(next_X, next_y)

            iteration += 1

        return self.best_y, self.best_x

```
The algorithm BayesianEvolutionaryBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1415 with standard deviation 0.1019.

took 1149.15 seconds to run.

Combine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it. If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.




Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

