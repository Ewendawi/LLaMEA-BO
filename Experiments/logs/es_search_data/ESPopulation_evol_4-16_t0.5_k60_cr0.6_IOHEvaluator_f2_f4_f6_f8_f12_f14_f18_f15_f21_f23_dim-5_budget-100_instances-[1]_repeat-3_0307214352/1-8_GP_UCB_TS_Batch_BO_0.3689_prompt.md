You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- DE_BO: 0.1641, 1002.44 seconds, DE_BO: Differential Evolution Bayesian Optimization. This algorithm uses Differential Evolution (DE) to optimize the acquisition function in Bayesian Optimization. It employs a Gaussian Process (GP) as the surrogate model and Expected Improvement (EI) as the acquisition function. DE is used to efficiently explore the search space and find promising candidate solutions for evaluation. The initial population for DE is sampled using a Latin Hypercube design.


- RBF_Bandwidth_BO: 0.1619, 322.21 seconds, RBF_Bandwidth_BO: Bayesian Optimization with dynamically adjusted RBF kernel bandwidth based on the data distribution. This algorithm employs a Gaussian Process (GP) as a surrogate model and Expected Improvement (EI) as the acquisition function. The key novelty lies in adaptively tuning the RBF kernel's bandwidth parameter during the optimization process using the median heuristic, which adjusts the bandwidth based on the median distance between data points. This aims to improve the GP's ability to model the underlying function by dynamically adapting to the data's characteristics. The acquisition function is optimized using a combination of random sampling and local search (L-BFGS-B).


- EI_ABS_BO: 0.1616, 745.98 seconds, Efficient Bayesian Optimization with Expected Improvement and Adaptive Batch Size (EI-ABS-BO). This algorithm uses a Gaussian Process (GP) as a surrogate model, Expected Improvement (EI) as the acquisition function, and an adaptive batch size strategy to balance exploration and exploitation. The batch size is adjusted based on the uncertainty of the GP predictions. QMC sampling is used for initial sampling and acquisition function optimization.


- GP_UCB_TS_BO: 0.1410, 238.66 seconds, GP_UCB_TS_BO: Gaussian Process Upper Confidence Bound with Thompson Sampling Bayesian Optimization. This algorithm combines the Upper Confidence Bound (UCB) acquisition function with Thompson Sampling (TS) to balance exploration and exploitation. It uses a Gaussian Process (GP) as a surrogate model. Instead of optimizing the acquisition function, Thompson Sampling draws samples from the posterior distribution of the GP, and UCB is used to select the next point to evaluate. This approach aims to improve exploration by considering the uncertainty in the GP predictions more directly.




The selected solution to update is:
GP_UCB_TS_BO: Gaussian Process Upper Confidence Bound with Thompson Sampling Bayesian Optimization. This algorithm combines the Upper Confidence Bound (UCB) acquisition function with Thompson Sampling (TS) to balance exploration and exploitation. It uses a Gaussian Process (GP) as a surrogate model. Instead of optimizing the acquisition function, Thompson Sampling draws samples from the posterior distribution of the GP, and UCB is used to select the next point to evaluate. This approach aims to improve exploration by considering the uncertainty in the GP predictions more directly.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class GP_UCB_TS_BO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 4 * dim

        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.kappa = 2.0  # UCB exploration-exploitation parameter

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        points = sampler.random(n=n_points)
        return qmc.scale(points, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X, kappa):
        # Implement UCB acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        return mu + kappa * sigma

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using Thompson Sampling and UCB
        # return array of shape (batch_size, n_dims)

        # Thompson Sampling: Draw a sample from the posterior
        sampled_f = self.gp.sample_y(self.X, n_samples=1)

        # UCB on a set of randomly sampled points
        num_candidates = 100 * self.dim
        X_candidate = self._sample_points(num_candidates)
        ucb_values = self._acquisition_function(X_candidate, self.kappa)

        # Select the point with the maximum UCB value
        next_point = X_candidate[np.argmax(ucb_values)]
        return next_point.reshape(1, -1)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)
    
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
        if np.min(new_y) < self.best_y:
            self.best_y = np.min(new_y)
            self.best_x = new_X[np.argmin(new_y)]
    
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
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Adjust kappa over time
            self.kappa = 2.0 - 1.8 * (self.n_evals / self.budget)

            # Select next points by acquisition function
            next_X = self._select_next_points(1)

            # Evaluate the points
            next_y = self._evaluate_points(func, next_X)
            
            # Update evaluated points
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x

```
The algorithm GP_UCB_TS_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1410 with standard deviation 0.0995.

took 238.66 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

