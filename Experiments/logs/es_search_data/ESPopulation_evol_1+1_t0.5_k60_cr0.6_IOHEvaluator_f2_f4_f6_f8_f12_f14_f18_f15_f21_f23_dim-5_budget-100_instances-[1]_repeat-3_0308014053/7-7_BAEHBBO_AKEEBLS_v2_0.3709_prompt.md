You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- BAEHBBO_AKEEBLS: 0.1466, 70.17 seconds, **Budget-Aware Efficient Hybrid Bayesian Optimization with Adaptive Kernel, Exploration-Exploitation Balance, and Local Search Refinement (BAEHBBO-AKEEBLS):** This algorithm enhances the BAEHBBO-AKEEB framework by refining the local search strategy and incorporating a more sophisticated exploration-exploitation balance. The local search now uses L-BFGS-B for potentially faster convergence, and the exploration weight is adjusted based on the uncertainty of the Gaussian Process model. Additionally, a separate acquisition function is used for the local search step to encourage exploitation around the current best.




The selected solution to update is:
**Budget-Aware Efficient Hybrid Bayesian Optimization with Adaptive Kernel, Exploration-Exploitation Balance, and Local Search Refinement (BAEHBBO-AKEEBLS):** This algorithm enhances the BAEHBBO-AKEEB framework by refining the local search strategy and incorporating a more sophisticated exploration-exploitation balance. The local search now uses L-BFGS-B for potentially faster convergence, and the exploration weight is adjusted based on the uncertainty of the Gaussian Process model. Additionally, a separate acquisition function is used for the local search step to encourage exploitation around the current best.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize

class BAEHBBO_AKEEBLS:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.exploration_weight = 0.2  # Initial exploration weight

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        samples = sampler.random(n=n_points)
        return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Adaptive kernel length scale estimation
        if len(X) > self.n_init:
            distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
            distances = np.triu(distances, k=1)
            median_distance = np.median(distances[distances > 0])
            length_scale = median_distance
        else:
            length_scale = 1.0  # Initial length scale

        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=length_scale, length_scale_bounds="fixed")
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp, y_best):
        mu, sigma = gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)
        imp = mu - y_best
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        # Exploration component (using UCB)
        ucb = mu + self.exploration_weight * sigma

        # Combine EI and UCB
        acquisition = ei + self.exploration_weight * ucb

        return acquisition

    def _acquisition_function_local_search(self, X, gp, y_best):
        mu, sigma = gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)
        imp = mu - y_best
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei

    def _select_next_points(self, batch_size, gp, y_best):
        best_x = None
        best_acq = -np.inf
        for _ in range(10 * batch_size):
            x = self._sample_points(1)
            acq = self._acquisition_function(x, gp, y_best)[0]
            if acq > best_acq:
                best_acq = acq
                best_x = x

        return best_x

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)
    
    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        while self.n_evals < self.budget:
            # Fit the GP model
            gp = self._fit_model(self.X, self.y)
            
            # Determine the batch size adaptively
            remaining_evals = self.budget - self.n_evals
            batch_size = min(max(1, remaining_evals // 5), 5)  # Adaptive batch size
            
            # Select the next points using EI
            next_X = self._select_next_points(batch_size, gp, best_y)
            
            # Evaluate the next points
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Update the best solution
            best_idx = np.argmin(self.y)
            best_y = self.y[best_idx][0]
            best_x = self.X[best_idx]

            # Local search refinement using the surrogate model
            def surrogate_objective(x):
                return gp.predict(x.reshape(1, -1))[0]

            # Limit the number of iterations based on remaining budget
            max_iter = min(5, remaining_evals)  # Limit iterations
            if max_iter > 0:
                # Use L-BFGS-B for local search
                res = minimize(surrogate_objective, best_x, method='L-BFGS-B', bounds=list(zip(self.bounds[0], self.bounds[1])), options={'maxiter': max_iter})  # Limit function evaluations
                
                # Evaluate the result of the local search with the real function
                if self.n_evals < self.budget:
                    refined_y = self._evaluate_points(func, res.x.reshape(1, -1))[0][0]
                    if refined_y < best_y:
                        best_y = refined_y
                        best_x = res.x
            
            # Update exploration weight (adaptive decay based on GP uncertainty)
            mu, sigma = gp.predict(self.X, return_std=True)
            avg_sigma = np.mean(sigma)
            self.exploration_weight = max(0.01, min(0.5, avg_sigma))

        return best_y, best_x

```
The algorithm BAEHBBO_AKEEBLS got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1466 with standard deviation 0.0938.

took 70.17 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

