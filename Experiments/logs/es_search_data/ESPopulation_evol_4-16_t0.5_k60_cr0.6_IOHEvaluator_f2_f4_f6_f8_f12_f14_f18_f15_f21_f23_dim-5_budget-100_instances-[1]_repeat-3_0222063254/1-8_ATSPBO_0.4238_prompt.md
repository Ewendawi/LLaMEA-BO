You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- ATRBO: 0.0905, 338.70 seconds, Adaptive Trust Region Bayesian Optimization (ATRBO) uses a Gaussian Process (GP) surrogate model with a lower confidence bound (LCB) acquisition function. It dynamically adjusts a trust region around the current best point, focusing exploration within this region while still allowing for occasional exploration outside. The size of the trust region is adapted based on the GP's uncertainty and the optimization progress. Instead of batch evaluation, it uses sequential evaluation with trust region shrinking, which is especially effective when function evaluations are costly. The shrinking factor is also adjusted dynamically during the optimization process to balance exploration and exploitation.


- EHBO: 0.0479, 85.21 seconds, Efficient Hybrid Bayesian Optimization (EHBO) leverages a Gaussian Process (GP) surrogate model with Expected Improvement (EI) acquisition. For computational efficiency, it employs a batch-oriented approach using a combination of quasi-Monte Carlo (QMC) sampling (Sobol sequence) for exploration and gradient-based optimization (L-BFGS-B) of the acquisition function for exploitation. A crucial component is the dynamic adaptation of the batch size based on remaining budget and the dimension of the search space. This balances exploration and exploitation throughout the optimization process. Specifically, the batch size starts relatively large to promote initial exploration, and then it decreases as the remaining budget dwindles, to enable more focused exploitation near the end of the budget. This also allows for efficient parallelization, if available.


- DEHBBO: 0.0464, 45.95 seconds, DEHBBO: Diversity Enhanced Hybrid Bayesian Black Optimization. This algorithm combines aspects of both EHBO and ATRBO, while introducing a diversity-enhancing component to mitigate premature convergence. It uses a Gaussian Process with Expected Improvement acquisition, similar to EHBO, but incorporates ideas from ATRBO through adaptive sampling. The core innovation lies in a dynamic diversity maintenance strategy. After a few iterations, the algorithm begins to maintain a "Hall of Fame" of diverse solutions based on the euclidean distance in the decision space. The acquisition function is modified to penalize points close to existing members in the hall of fame, promoting exploration of less-visited regions. The size of the Hall of Fame is dynamically adjusted. The batch selection uses a combination of L-BFGS-B, like EHBO, and random sampling around the best location.


- SPBO: 0.0000, 0.00 seconds, Stochastic Patch Bayesian Optimization (SPBO) uses a combination of a Gaussian Process (GP) surrogate model and a Thompson Sampling acquisition function. The key innovation is the use of "stochastic patches" which are randomly sampled subsets of the input dimensions. The GP is trained on the full dataset, but the acquisition function is evaluated by randomly selecting a patch of dimensions and performing Thompson sampling on that patch. This introduces stochasticity and encourages exploration in different subspaces, especially useful in high-dimensional problems where it's computationally expensive to explore the entire space exhaustively. A crucial component is the dynamic adaptation of the patch size based on remaining budget and the dimension of the search space, ensuring that exploration and exploitation are balanced.




The selected solutions to update are:
## ATRBO
Adaptive Trust Region Bayesian Optimization (ATRBO) uses a Gaussian Process (GP) surrogate model with a lower confidence bound (LCB) acquisition function. It dynamically adjusts a trust region around the current best point, focusing exploration within this region while still allowing for occasional exploration outside. The size of the trust region is adapted based on the GP's uncertainty and the optimization progress. Instead of batch evaluation, it uses sequential evaluation with trust region shrinking, which is especially effective when function evaluations are costly. The shrinking factor is also adjusted dynamically during the optimization process to balance exploration and exploitation.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ATRBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(10 * dim, self.budget // 5)
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.5  # Initial trust region radius
        self.rho = 0.95  # Shrinking factor
        self.kappa = 2.0 # Exploration-exploitation trade-off for LCB

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, center=None, radius=None):
        # sample points within the trust region
        if center is None:
            center = (self.bounds[1] + self.bounds[0]) / 2
        if radius is None:
            radius = np.max(self.bounds[1] - self.bounds[0]) / 2

        sampler = qmc.Sobol(d=self.dim, scramble=True)
        points = sampler.random(n=n_points)
        points = qmc.scale(points, -1, 1)  # Scale to [-1, 1]

        # Project points to a hypersphere
        lengths = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / lengths * np.random.uniform(0, 1, size=lengths.shape) ** (1/self.dim)

        points = points * radius + center
        
        # Clip to the bounds
        points = np.clip(points, self.bounds[0], self.bounds[1])
        return points

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp):
        # Implement Lower Confidence Bound acquisition function
        mu, sigma = gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        return mu - self.kappa * sigma

    def _select_next_point(self, gp):
        # Select the next point to evaluate within the trust region
        n_samples = 100 * self.dim
        samples = self._sample_points(n_samples, center=self.best_x, radius=self.trust_region_radius)
        acq_values = self._acquisition_function(samples, gp)
        best_index = np.argmin(acq_values)
        return samples[best_index]

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y
    
    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))
        
        # Update best seen value
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Initialize best_x to a random initial point
        self.best_x = self.X[np.argmin(self.y)].copy()

        while self.n_evals < self.budget:
            # Optimization
            gp = self._fit_model(self.X, self.y)

            # Select next point
            next_x = self._select_next_point(gp)
            next_y = self._evaluate_points(func, next_x.reshape(1, -1))
            self._update_eval_points(next_x.reshape(1, -1), next_y)

            # Adjust trust region radius
            if next_y < self.best_y:
                self.trust_region_radius /= self.rho # Expand
                self.kappa *= self.rho
            else:
                self.trust_region_radius *= self.rho  # Shrink
                self.kappa /= self.rho # More exploration

            self.trust_region_radius = np.clip(self.trust_region_radius, 1e-2, np.max(self.bounds[1] - self.bounds[0]) / 2)
            self.kappa = np.clip(self.kappa, 0.1, 10.0)

        return self.best_y, self.best_x

```
The algorithm ATRBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.0905 with standard deviation 0.0914.

took 338.70 seconds to run.

## SPBO
Stochastic Patch Bayesian Optimization (SPBO) uses a combination of a Gaussian Process (GP) surrogate model and a Thompson Sampling acquisition function. The key innovation is the use of "stochastic patches" which are randomly sampled subsets of the input dimensions. The GP is trained on the full dataset, but the acquisition function is evaluated by randomly selecting a patch of dimensions and performing Thompson sampling on that patch. This introduces stochasticity and encourages exploration in different subspaces, especially useful in high-dimensional problems where it's computationally expensive to explore the entire space exhaustively. A crucial component is the dynamic adaptation of the patch size based on remaining budget and the dimension of the search space, ensuring that exploration and exploitation are balanced.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class SPBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(10 * dim, self.budget // 5)
        self.best_y = float('inf')
        self.best_x = None

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
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp, patch_indices):
        # Implement Thompson Sampling on a stochastic patch
        mu, sigma = gp.predict(X[:, patch_indices], return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        
        # Thompson Sampling: sample from the Gaussian posterior
        thompson_samples = np.random.normal(mu, sigma)
        return thompson_samples

    def _select_next_points(self, batch_size, gp):
        # Select the next points to evaluate
        next_X = []
        for _ in range(batch_size):
            # Dynamic patch size
            remaining_evals = self.budget - self.n_evals
            patch_size = max(1, min(self.dim, int(self.dim * remaining_evals / self.budget) + 1))  # Adapt patch size
            
            # Randomly select a patch of dimensions
            patch_indices = np.random.choice(self.dim, patch_size, replace=False)
            
            # Sample a candidate point within the full dimension space
            candidate_x = self._sample_points(1)
            
            # Evaluate the acquisition function (Thompson Sampling) on the patch
            acq_value = self._acquisition_function(candidate_x, gp, patch_indices)
            
            next_X.append(candidate_x.flatten())

        return np.array(next_X)

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
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]

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
            gp = self._fit_model(self.X, self.y)

            # Dynamic batch size
            remaining_evals = self.budget - self.n_evals
            batch_size = min(max(1, int(remaining_evals / (self.dim * 0.1))), 20) # Ensure at least 1 point and limit to 20

            # select points by acquisition function
            next_X = self._select_next_points(batch_size, gp)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x

```
An error occurred : Traceback (most recent call last):
  File "<SPBO>", line 115, in __call__
 115->             next_X = self._select_next_points(batch_size, gp)
  File "<SPBO>", line 65, in _select_next_points
  65->             acq_value = self._acquisition_function(candidate_x, gp, patch_indices)
  File "<SPBO>", line 42, in _acquisition_function
  40 |     def _acquisition_function(self, X, gp, patch_indices):
  41 |         # Implement Thompson Sampling on a stochastic patch
  42->         mu, sigma = gp.predict(X[:, patch_indices], return_std=True)
  43 |         mu = mu.reshape(-1, 1)
  44 |         sigma = sigma.reshape(-1, 1)
ValueError: X has 4 features, but GaussianProcessRegressor is expecting 5 features as input.


Combine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it. If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.




Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

