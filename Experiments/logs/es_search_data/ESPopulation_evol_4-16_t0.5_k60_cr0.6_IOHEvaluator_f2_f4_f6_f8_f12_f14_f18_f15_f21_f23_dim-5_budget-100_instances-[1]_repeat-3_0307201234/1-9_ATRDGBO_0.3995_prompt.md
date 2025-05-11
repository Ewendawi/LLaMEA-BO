You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- ATRBO: 0.1823, 194.42 seconds, **Adaptive Trust Region Bayesian Optimization (ATRBO):** This algorithm employs a Gaussian Process (GP) surrogate model with a Matérn kernel for enhanced flexibility in modeling functions with varying degrees of smoothness. It uses a trust-region approach, where the size of the trust region is adaptively adjusted based on the agreement between the GP model's predictions and the actual function evaluations. The acquisition function is based on the Lower Confidence Bound (LCB), which balances exploration and exploitation. To further improve exploration, especially in early stages, the algorithm incorporates a dynamic exploration factor in the LCB. The initial design uses a Sobol sequence for better space-filling properties compared to LHS.


- EHBBO: 0.1606, 11.45 seconds, **Efficient Hybrid Bayesian Optimization (EHBBO):** This algorithm combines an efficient space-filling initial design using Latin Hypercube Sampling (LHS), a Gaussian Process (GP) surrogate model with a Radial Basis Function (RBF) kernel for flexible modeling, and an acquisition function that balances exploration and exploitation using Expected Improvement (EI). To enhance exploration, especially in high-dimensional spaces, the algorithm incorporates a local search strategy around the best-observed point. The local search uses a simple Gaussian mutation operator to generate candidate solutions. The number of local search steps is dynamically adjusted based on the remaining budget to ensure effective exploration throughout the optimization process.


- GBO: 0.1528, 72.74 seconds, **GBO: Gradient-Boosted Bayesian Optimization:** This algorithm leverages gradient boosting to enhance the surrogate model's accuracy and adaptability. Instead of relying solely on a Gaussian Process, it uses a gradient-boosted tree model (specifically, HistGradientBoostingRegressor from scikit-learn) as the primary surrogate. This allows for capturing more complex relationships in the data and handling potential discontinuities or non-smoothness in the objective function. The acquisition function is based on the Upper Confidence Bound (UCB), and the exploration-exploitation trade-off is controlled by an adaptive exploration factor. Initial sampling is performed using a Latin Hypercube design. To mitigate the NaN error encountered in DIBO, a check for NaN values is added before fitting the model, and a simple imputation strategy (replacing NaNs with the mean) is employed if necessary.


- DIBO: 0.0000, 0.00 seconds, **DIBO: Diversity-Injected Bayesian Optimization:** This algorithm focuses on enhancing diversity in the search process to avoid premature convergence, which can be a problem in Bayesian Optimization, especially for complex functions. It uses a Gaussian Process (GP) with a dynamic kernel that adapts its length scale based on the observed data distribution. The acquisition function combines Expected Improvement (EI) with a diversity term that encourages exploration in less-visited regions of the search space. A clustering-based approach is used to identify these regions. Initial sampling uses a Halton sequence for good space-filling properties.




The selected solutions to update are:
## ATRBO
**Adaptive Trust Region Bayesian Optimization (ATRBO):** This algorithm employs a Gaussian Process (GP) surrogate model with a Matérn kernel for enhanced flexibility in modeling functions with varying degrees of smoothness. It uses a trust-region approach, where the size of the trust region is adaptively adjusted based on the agreement between the GP model's predictions and the actual function evaluations. The acquisition function is based on the Lower Confidence Bound (LCB), which balances exploration and exploitation. To further improve exploration, especially in early stages, the algorithm incorporates a dynamic exploration factor in the LCB. The initial design uses a Sobol sequence for better space-filling properties compared to LHS.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.optimize import minimize

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
        self.n_init = 2 * self.dim # Initial samples
        self.trust_region_size = 2.0  # Initial trust region size
        self.exploration_factor = 2.0 # Initial exploration factor

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.Sobol(d=self.dim, scramble=False)
        samples = sampler.random(n=n_points)
        return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, nu=2.5)
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Lower Confidence Bound
        lcb = mu - self.exploration_factor * sigma
        return lcb

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Optimize the acquisition function within the trust region using L-BFGS-B
        best_idx = np.argmin(self.y)
        best_x = self.X[best_idx]
        
        x_starts = best_x + np.random.normal(0, 0.1, size=(batch_size, self.dim))
        x_starts = np.clip(x_starts, self.bounds[0], self.bounds[1])

        candidates = []
        values = []
        for x_start in x_starts:
            # Define trust region bounds
            lower_bound = np.maximum(x_start - self.trust_region_size / 2, self.bounds[0])
            upper_bound = np.minimum(x_start + self.trust_region_size / 2, self.bounds[1])
            
            res = minimize(lambda x: self._acquisition_function(x.reshape(1, -1)),
                           x_start,
                           bounds=np.array([lower_bound, upper_bound]).T,
                           method="L-BFGS-B")
            candidates.append(res.x)
            values.append(res.fun)
        
        return np.array(candidates)

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
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        self.model = self._fit_model(self.X, self.y)
        
        while self.n_evals < self.budget:
            # Optimization
            batch_size = 1
            X_next = self._select_next_points(batch_size)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

            # Adaptive trust region adjustment
            y_pred, sigma = self.model.predict(X_next, return_std=True)
            y_pred = y_pred.reshape(-1, 1)
            
            # Agreement between prediction and actual value
            agreement = np.abs(y_pred - y_next) / sigma.reshape(-1, 1)
            
            if np.mean(agreement) < 1.0:
                self.trust_region_size *= 1.1  # Increase trust region if model is accurate
            else:
                self.trust_region_size *= 0.9  # Decrease trust region if model is inaccurate
            
            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0) # Clip trust region size

            # Dynamic exploration factor adjustment
            self.exploration_factor = 1.0 + (self.budget - self.n_evals) / self.budget # Reduce exploration over time
            
            self.model = self._fit_model(self.X, self.y) # Refit the model with new data
            
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        return best_y, best_x

```
The algorithm ATRBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1823 with standard deviation 0.1030.

took 194.42 seconds to run.

## DIBO
**DIBO: Diversity-Injected Bayesian Optimization:** This algorithm focuses on enhancing diversity in the search process to avoid premature convergence, which can be a problem in Bayesian Optimization, especially for complex functions. It uses a Gaussian Process (GP) with a dynamic kernel that adapts its length scale based on the observed data distribution. The acquisition function combines Expected Improvement (EI) with a diversity term that encourages exploration in less-visited regions of the search space. A clustering-based approach is used to identify these regions. Initial sampling uses a Halton sequence for good space-filling properties.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

class DIBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * self.dim # Initial samples
        self.diversity_weight = 0.1 # Weight for the diversity term in the acquisition function
        self.kernel_length_scale = 1.0 # Initial kernel length scale

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.Halton(d=self.dim, scramble=False)
        samples = sampler.random(n=n_points)
        return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        # Dynamically adjust kernel length scale
        self.kernel_length_scale = np.median(pairwise_distances(X))
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.kernel_length_scale, length_scale_bounds="fixed")
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement
        best = np.min(self.y)
        imp = mu - best
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # avoid division by zero

        # Diversity term: encourage exploration in less-visited regions
        if self.X is not None and len(self.X) > 5:
            kmeans = KMeans(n_clusters=min(5, len(self.X)), random_state=0, n_init = 'auto').fit(self.X)
            clusters = kmeans.predict(X)
            distances = np.array([np.min(pairwise_distances(x.reshape(1, -1), self.X[kmeans.labels_ == cluster])) for x, cluster in zip(X, clusters)])
            diversity = distances.reshape(-1, 1)
        else:
            diversity = np.zeros_like(ei)

        return -ei + self.diversity_weight * diversity # we want to maximize EI and diversity

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Optimize the acquisition function using L-BFGS-B
        x_starts = self._sample_points(batch_size)
        candidates = []
        values = []
        for x_start in x_starts:
            res = minimize(lambda x: self._acquisition_function(x.reshape(1, -1)),
                           x_start,
                           bounds=[(-5, 5)] * self.dim,
                           method="L-BFGS-B")
            candidates.append(res.x)
            values.append(res.fun)
        
        return np.array(candidates)

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
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        self.model = self._fit_model(self.X, self.y)
        
        while self.n_evals < self.budget:
            # Optimization
            batch_size = 1
            X_next = self._select_next_points(batch_size)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

            self.model = self._fit_model(self.X, self.y) # Refit the model with new data
            
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        return best_y, best_x

```
An error occurred : Traceback (most recent call last):
  File "<DIBO>", line 126, in __call__
 126->             X_next = self._select_next_points(batch_size)
  File "<DIBO>", line 83, in _select_next_points
  83->             res = minimize(lambda x: self._acquisition_function(x.reshape(1, -1)),
  File "<DIBO>", line 83, in <lambda>
  83->             res = minimize(lambda x: self._acquisition_function(x.reshape(1, -1)),
  File "<DIBO>", line 49, in _acquisition_function
  47 |         # calculate the acquisition function value for each point in X
  48 |         # return array of shape (n_points, 1)
  49->         mu, sigma = self.model.predict(X, return_std=True)
  50 |         mu = mu.reshape(-1, 1)
  51 |         sigma = sigma.reshape(-1, 1)
ValueError: Input X contains NaN.
GaussianProcessRegressor does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values


Combine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it. If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.




Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

