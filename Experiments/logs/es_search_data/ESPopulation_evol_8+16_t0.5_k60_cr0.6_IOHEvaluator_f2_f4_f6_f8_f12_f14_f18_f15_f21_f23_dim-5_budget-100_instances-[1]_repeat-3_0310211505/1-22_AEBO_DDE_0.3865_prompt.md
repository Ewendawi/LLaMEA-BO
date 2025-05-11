You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- ATRBO: 0.1891, 936.74 seconds, Adaptive Trust Region Bayesian Optimization (ATRBO) is a novel metaheuristic algorithm for black-box optimization. It employs a Gaussian Process Regression (GPR) model as a surrogate. The algorithm adaptively adjusts the trust region based on the GPR's uncertainty and the success of previous iterations. It uses Expected Improvement (EI) as the acquisition function within the trust region. To address the `NaN` error from the previous algorithm, a robust EI calculation is implemented, and the K-means clustering is replaced with a simpler top-k selection based on acquisition function values.


- SMABO: 0.1638, 324.62 seconds, SMABO: Surrogate Model Averaging Bayesian Optimization is a Bayesian optimization algorithm that uses an ensemble of surrogate models to improve robustness and exploration. It averages predictions from multiple Gaussian Process Regression (GPR) models, each with a different kernel. The acquisition function is Expected Improvement (EI). To enhance exploration, a dynamic temperature parameter is introduced, influencing the exploration-exploitation trade-off. The next points are selected by sampling a large number of candidates and choosing the top ones based on the acquisition function.


- DEBO: 0.1625, 213.86 seconds, DEBO: Diversity-Enhanced Bayesian Optimization is a Bayesian optimization algorithm that emphasizes diversity in the selection of candidate points. It uses a Gaussian Process Regression (GPR) model as a surrogate and employs a combination of Expected Improvement (EI) and a diversity-promoting term in the acquisition function. The diversity term is based on the minimum distance to existing points. The initial sampling is space-filling using Latin Hypercube Sampling. To avoid the `NaN` errors observed in EHBBO, a clipping mechanism is added to the EI calculation. Instead of using local optimization or clustering, DEBO uses a more direct approach by sampling a large number of candidates and selecting the batch that maximizes the acquisition function including the diversity term.


- LGBO: 0.1613, 159.93 seconds, LGBO: Landscape-Guided Bayesian Optimization is a Bayesian optimization algorithm that uses a combination of landscape analysis and Gaussian Process Regression (GPR) to guide the search. It starts with an initial sampling using Latin Hypercube Sampling (LHS). After the initial sampling, it performs a simple landscape analysis by evaluating the correlation between distances in the search space and differences in function values. Based on this correlation, it adaptively adjusts the exploration-exploitation trade-off. Specifically, if the correlation is high (indicating a smooth landscape), the algorithm focuses more on exploitation using Expected Improvement (EI). If the correlation is low (indicating a rugged landscape), it increases exploration by using a modified EI with a higher temperature. To prevent the `NaN` errors from `EHBBO` and `DAEBO`, a clipping mechanism is added to the EI calculation, and the probabilities in `np.random.choice` are checked for `NaN` values.


- GEBO: 0.1554, 380.88 seconds, Gradient-Enhanced Bayesian Optimization (GEBO) is a metaheuristic algorithm designed for black-box optimization that leverages gradient information to enhance the efficiency of the search. It employs a Gaussian Process Regression (GPR) model as a surrogate, incorporating gradient observations directly into the GPR training. The acquisition function is based on Expected Improvement (EI), modified to account for gradient information. A key feature is the use of gradient estimation via finite differences when true gradients are unavailable, allowing the algorithm to adapt to functions with or without readily accessible gradient data.


- EHBBO: 0.0000, 0.00 seconds, Efficient Hybrid Bayesian Optimization (EHBBO) is a metaheuristic algorithm designed for black-box optimization. It combines an efficient space-filling initial sampling strategy (Latin Hypercube Sampling), a Gaussian Process Regression (GPR) surrogate model with automatic kernel selection, and a hybrid acquisition function blending Expected Improvement (EI) and Upper Confidence Bound (UCB). A computationally efficient batch selection strategy based on k-means clustering is employed to select diverse points in the acquisition landscape.


- RaNDBO: 0.0000, 0.00 seconds, RaNDBO: Randomly-initialized Neural Dynamics Bayesian Optimization is a novel Bayesian optimization algorithm that leverages neural dynamics to guide the search process. It uses a Gaussian Process Regression (GPR) model as a surrogate and employs a neural network, initialized with random weights and biases, to generate candidate points. The acquisition function is Expected Improvement (EI). To enhance exploration, the algorithm introduces a mechanism for periodically re-initializing the neural network with new random weights, promoting diversity in the search. The neural network acts as a dynamic point generator, and its re-initialization ensures that the algorithm explores different regions of the search space throughout the optimization process.


- DAEBO: 0.0000, 0.00 seconds, DAEBO: Dueling Acquisition Ensemble Bayesian Optimization is a Bayesian optimization algorithm that combines an ensemble of acquisition functions with a dueling bandit approach for adaptive acquisition function selection. It uses a Gaussian Process Regression (GPR) model as a surrogate. The algorithm maintains a set of acquisition functions (e.g., EI, UCB, PI) and learns to select the best-performing one based on their observed performance during optimization. The dueling bandit mechanism compares pairs of acquisition functions and updates their selection probabilities based on which one leads to a better function evaluation. This adaptive selection of acquisition functions allows the algorithm to dynamically adjust its exploration-exploitation trade-off during the optimization process. To address the `ValueError: Sample is not in unit hypercube` error from `RaNDBO`, the neural network is removed. Instead, the candidate points are generated using Latin Hypercube Sampling.




The selected solutions to update are:
## SMABO
SMABO: Surrogate Model Averaging Bayesian Optimization is a Bayesian optimization algorithm that uses an ensemble of surrogate models to improve robustness and exploration. It averages predictions from multiple Gaussian Process Regression (GPR) models, each with a different kernel. The acquisition function is Expected Improvement (EI). To enhance exploration, a dynamic temperature parameter is introduced, influencing the exploration-exploitation trade-off. The next points are selected by sampling a large number of candidates and choosing the top ones based on the acquisition function.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel

class SMABO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(10*dim, self.budget//5)
        self.temperature = 1.0 # Initial temperature for exploration
        self.temperature_decay = 0.95 # Decay rate for the temperature


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
        
        # Define kernels
        kernel_rbf = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        kernel_matern = ConstantKernel(1.0) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        # Initialize models
        gp_rbf = GaussianProcessRegressor(kernel=kernel_rbf, n_restarts_optimizer=5)
        gp_matern = GaussianProcessRegressor(kernel=kernel_matern, n_restarts_optimizer=5)
        
        # Fit models
        gp_rbf.fit(X, y)
        gp_matern.fit(X, y)
        
        return gp_rbf, gp_matern

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1)) # Return zeros if no data is available

        mu_rbf, sigma_rbf = self.model_rbf.predict(X, return_std=True)
        mu_matern, sigma_matern = self.model_matern.predict(X, return_std=True)

        mu_rbf = mu_rbf.reshape(-1, 1)
        sigma_rbf = sigma_rbf.reshape(-1, 1)
        mu_matern = mu_matern.reshape(-1, 1)
        sigma_matern = sigma_matern.reshape(-1, 1)
        
        # Average the predictions
        mu = (mu_rbf + mu_matern) / 2.0
        sigma = (sigma_rbf + sigma_matern) / 2.0

        # Expected Improvement with temperature
        best = np.min(self.y)
        imp = best - mu
        Z = imp / (self.temperature * sigma + 1e-9)  # Adding a small constant to avoid division by zero
        ei = imp * norm.cdf(Z) + self.temperature * sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0 # avoid division by zero

        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Generate candidate points
        n_candidates = max(2000, batch_size * 200)
        X_cand = self._sample_points(n_candidates)

        # Calculate acquisition function values
        acq_values = self._acquisition_function(X_cand)

        # Select top-k points based on acquisition function values
        top_indices = np.argsort(acq_values.flatten())[::-1][:batch_size]
        return X_cand[top_indices]

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
            self.X = np.concatenate((self.X, new_X), axis=0)
            self.y = np.concatenate((self.y, new_y), axis=0)

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Optimization loop
        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            # Fit the model
            self.model_rbf, self.model_matern = self._fit_model(self.X, self.y)

            # Select next points
            X_next = self._select_next_points(batch_size)

            # Evaluate points
            y_next = self._evaluate_points(func, X_next)

            # Update evaluated points
            self._update_eval_points(X_next, y_next)

            # Update temperature
            self.temperature *= self.temperature_decay

        # Return best solution
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x

```
The algorithm SMABO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1638 with standard deviation 0.0992.

took 324.62 seconds to run.

## EHBBO
Efficient Hybrid Bayesian Optimization (EHBBO) is a metaheuristic algorithm designed for black-box optimization. It combines an efficient space-filling initial sampling strategy (Latin Hypercube Sampling), a Gaussian Process Regression (GPR) surrogate model with automatic kernel selection, and a hybrid acquisition function blending Expected Improvement (EI) and Upper Confidence Bound (UCB). A computationally efficient batch selection strategy based on k-means clustering is employed to select diverse points in the acquisition landscape.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from sklearn.cluster import KMeans

class EHBBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(10*dim, self.budget//5) # initial samples, at least 10*dim, at most 1/5 of budget
        self.acq_weight = 0.5 # initial weight for EI, UCB is 1 - acq_weight

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
        
        # Define kernels
        kernel_rbf = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        kernel_matern = ConstantKernel(1.0) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        # Initialize models
        gp_rbf = GaussianProcessRegressor(kernel=kernel_rbf, n_restarts_optimizer=5)
        gp_matern = GaussianProcessRegressor(kernel=kernel_matern, n_restarts_optimizer=5)
        
        # Fit models
        gp_rbf.fit(X, y)
        gp_matern.fit(X, y)
        
        # Select the best kernel based on log marginal likelihood
        if gp_rbf.log_marginal_likelihood(gp_rbf.kernel_.theta) > gp_matern.log_marginal_likelihood(gp_matern.kernel_.theta):
            return gp_rbf
        else:
            return gp_matern

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1)) # Return zeros if no data is available

        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement
        best = np.min(self.y)
        imp = best - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0 # avoid division by zero

        # Upper Confidence Bound
        ucb = mu + 2 * sigma

        # Hybrid acquisition function
        acq = self.acq_weight * ei + (1 - self.acq_weight) * ucb
        return acq

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Generate candidate points
        n_candidates = max(1000, batch_size * 100)
        X_cand = self._sample_points(n_candidates)
        
        # Calculate acquisition function values
        acq_values = self._acquisition_function(X_cand)
        
        # Select diverse points using k-means clustering
        kmeans = KMeans(n_clusters=min(batch_size, len(X_cand)), random_state=0, n_init='auto') # ensure batch_size <= len(X_cand)
        kmeans.fit(X_cand, sample_weight=acq_values.flatten())
        
        # Return cluster centers as selected points
        return kmeans.cluster_centers_

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
            self.X = np.concatenate((self.X, new_X), axis=0)
            self.y = np.concatenate((self.y, new_y), axis=0)
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Optimization loop
        batch_size = max(1, self.dim // 2) # dynamic batch size
        while self.n_evals < self.budget:
            # Fit the model
            self.model = self._fit_model(self.X, self.y)
            
            # Select next points
            X_next = self._select_next_points(batch_size)
            
            # Evaluate points
            y_next = self._evaluate_points(func, X_next)
            
            # Update evaluated points
            self._update_eval_points(X_next, y_next)

            # Dynamic adjustment of acquisition weight
            self.acq_weight = min(1.0, self.acq_weight + 0.05) # shift weight towards EI

        # Return best solution
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x

```
An error occurred : Traceback (most recent call last):
  File "<EHBBO>", line 138, in __call__
 138->             X_next = self._select_next_points(batch_size)
  File "<EHBBO>", line 96, in _select_next_points
  94 |         # Select diverse points using k-means clustering
  95 |         kmeans = KMeans(n_clusters=min(batch_size, len(X_cand)), random_state=0, n_init='auto') # ensure batch_size <= len(X_cand)
  96->         kmeans.fit(X_cand, sample_weight=acq_values.flatten())
  97 |         
  98 |         # Return cluster centers as selected points
  File "mtrand.pyx", line 954, in numpy.random.mtrand.RandomState.choice
ValueError: probabilities contain NaN


Combine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it. If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.




Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

