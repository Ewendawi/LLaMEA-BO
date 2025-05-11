You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- AGATBO: 0.1767, 510.80 seconds, **AGATBO: Adaptive Gradient-Aware Trust Region Bayesian Optimization:** This algorithm combines the strengths of AGABO and ATRBO, incorporating gradient information into the acquisition function within an adaptive trust region framework. It uses a Gaussian Process Regression (GPR) model with a Matérn kernel for surrogate modeling. The acquisition function is a combination of Expected Improvement (EI) and a gradient-based exploration term, evaluated within a trust region. The size of the trust region is adaptively adjusted based on the agreement between the GPR model and the true objective function. Halton sequence is used for initial exploration, and Sobol sequence is used for sampling within the trust region.


- APTRBO: 0.1724, 104.60 seconds, **Adaptive Pareto Trust Region Bayesian Optimization (APTRBO):** This algorithm combines the strengths of Adaptive Trust Region Bayesian Optimization (ATRBO) and Pareto-based Acquisition Steering Bayesian Optimization (PaSBO) to achieve a more robust and efficient optimization process. It adaptively adjusts the trust region based on model agreement, uses a Pareto-based approach to manage multiple acquisition functions (EI, PI, UCB) within the trust region, and incorporates a dynamic exploration-exploitation balance by adjusting the UCB kappa parameter based on the trust region size. A Matern kernel is used for the Gaussian Process Regression (GPR) model.


- DEBO: 0.1705, 63.62 seconds, **DEBO: Diversity Enhanced Bayesian Optimization:** This algorithm focuses on enhancing diversity in Bayesian Optimization using a combination of techniques. It employs a Gaussian Process Regression (GPR) model with a Radial Basis Function (RBF) kernel for surrogate modeling. The acquisition function is a combination of Expected Improvement (EI) and a distance-based diversity term. This diversity term encourages the selection of points that are far away from previously evaluated points. A clustering-based approach is used to identify diverse regions in the search space. The initial exploration is performed using a Sobol sequence to ensure good coverage of the search space.


- DEDABO: 0.1679, 92.85 seconds, **DEDABO: Diversity Enhanced Distribution-Aware Bayesian Optimization:** This algorithm combines the strengths of DEBO (Diversity Enhanced Bayesian Optimization) and DABO (Distribution-Aware Bayesian Optimization). It uses a Gaussian Process Regression (GPR) model with an RBF kernel for surrogate modeling. The acquisition function is a hybrid of Expected Improvement (EI), a distance-based diversity term (from DEBO), and a distribution matching term using Kernel Density Estimation (KDE) (from DABO). The diversity term encourages exploration of regions far from previously evaluated points, while the distribution matching term focuses on sampling from regions identified as promising by the KDE. The algorithm uses Sobol sampling for initial exploration and KDE-guided sampling for subsequent iterations, enhanced by a diversity-promoting mechanism. The KDE is updated dynamically based on the top-performing samples. The error in DABO was due to a dimension mismatch when using boolean indexing. This is addressed by ensuring that the boolean index and the indexed array have compatible dimensions.


- EHBBORLS: 0.1646, 122.69 seconds, **EHBBO-AERLS: Efficient Hybrid Bayesian Optimization with Adaptive Exploration-Refinement and Local Search.** This algorithm builds upon EHBBO by introducing an adaptive strategy to balance exploration and refinement during point selection. It dynamically adjusts the number of candidates used for Thompson Sampling based on the uncertainty of the Gaussian Process model. Additionally, it integrates a more robust local search method (SLSQP) and adaptively scales the bounds of the local search based on the iteration number to control the search space.


- ARBO: 0.1644, 152.02 seconds, **Adaptive Regularized Bayesian Optimization with Dynamic Batch Size (ARBO):** This algorithm refines the Regularized Bayesian Optimization (ReBO) by introducing an adaptive batch size strategy and refining the regularization term. The batch size dynamically adjusts based on the iteration number, starting with a larger batch size for exploration and decreasing it as the algorithm progresses to focus on exploitation. The regularization term is modified to incorporate the uncertainty (sigma) predicted by the Gaussian Process, further encouraging exploration in uncertain regions. An additional exploration factor is added to the acquisition function to ensure sufficient exploration.


- ATRBO: 0.1639, 99.40 seconds, **Adaptive Trust Region Bayesian Optimization (ATRBO):** This algorithm uses a Gaussian Process Regression (GPR) model with a Matérn kernel for surrogate modeling. It employs a trust region approach, adaptively adjusting the size of the trust region based on the agreement between the GPR model and the true objective function. The acquisition function is the Expected Improvement (EI) within the trust region. The initial exploration is performed using Latin Hypercube Sampling (LHS). To enhance diversity, a Sobol sequence is used for sampling within the trust region.


- TraRBO: 0.1634, 97.87 seconds, **TraRBO: Trust Region and Regularized Bayesian Optimization:** This algorithm combines the strengths of Adaptive Trust Region Bayesian Optimization (ATRBO) and Regularized Bayesian Optimization (ReBO). It uses a Gaussian Process Regression (GPR) model with a Matérn kernel for surrogate modeling, similar to ATRBO, but incorporates the adaptive regularization term from ReBO into the Expected Improvement (EI) acquisition function. The trust region mechanism adaptively adjusts the search space based on model agreement, while the regularization term prevents overfitting and encourages exploration. This combination aims to balance exploration and exploitation more effectively, leading to improved performance. An adaptive mechanism for the trust region radius and regularization weight is also included.




The selected solution to update is:
**DEDABO: Diversity Enhanced Distribution-Aware Bayesian Optimization:** This algorithm combines the strengths of DEBO (Diversity Enhanced Bayesian Optimization) and DABO (Distribution-Aware Bayesian Optimization). It uses a Gaussian Process Regression (GPR) model with an RBF kernel for surrogate modeling. The acquisition function is a hybrid of Expected Improvement (EI), a distance-based diversity term (from DEBO), and a distribution matching term using Kernel Density Estimation (KDE) (from DABO). The diversity term encourages exploration of regions far from previously evaluated points, while the distribution matching term focuses on sampling from regions identified as promising by the KDE. The algorithm uses Sobol sampling for initial exploration and KDE-guided sampling for subsequent iterations, enhanced by a diversity-promoting mechanism. The KDE is updated dynamically based on the top-performing samples. The error in DABO was due to a dimension mismatch when using boolean indexing. This is addressed by ensuring that the boolean index and the indexed array have compatible dimensions.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KernelDensity
from sklearn.metrics import pairwise_distances

class DEDABO:
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
        self.kde = None
        self.best_x = None
        self.best_y = float('inf')
        self.distribution_weight = 0.1
        self.diversity_weight = 0.1

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, method='sobol'):
        # sample points
        # return array of shape (n_points, n_dims)
        if method == 'sobol':
            sampler = qmc.Sobol(d=self.dim, seed=42)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        elif method == 'kde':
            if self.kde is None:
                return self._sample_points(n_points, method='sobol')
            else:
                # Sample from KDE
                samples = self.kde.sample(n_points)
                samples = np.clip(samples, self.bounds[0], self.bounds[1])
                return samples
        else:
            raise ValueError("Invalid sampling method.")

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X, model):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if self.best_y is None:
            ei = np.zeros_like(mu)
        else:
            imp = self.best_y - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0  # handle zero sigma

        # Add distribution matching term
        if self.kde is not None:
            log_likelihood = self.kde.score_samples(X).reshape(-1, 1)
            ei = ei + self.distribution_weight * np.exp(log_likelihood) # Use exp to avoid negative values

        # Add diversity term
        if self.X is not None:
            distances = pairwise_distances(X, self.X)
            min_distances = np.min(distances, axis=1).reshape(-1, 1)
            ei = ei + self.diversity_weight * min_distances

        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # return array of shape (batch_size, n_dims)
        
        # Sample candidate points using KDE
        candidate_points = self._sample_points(100 * batch_size, method='kde')

        # Calculate acquisition function values
        model = self._fit_model(self.X, self.y)
        acquisition_values = self._acquisition_function(candidate_points, model)

        # Select top batch_size points based on acquisition function values
        indices = np.argsort(-acquisition_values.flatten())[:batch_size]
        selected_points = candidate_points[indices]

        return selected_points

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

        # Update KDE
        if self.X is not None:
            # Identify promising regions (e.g., top 20% of evaluated points)
            threshold = np.percentile(self.y, 20)
            promising_points = self.X[(self.y <= threshold).flatten()]

            if len(promising_points) > self.dim + 1:  # Ensure enough points for KDE
                self.kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(promising_points)
            else:
                self.kde = None
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        initial_X = self._sample_points(self.n_init, method='sobol')
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        
        batch_size = 5
        while self.n_evals < self.budget:
            # Optimization
            model = self._fit_model(self.X, self.y)
            
            # select points by acquisition function
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x

```
The algorithm DEDABO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1679 with standard deviation 0.1015.

took 92.85 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

