You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- ALGTRBOUE: 0.1966, 1060.63 seconds, **Adaptive Landscape-Guided Trust Region Bayesian Optimization with Uncertainty-Aware Exploration (ALGTRBOUE)**

This algorithm combines the strengths of ALTRBO and ATRBOUE to achieve a more robust and efficient black-box optimization. It integrates landscape analysis (from ALTRBO) to adapt the exploration-exploitation trade-off with uncertainty-aware exploration (from ATRBOUE) within an adaptive trust region framework. The landscape analysis adjusts a temperature parameter in the Expected Improvement (EI) acquisition function. The uncertainty-aware exploration adds a term proportional to the predicted standard deviation to the EI, encouraging exploration in regions of high uncertainty. The trust region is adapted based on the success rate and model uncertainty.


- ATRDABOUE: 0.1949, 929.56 seconds, **Adaptive Trust Region Bayesian Optimization with Dynamic Acquisition Function Balancing and Uncertainty-Aware Exploration (ATRDABOUE)**

This algorithm synergistically combines the strengths of ATRBOUE and ATRDABO. It leverages the adaptive trust region mechanism of ATRBO, the uncertainty-aware exploration of ATRBOUE, and the dueling acquisition function selection of ATRDABO. The core idea is to dynamically balance exploration and exploitation within a dynamically adjusted trust region by adaptively selecting the most promising acquisition function and incorporating an uncertainty-aware exploration term. The acquisition function weights are updated based on their performance and a dynamic temperature parameter is introduced to control the exploration-exploitation trade-off in the Thompson Sampling process.


- AGETRBO: 0.1912, 486.39 seconds, Adaptive Gradient-Enhanced Trust Region Bayesian Optimization (AGETRBO) is a novel algorithm that combines the strengths of Adaptive Trust Region Bayesian Optimization (ATRBO) and Gradient-Enhanced Bayesian Optimization (GEBO). It adaptively adjusts the trust region based on the success of previous iterations, similar to ATRBO, but also incorporates gradient information, estimated using finite differences, into the Gaussian Process Regression (GPR) model, similar to GEBO. The acquisition function is Expected Improvement (EI). AGETRBO aims to improve the efficiency and robustness of the search by leveraging both trust region management and gradient information. The next points are selected by L-BFGS-B optimization within the trust region.


- AGETRBODE: 0.1910, 574.23 seconds, **Adaptive Gradient-Enhanced Trust Region Bayesian Optimization with Dynamic Exploration (AGETRBODE)** combines the strengths of AGETRBO and ATRBO by incorporating gradient information and adaptively adjusting the trust region. In addition, it introduces a dynamic exploration strategy based on the uncertainty of the Gaussian Process model, similar to ATRBOUE, but simplifies the uncertainty measure and integrates it directly into the acquisition function. The algorithm estimates gradients using finite differences, fits a Gaussian Process Regression (GPR) model, and uses Expected Improvement (EI) as the acquisition function. The trust region is adjusted based on the success rate, and the exploration-exploitation trade-off is dynamically controlled by the uncertainty of the GPR model.


- ATRBOUE: 0.1902, 761.17 seconds, **Adaptive Trust Region Bayesian Optimization with Uncertainty-Aware Exploration (ATRBOUE)**

This algorithm builds upon the foundation of ATRBO by incorporating a more sophisticated mechanism for balancing exploration and exploitation within the trust region framework. The core idea is to dynamically adjust the exploration-exploitation trade-off based on the uncertainty of the Gaussian Process model, specifically focusing on regions with high predictive variance. This is achieved by modifying the acquisition function to include an uncertainty-aware exploration term. Additionally, the trust region adaptation is refined to consider both the success rate and the uncertainty of the GPR model.


- AGEDTRBO: 0.1901, 797.00 seconds, **AGEDTRBO: Adaptive Gradient-Enhanced Diversity Trust Region Bayesian Optimization.** This algorithm combines the strengths of AGETRBO and ADTRBO by incorporating gradient information and a diversity-enhancing acquisition function within an adaptive trust region framework. It uses finite differences to estimate gradients, which are then used to improve the accuracy of the Gaussian Process Regression (GPR) model. The acquisition function combines Expected Improvement (EI) with a diversity term based on the minimum distance to existing points, promoting exploration and preventing premature convergence. The trust region is adaptively adjusted based on the success of previous iterations. The next points are selected by sampling a large number of candidates within the trust region and choosing the top ones based on the acquisition function.


- ATRSMABO_TS: 0.1899, 2137.12 seconds, **Adaptive Trust Region with Surrogate Model Averaging and Thompson Sampling Bayesian Optimization (ATRSMABO-TS)**

This algorithm refines ATRSMABO by replacing the fixed temperature-based Expected Improvement (EI) acquisition function with Thompson Sampling (TS) for acquisition function selection. It maintains the adaptive trust region and surrogate model averaging aspects of ATRSMABO but introduces a more adaptive and probabilistic approach to balancing exploration and exploitation. Instead of directly averaging the predictions of the RBF and Matern kernels, Thompson Sampling is used to sample from the posterior predictive distribution of each kernel, and the acquisition function is calculated based on these samples. This allows the algorithm to adaptively favor the kernel that is performing better, leading to more efficient exploration and exploitation. The trust region adaptation is also modified to incorporate the uncertainty estimates from the GPR models.


- ATRBODGE: 0.1897, 855.52 seconds, **Adaptive Trust Region Bayesian Optimization with Dynamic Exploration and Gradient Estimation (ATRBO-DGE)**

This algorithm combines the adaptive trust region strategy of ATRBO with dynamic exploration control and gradient estimation to improve search efficiency and robustness. It adaptively adjusts the trust region based on success and uncertainty, uses a dynamic exploration weight in the acquisition function, and incorporates gradient information to guide the search, particularly when the model uncertainty is low.




The selected solution to update is:
**Adaptive Trust Region Bayesian Optimization with Uncertainty-Aware Exploration (ATRBOUE)**

This algorithm builds upon the foundation of ATRBO by incorporating a more sophisticated mechanism for balancing exploration and exploitation within the trust region framework. The core idea is to dynamically adjust the exploration-exploitation trade-off based on the uncertainty of the Gaussian Process model, specifically focusing on regions with high predictive variance. This is achieved by modifying the acquisition function to include an uncertainty-aware exploration term. Additionally, the trust region adaptation is refined to consider both the success rate and the uncertainty of the GPR model.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

class ATRBOUE:
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
        self.trust_region_width = 2.0  # Initial trust region width
        self.success_threshold = 0.1 # Threshold for increasing trust region
        self.best_y = np.inf # Initialize best_y with a large value
        self.exploration_weight = 0.1 # Weight for exploration term in acquisition function

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, center=None, width=None):
        # sample points within the trust region
        # return array of shape (n_points, n_dims)
        if center is None:
            sampler = qmc.Sobol(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            # Sample within the trust region
            lower_bound = np.maximum(self.bounds[0], center - width / 2)
            upper_bound = np.minimum(self.bounds[1], center + width / 2)
            sampler = qmc.Sobol(d=self.dim)
            sample = sampler.random(n=n_points)
            scaled_sample = qmc.scale(sample, lower_bound, upper_bound)
            return scaled_sample

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        model.fit(X, y)
        return model

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
        imp = self.best_y - mu
        Z = imp / (sigma + 1e-9)  # Add a small constant to avoid division by zero
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # avoid division by zero, if sigma is too small, set EI to 0

        # Uncertainty-aware exploration
        exploration_term = self.exploration_weight * sigma
        acq_values = ei + exploration_term

        return acq_values

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Generate candidate points within the trust region
        n_candidates = max(1000, batch_size * 100)
        X_cand = self._sample_points(n_candidates, center=self.best_x, width=self.trust_region_width)

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
        self.best_x = X_init[np.argmin(y_init)][0]
        self.best_y = np.min(y_init)

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

            # Update best solution
            new_best_y = np.min(self.y)
            new_best_x = self.X[np.argmin(self.y)]

            # Adjust trust region width
            success_ratio = (self.best_y - new_best_y) / self.best_y
            mean_sigma = np.mean(self.model.predict(self.X, return_std=True)[1]) #average predicted variance

            if success_ratio > self.success_threshold and mean_sigma < 0.5:
                self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)  # Increase if successful and low uncertainty
            else:
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)  # Decrease if not successful or high uncertainty

            self.best_y = new_best_y
            self.best_x = new_best_x

        # Return best solution
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x

```
The algorithm ATRBOUE got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1902 with standard deviation 0.1074.

took 761.17 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

