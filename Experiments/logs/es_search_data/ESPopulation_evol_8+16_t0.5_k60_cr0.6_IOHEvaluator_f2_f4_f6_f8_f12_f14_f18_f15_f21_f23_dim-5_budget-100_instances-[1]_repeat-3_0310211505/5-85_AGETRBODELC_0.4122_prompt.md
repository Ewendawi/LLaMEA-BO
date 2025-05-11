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


- ALGTRBODTUE: 0.1929, 787.30 seconds, **Adaptive Landscape-Guided Trust Region Bayesian Optimization with Dynamic Temperature and Uncertainty-Aware Exploration (ALGTRBODTUE)**

This algorithm refines ALGTRBOUE by introducing a more sophisticated dynamic temperature adjustment mechanism based on both landscape correlation and the uncertainty of the Gaussian Process model. The temperature, which controls the exploration-exploitation trade-off in the Expected Improvement (EI) acquisition function, is adapted based on the landscape correlation (as in ALGTRBOUE) and the average predicted standard deviation (uncertainty) of the GPR model. This allows for a more nuanced control of exploration, increasing it when the landscape is rugged or the model is uncertain, and decreasing it when the landscape is smooth and the model is confident. The exploration weight is also dynamically adjusted based on the uncertainty. This helps the algorithm to strike a better balance between exploration and exploitation, leading to more efficient optimization.


- LaAGETRBOUE: 0.1926, 563.85 seconds, **LaAGETRBOUE: Landscape-Aware Adaptive Gradient-Enhanced Trust Region Bayesian Optimization with Uncertainty-Aware Exploration.** This algorithm combines landscape analysis, gradient estimation, adaptive trust region, and uncertainty-aware exploration. It estimates gradients using finite differences to enhance the Gaussian Process Regression (GPR) model. Landscape analysis dynamically adjusts a temperature parameter in the Expected Improvement (EI) acquisition function. An uncertainty-aware exploration term is added to the EI, encouraging exploration in regions of high uncertainty. The trust region is adapted based on the success rate and model uncertainty. L-BFGS-B optimization is used within the trust region to select the next points.


- ATRBODGEAD: 0.1914, 1040.20 seconds, **Adaptive Trust Region Bayesian Optimization with Dynamic Exploration, Gradient Estimation, and Adaptive Delta (ATRBO-DGE-AD)**

This algorithm builds upon ATRBO-DGE by introducing an adaptive delta parameter for gradient estimation. The delta value used in the finite difference method for gradient estimation is dynamically adjusted based on the trust region width. A smaller trust region width implies a more localized search, and thus a smaller delta is used for more accurate gradient estimation. Conversely, a larger trust region width warrants a larger delta to explore a broader range. This adaptive delta aims to improve the accuracy and efficiency of gradient estimation, leading to better exploitation of the local landscape.


- AGETRBO: 0.1912, 486.39 seconds, Adaptive Gradient-Enhanced Trust Region Bayesian Optimization (AGETRBO) is a novel algorithm that combines the strengths of Adaptive Trust Region Bayesian Optimization (ATRBO) and Gradient-Enhanced Bayesian Optimization (GEBO). It adaptively adjusts the trust region based on the success of previous iterations, similar to ATRBO, but also incorporates gradient information, estimated using finite differences, into the Gaussian Process Regression (GPR) model, similar to GEBO. The acquisition function is Expected Improvement (EI). AGETRBO aims to improve the efficiency and robustness of the search by leveraging both trust region management and gradient information. The next points are selected by L-BFGS-B optimization within the trust region.


- AGETRBODE: 0.1910, 574.23 seconds, **Adaptive Gradient-Enhanced Trust Region Bayesian Optimization with Dynamic Exploration (AGETRBODE)** combines the strengths of AGETRBO and ATRBO by incorporating gradient information and adaptively adjusting the trust region. In addition, it introduces a dynamic exploration strategy based on the uncertainty of the Gaussian Process model, similar to ATRBOUE, but simplifies the uncertainty measure and integrates it directly into the acquisition function. The algorithm estimates gradients using finite differences, fits a Gaussian Process Regression (GPR) model, and uses Expected Improvement (EI) as the acquisition function. The trust region is adjusted based on the success rate, and the exploration-exploitation trade-off is dynamically controlled by the uncertainty of the GPR model.


- ATRBOUE: 0.1902, 761.17 seconds, **Adaptive Trust Region Bayesian Optimization with Uncertainty-Aware Exploration (ATRBOUE)**

This algorithm builds upon the foundation of ATRBO by incorporating a more sophisticated mechanism for balancing exploration and exploitation within the trust region framework. The core idea is to dynamically adjust the exploration-exploitation trade-off based on the uncertainty of the Gaussian Process model, specifically focusing on regions with high predictive variance. This is achieved by modifying the acquisition function to include an uncertainty-aware exploration term. Additionally, the trust region adaptation is refined to consider both the success rate and the uncertainty of the GPR model.




The selected solution to update is:
**Adaptive Gradient-Enhanced Trust Region Bayesian Optimization with Dynamic Exploration (AGETRBODE)** combines the strengths of AGETRBO and ATRBO by incorporating gradient information and adaptively adjusting the trust region. In addition, it introduces a dynamic exploration strategy based on the uncertainty of the Gaussian Process model, similar to ATRBOUE, but simplifies the uncertainty measure and integrates it directly into the acquisition function. The algorithm estimates gradients using finite differences, fits a Gaussian Process Regression (GPR) model, and uses Expected Improvement (EI) as the acquisition function. The trust region is adjusted based on the success rate, and the exploration-exploitation trade-off is dynamically controlled by the uncertainty of the GPR model.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

class AGETRBODE:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(10*dim, self.budget//5)
        self.trust_region_width = 2.0
        self.success_threshold = 0.1
        self.best_y = np.inf
        self.delta = 1e-3
        self.exploration_weight = 0.1 # Weight for uncertainty-based exploration

    def _sample_points(self, n_points, center=None, width=None):
        if center is None:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            lower_bound = np.maximum(self.bounds[0], center - width / 2)
            upper_bound = np.minimum(self.bounds[1], center + width / 2)
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            scaled_sample = qmc.scale(sample, lower_bound, upper_bound)
            return scaled_sample

    def _fit_model(self, X, y):
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        model.fit(X, y)
        return model

    def _estimate_gradient(self, func, x):
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.delta
            x_minus[i] -= self.delta
            x_plus = np.clip(x_plus, self.bounds[0][i], self.bounds[1][i])
            x_minus = np.clip(x_minus, self.bounds[0][i], self.bounds[1][i])
            gradient[i] = (func(x_plus) - func(x_minus)) / (2 * self.delta)
        return gradient

    def _acquisition_function(self, X):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement
        imp = self.best_y - mu
        Z = imp / (sigma + 1e-9)
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Uncertainty-aware exploration
        exploration_term = self.exploration_weight * sigma
        
        return ei + exploration_term

    def _select_next_points(self, func, batch_size):
        X_next = []
        for _ in range(batch_size):
            def objective(x):
                x = x.reshape(1, -1)
                return -self._acquisition_function(x)[0, 0]

            lower_bound = np.maximum(self.bounds[0], self.best_x - self.trust_region_width / 2)
            upper_bound = np.minimum(self.bounds[1], self.best_x + self.trust_region_width / 2)
            bounds = [(lower_bound[i], upper_bound[i]) for i in range(self.dim)]
            
            x0 = self._sample_points(1, center=self.best_x, width=self.trust_region_width).flatten()
            
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            
            X_next.append(result.x)

        return np.array(X_next)

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.concatenate((self.X, new_X), axis=0)
            self.y = np.concatenate((self.y, new_y), axis=0)

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_x = X_init[np.argmin(y_init)]
        self.best_y = np.min(y_init)

        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            self.model = self._fit_model(self.X, self.y)

            X_next = self._select_next_points(func, batch_size)

            y_next = self._evaluate_points(func, X_next)

            self._update_eval_points(X_next, y_next)

            new_best_y = np.min(self.y)
            new_best_x = self.X[np.argmin(self.y)]

            if (self.best_y - new_best_y) / self.best_y > self.success_threshold:
                self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)
            else:
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)

            self.best_y = new_best_y
            self.best_x = new_best_x

            # Dynamically adjust exploration weight based on trust region width
            self.exploration_weight = 0.1 * (self.trust_region_width / 2.0)  # Scale exploration with trust region size

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x

```
The algorithm AGETRBODE got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1910 with standard deviation 0.1148.

took 574.23 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

