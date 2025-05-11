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


- LaAGETRBODGEAD: 0.1929, 525.23 seconds, **Landscape-Aware Adaptive Gradient-Enhanced Trust Region Bayesian Optimization with Dynamic Exploration and Adaptive Delta (LaAGETRBODGEAD)** This algorithm combines the strengths of LaAGETRBOUE and ATRBODGEAD. It leverages landscape analysis, gradient estimation with adaptive delta, adaptive trust region, and dynamic exploration based on both landscape correlation and model uncertainty. The gradient estimation uses an adaptive delta parameter based on the trust region size, similar to ATRBODGEAD. Landscape analysis adjusts the temperature parameter in the EI acquisition function, and an uncertainty-aware exploration term is added to the EI, similar to LaAGETRBOUE. Additionally, a dynamic exploration weight is introduced, adjusting the exploration term based on the landscape correlation. The trust region is adapted based on success rate and model uncertainty. L-BFGS-B optimization is used within the trust region to select the next points.


- ALGTRBODTUE: 0.1929, 787.30 seconds, **Adaptive Landscape-Guided Trust Region Bayesian Optimization with Dynamic Temperature and Uncertainty-Aware Exploration (ALGTRBODTUE)**

This algorithm refines ALGTRBOUE by introducing a more sophisticated dynamic temperature adjustment mechanism based on both landscape correlation and the uncertainty of the Gaussian Process model. The temperature, which controls the exploration-exploitation trade-off in the Expected Improvement (EI) acquisition function, is adapted based on the landscape correlation (as in ALGTRBOUE) and the average predicted standard deviation (uncertainty) of the GPR model. This allows for a more nuanced control of exploration, increasing it when the landscape is rugged or the model is uncertain, and decreasing it when the landscape is smooth and the model is confident. The exploration weight is also dynamically adjusted based on the uncertainty. This helps the algorithm to strike a better balance between exploration and exploitation, leading to more efficient optimization.


- LaAGETRBOUE: 0.1926, 563.85 seconds, **LaAGETRBOUE: Landscape-Aware Adaptive Gradient-Enhanced Trust Region Bayesian Optimization with Uncertainty-Aware Exploration.** This algorithm combines landscape analysis, gradient estimation, adaptive trust region, and uncertainty-aware exploration. It estimates gradients using finite differences to enhance the Gaussian Process Regression (GPR) model. Landscape analysis dynamically adjusts a temperature parameter in the Expected Improvement (EI) acquisition function. An uncertainty-aware exploration term is added to the EI, encouraging exploration in regions of high uncertainty. The trust region is adapted based on the success rate and model uncertainty. L-BFGS-B optimization is used within the trust region to select the next points.


- LaAGETRBOUEPlus: 0.1922, 571.86 seconds, **LaAGETRBOUE+: Landscape-Aware Adaptive Gradient-Enhanced Trust Region Bayesian Optimization with Uncertainty-Aware Exploration and Adaptive Delta.** This algorithm combines landscape analysis, gradient estimation, adaptive trust region, uncertainty-aware exploration, and adaptive delta for gradient estimation. It estimates gradients using finite differences to enhance the Gaussian Process Regression (GPR) model. Landscape analysis dynamically adjusts a temperature parameter in the Expected Improvement (EI) acquisition function. An uncertainty-aware exploration term is added to the EI, encouraging exploration in regions of high uncertainty. The trust region is adapted based on the success rate and model uncertainty. L-BFGS-B optimization is used within the trust region to select the next points. The delta value used in the finite difference method for gradient estimation is dynamically adjusted based on the trust region width.


- ATRBODGEADv2: 0.1921, 996.37 seconds, **ATRBODGEADv2: Adaptive Trust Region Bayesian Optimization with Dynamic Exploration, Gradient Estimation, Adaptive Delta, and Enhanced Acquisition Function.** This algorithm refines ATRBODGEAD by incorporating two key improvements: 1) An enhanced acquisition function that combines Expected Improvement (EI), an uncertainty-aware exploration term, and a gradient-based exploitation term, but with adaptive weights for each term based on the trust region size and model uncertainty. The gradient term is only activated when the model is sufficiently confident and the trust region is small enough, and its weight is dynamically adjusted. 2) Instead of using a fixed success threshold, the algorithm now adapts the success threshold based on the dimensionality of the problem. This allows for a more robust trust region adaptation across different problem dimensions.


- ATRBODGEAD: 0.1914, 1040.20 seconds, **Adaptive Trust Region Bayesian Optimization with Dynamic Exploration, Gradient Estimation, and Adaptive Delta (ATRBO-DGE-AD)**

This algorithm builds upon ATRBO-DGE by introducing an adaptive delta parameter for gradient estimation. The delta value used in the finite difference method for gradient estimation is dynamically adjusted based on the trust region width. A smaller trust region width implies a more localized search, and thus a smaller delta is used for more accurate gradient estimation. Conversely, a larger trust region width warrants a larger delta to explore a broader range. This adaptive delta aims to improve the accuracy and efficiency of gradient estimation, leading to better exploitation of the local landscape.




The selected solutions to update are:
## ALGTRBOUE
**Adaptive Landscape-Guided Trust Region Bayesian Optimization with Uncertainty-Aware Exploration (ALGTRBOUE)**

This algorithm combines the strengths of ALTRBO and ATRBOUE to achieve a more robust and efficient black-box optimization. It integrates landscape analysis (from ALTRBO) to adapt the exploration-exploitation trade-off with uncertainty-aware exploration (from ATRBOUE) within an adaptive trust region framework. The landscape analysis adjusts a temperature parameter in the Expected Improvement (EI) acquisition function. The uncertainty-aware exploration adds a term proportional to the predicted standard deviation to the EI, encouraging exploration in regions of high uncertainty. The trust region is adapted based on the success rate and model uncertainty.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances

class ALGTRBOUE:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(10 * dim, self.budget // 5)
        self.trust_region_width = 2.0  # Initial trust region width
        self.success_threshold = 0.1  # Threshold for increasing trust region
        self.best_y = np.inf  # Initialize best_y with a large value
        self.best_x = None
        self.temperature = 1.0  # Initial temperature for exploration
        self.landscape_correlation = 0.0  # Initial landscape correlation
        self.smoothness_threshold = 0.5  # Threshold for considering the landscape smooth
        self.exploration_weight = 0.1  # Weight for exploration term in acquisition function

    def _sample_points(self, n_points, center=None, width=None):
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
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        model.fit(X, y)
        return model

    def _analyze_landscape(self):
        if self.X is None or self.y is None or len(self.X) < 2:
            return 0.0

        distances = pairwise_distances(self.X)
        value_differences = np.abs(self.y - self.y.T)

        # Flatten the matrices and remove the diagonal elements
        distances = distances.flatten()
        value_differences = value_differences.flatten()
        indices = np.arange(len(distances))
        distances = distances[indices % (len(self.X) + 1) != 0]
        value_differences = value_differences[indices % (len(self.X) + 1) != 0]

        # Calculate Pearson correlation coefficient
        correlation, _ = pearsonr(distances, value_differences)
        return correlation if not np.isnan(correlation) else 0.0

    def _acquisition_function(self, X):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement with temperature
        imp = self.best_y - mu
        Z = imp / (self.temperature * sigma + 1e-9)  # Adding a small constant to avoid division by zero
        ei = imp * norm.cdf(Z) + self.temperature * sigma * norm.pdf(Z)
        ei = np.clip(ei, 0, 1e10)  # Clip EI to avoid potential NaN issues
        ei[sigma <= 1e-6] = 0.0  # avoid division by zero

        # Uncertainty-aware exploration
        exploration_term = self.exploration_weight * sigma
        acq_values = ei + exploration_term

        return acq_values

    def _select_next_points(self, batch_size):
        n_candidates = max(1000, batch_size * 100)
        X_cand = self._sample_points(n_candidates, center=self.best_x, width=self.trust_region_width)

        acq_values = self._acquisition_function(X_cand)

        top_indices = np.argsort(acq_values.flatten())[::-1][:batch_size]
        return X_cand[top_indices]

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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_x = X_init[np.argmin(y_init)]
        self.best_y = np.min(y_init)

        # Optimization loop
        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            # Fit the model
            self.model = self._fit_model(self.X, self.y)

            # Analyze the landscape
            self.landscape_correlation = self._analyze_landscape()

            # Adjust temperature based on landscape correlation
            if self.landscape_correlation > self.smoothness_threshold:
                self.temperature = max(0.1, self.temperature * 0.9)  # Reduce temperature for smoother landscapes
            else:
                self.temperature = min(2.0, self.temperature * 1.1)  # Increase temperature for rugged landscapes

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
            mean_sigma = np.mean(self.model.predict(self.X, return_std=True)[1])

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
The algorithm ALGTRBOUE got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1966 with standard deviation 0.1158.

took 1060.63 seconds to run.

## ALGTRBODTUE
**Adaptive Landscape-Guided Trust Region Bayesian Optimization with Dynamic Temperature and Uncertainty-Aware Exploration (ALGTRBODTUE)**

This algorithm refines ALGTRBOUE by introducing a more sophisticated dynamic temperature adjustment mechanism based on both landscape correlation and the uncertainty of the Gaussian Process model. The temperature, which controls the exploration-exploitation trade-off in the Expected Improvement (EI) acquisition function, is adapted based on the landscape correlation (as in ALGTRBOUE) and the average predicted standard deviation (uncertainty) of the GPR model. This allows for a more nuanced control of exploration, increasing it when the landscape is rugged or the model is uncertain, and decreasing it when the landscape is smooth and the model is confident. The exploration weight is also dynamically adjusted based on the uncertainty. This helps the algorithm to strike a better balance between exploration and exploitation, leading to more efficient optimization.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances

class ALGTRBODTUE:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = min(10 * dim, self.budget // 5)
        self.trust_region_width = 2.0  # Initial trust region width
        self.success_threshold = 0.1  # Threshold for increasing trust region
        self.best_y = np.inf  # Initialize best_y with a large value
        self.best_x = None
        self.temperature = 1.0  # Initial temperature for exploration
        self.landscape_correlation = 0.0  # Initial landscape correlation
        self.smoothness_threshold = 0.5  # Threshold for considering the landscape smooth
        self.exploration_weight = 0.1  # Initial weight for exploration term in acquisition function
        self.temperature_scaling = 0.1

    def _sample_points(self, n_points, center=None, width=None):
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
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        model.fit(X, y)
        return model

    def _analyze_landscape(self):
        if self.X is None or self.y is None or len(self.X) < 2:
            return 0.0

        distances = pairwise_distances(self.X)
        value_differences = np.abs(self.y - self.y.T)

        # Flatten the matrices and remove the diagonal elements
        distances = distances.flatten()
        value_differences = value_differences.flatten()
        indices = np.arange(len(distances))
        distances = distances[indices % (len(self.X) + 1) != 0]
        value_differences = value_differences[indices % (len(self.X) + 1) != 0]

        # Calculate Pearson correlation coefficient
        correlation, _ = pearsonr(distances, value_differences)
        return correlation if not np.isnan(correlation) else 0.0

    def _acquisition_function(self, X):
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement with temperature
        imp = self.best_y - mu
        Z = imp / (self.temperature * sigma + 1e-9)  # Adding a small constant to avoid division by zero
        ei = imp * norm.cdf(Z) + self.temperature * sigma * norm.pdf(Z)
        ei = np.clip(ei, 0, 1e10)  # Clip EI to avoid potential NaN issues
        ei[sigma <= 1e-6] = 0.0  # avoid division by zero

        # Uncertainty-aware exploration
        exploration_term = self.exploration_weight * sigma
        acq_values = ei + exploration_term

        return acq_values

    def _select_next_points(self, batch_size):
        n_candidates = max(1000, batch_size * 100)
        X_cand = self._sample_points(n_candidates, center=self.best_x, width=self.trust_region_width)

        acq_values = self._acquisition_function(X_cand)

        top_indices = np.argsort(acq_values.flatten())[::-1][:batch_size]
        return X_cand[top_indices]

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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        self.best_x = X_init[np.argmin(y_init)]
        self.best_y = np.min(y_init)

        # Optimization loop
        batch_size = max(1, self.dim // 2)
        while self.n_evals < self.budget:
            # Fit the model
            self.model = self._fit_model(self.X, self.y)

            # Analyze the landscape
            self.landscape_correlation = self._analyze_landscape()

            # Calculate mean sigma (uncertainty)
            mean_sigma = np.mean(self.model.predict(self.X, return_std=True)[1])

            # Adjust temperature based on landscape correlation and uncertainty
            if self.landscape_correlation > self.smoothness_threshold:
                self.temperature = max(0.1, self.temperature * (1 - self.temperature_scaling))  # Reduce temperature for smoother landscapes
            else:
                self.temperature = min(2.0, self.temperature * (1 + self.temperature_scaling))  # Increase temperature for rugged landscapes

            self.temperature = max(0.01, min(2.0, self.temperature * (1 + (mean_sigma - 0.5) * 0.1)))

            # Adjust exploration weight based on uncertainty
            self.exploration_weight = min(0.5, max(0.01, 0.1 + (mean_sigma - 0.5) * 0.2))

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
The algorithm ALGTRBODTUE got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1929 with standard deviation 0.1105.

took 787.30 seconds to run.

Combine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it. If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.




Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

