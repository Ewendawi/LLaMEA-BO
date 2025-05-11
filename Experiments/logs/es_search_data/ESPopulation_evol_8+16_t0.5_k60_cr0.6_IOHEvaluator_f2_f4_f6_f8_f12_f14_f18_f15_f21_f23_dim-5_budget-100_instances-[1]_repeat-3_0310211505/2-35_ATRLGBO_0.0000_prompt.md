You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- AGETRBO: 0.1912, 486.39 seconds, Adaptive Gradient-Enhanced Trust Region Bayesian Optimization (AGETRBO) is a novel algorithm that combines the strengths of Adaptive Trust Region Bayesian Optimization (ATRBO) and Gradient-Enhanced Bayesian Optimization (GEBO). It adaptively adjusts the trust region based on the success of previous iterations, similar to ATRBO, but also incorporates gradient information, estimated using finite differences, into the Gaussian Process Regression (GPR) model, similar to GEBO. The acquisition function is Expected Improvement (EI). AGETRBO aims to improve the efficiency and robustness of the search by leveraging both trust region management and gradient information. The next points are selected by L-BFGS-B optimization within the trust region.


- ATRBO: 0.1891, 936.74 seconds, Adaptive Trust Region Bayesian Optimization (ATRBO) is a novel metaheuristic algorithm for black-box optimization. It employs a Gaussian Process Regression (GPR) model as a surrogate. The algorithm adaptively adjusts the trust region based on the GPR's uncertainty and the success of previous iterations. It uses Expected Improvement (EI) as the acquisition function within the trust region. To address the `NaN` error from the previous algorithm, a robust EI calculation is implemented, and the K-means clustering is replaced with a simpler top-k selection based on acquisition function values.


- ATRDABO: 0.1827, 1031.59 seconds, Adaptive Trust Region with Dueling Acquisition Bayesian Optimization (ATRDABO) combines the adaptive trust region approach of ATRBO with the dueling acquisition function selection of DAEBO. This aims to balance exploration and exploitation by focusing the search within a dynamically adjusted trust region while adaptively selecting the most promising acquisition function. To address the `ValueError: probabilities contain NaN` error encountered in DAEBO, a normalization step with a small constant is added to the weight update mechanism. This ensures that the probabilities used in `np.random.choice` are always valid.


- ATRSMABO: 0.1822, 1612.25 seconds, Adaptive Trust Region with Surrogate Model Averaging Bayesian Optimization (ATRSMABO) is a novel metaheuristic algorithm for black-box optimization. It combines the strengths of Adaptive Trust Region Bayesian Optimization (ATRBO) and Surrogate Model Averaging Bayesian Optimization (SMABO). ATRSMABO employs an ensemble of Gaussian Process Regression (GPR) models with different kernels (RBF and Matern) as surrogate models. The algorithm adaptively adjusts the trust region based on the GPR's uncertainty and the success of previous iterations. It uses Expected Improvement (EI) as the acquisition function within the trust region, where the EI is calculated using the averaged predictions from the GPR ensemble. To further enhance exploration, a dynamic temperature parameter is introduced, influencing the exploration-exploitation trade-off within the EI calculation. The next points are selected by sampling a large number of candidates within the trust region and choosing the top ones based on the acquisition function.


- ALTRBO: 0.1817, 845.46 seconds, Adaptive Landscape-Guided Trust Region Bayesian Optimization (ALTRBO) is a Bayesian optimization algorithm that combines adaptive trust region management from ATRBO and landscape analysis from LGBO to guide the search process. It employs a Gaussian Process Regression (GPR) model as a surrogate. The algorithm adaptively adjusts the trust region based on the GPR's uncertainty and the success of previous iterations, while also using landscape analysis to adjust the exploration-exploitation trade-off by modifying the temperature parameter in the Expected Improvement (EI) acquisition function. This combination allows the algorithm to efficiently explore rugged landscapes and exploit smooth regions.


- ADTRBO: 0.1811, 938.13 seconds, Adaptive Diversity Trust Region Bayesian Optimization (ADTRBO): This algorithm combines the adaptive trust region strategy from ATRBO with the diversity-enhancing acquisition function from DEBO. It aims to balance exploration and exploitation by focusing the search within a dynamically adjusted trust region while simultaneously promoting diversity in the selected points to avoid premature convergence. The trust region width is adapted based on the success of previous iterations, and the acquisition function combines Expected Improvement (EI) with a diversity term based on the minimum distance to existing points.


- ATRBO_DTR: 0.1784, 194.29 seconds, Adaptive Trust Region Bayesian Optimization with Dynamic Trust Region Adjustment and Exploration-Exploitation Balancing (ATRBO-DTR). This algorithm builds upon the original ATRBO by incorporating a more sophisticated mechanism for adjusting the trust region width and balancing exploration and exploitation. The trust region width is adjusted based on both the success rate and the uncertainty of the Gaussian Process model. Additionally, the acquisition function is modified to include an exploration term based on the predicted variance, promoting exploration in uncertain regions. The initial sampling is also improved by using Sobol sequence, which is known to have better space-filling properties than Latin Hypercube Sampling, especially for low-dimensional problems.


- ATSMABO: 0.1638, 298.09 seconds, Adaptive Temperature Surrogate Model Averaging Bayesian Optimization (ATSMABO) is a Bayesian optimization algorithm that builds upon SMABO by introducing an adaptive temperature schedule and a more refined surrogate model averaging strategy. Instead of a fixed temperature decay, ATSMABO adjusts the temperature based on the landscape characteristics encountered during the optimization process. Specifically, it monitors the change in the best function value and adjusts the temperature decay rate accordingly. Furthermore, ATSMABO introduces a weighted averaging of the surrogate model predictions, where the weights are determined by the models' performance on a validation set.




The selected solutions to update are:
## ATRBO
Adaptive Trust Region Bayesian Optimization (ATRBO) is a novel metaheuristic algorithm for black-box optimization. It employs a Gaussian Process Regression (GPR) model as a surrogate. The algorithm adaptively adjusts the trust region based on the GPR's uncertainty and the success of previous iterations. It uses Expected Improvement (EI) as the acquisition function within the trust region. To address the `NaN` error from the previous algorithm, a robust EI calculation is implemented, and the K-means clustering is replaced with a simpler top-k selection based on acquisition function values.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

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
        self.n_init = min(10*dim, self.budget//5) # initial samples, at least 10*dim, at most 1/5 of budget
        self.trust_region_width = 2.0  # Initial trust region width
        self.success_threshold = 0.1 # Threshold for increasing trust region
        self.best_y = np.inf # Initialize best_y with a large value

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, center=None, width=None):
        # sample points within the trust region
        # return array of shape (n_points, n_dims)
        if center is None:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            # Sample within the trust region
            lower_bound = np.maximum(self.bounds[0], center - width / 2)
            upper_bound = np.minimum(self.bounds[1], center + width / 2)
            sampler = qmc.LatinHypercube(d=self.dim)
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
        return ei

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
            if (self.best_y - new_best_y) / self.best_y > self.success_threshold:
                self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)  # Increase if successful
            else:
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)  # Decrease if not successful

            self.best_y = new_best_y
            self.best_x = new_best_x

        # Return best solution
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x

```
The algorithm ATRBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1891 with standard deviation 0.1091.

took 936.74 seconds to run.

## ALTRBO
Adaptive Landscape-Guided Trust Region Bayesian Optimization (ALTRBO) is a Bayesian optimization algorithm that combines adaptive trust region management from ATRBO and landscape analysis from LGBO to guide the search process. It employs a Gaussian Process Regression (GPR) model as a surrogate. The algorithm adaptively adjusts the trust region based on the GPR's uncertainty and the success of previous iterations, while also using landscape analysis to adjust the exploration-exploitation trade-off by modifying the temperature parameter in the Expected Improvement (EI) acquisition function. This combination allows the algorithm to efficiently explore rugged landscapes and exploit smooth regions.


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

class ALTRBO:
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

    def _sample_points(self, n_points, center=None, width=None):
        if center is None:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            # Sample within the trust region
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

        return ei

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
            if (self.best_y - new_best_y) / self.best_y > self.success_threshold:
                self.trust_region_width = min(self.trust_region_width * 1.1, 10.0)  # Increase if successful
            else:
                self.trust_region_width = max(self.trust_region_width * 0.9, 0.1)  # Decrease if not successful

            self.best_y = new_best_y
            self.best_x = new_best_x

        # Return best solution
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        return best_y, best_x

```
The algorithm ALTRBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1817 with standard deviation 0.1078.

took 845.46 seconds to run.

Combine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it. If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.




Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

