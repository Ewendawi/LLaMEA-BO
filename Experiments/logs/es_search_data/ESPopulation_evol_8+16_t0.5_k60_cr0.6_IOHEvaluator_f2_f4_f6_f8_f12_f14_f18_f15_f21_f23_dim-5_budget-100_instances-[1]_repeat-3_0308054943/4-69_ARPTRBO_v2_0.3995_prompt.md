You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- GADETBO: 0.1795, 109.58 seconds, **GADETBO: Gradient-Aware Diversity Enhanced Trust Region Bayesian Optimization:** This algorithm combines the strengths of AGATBO and DEBO, focusing on balancing exploration and exploitation by incorporating gradient information, diversity enhancement, and a trust region strategy. It uses a Gaussian Process Regression (GPR) model with a Matérn kernel for surrogate modeling. The acquisition function is a combination of Expected Improvement (EI), a gradient-based exploration term, and a distance-based diversity term, all evaluated within an adaptive trust region. The trust region size is adjusted based on the agreement between the GPR model and the true objective function. Clustering is used to identify diverse regions for sampling.


- GADETREBO: 0.1783, 111.03 seconds, **GADETREBO: Gradient-Aware Diversity Enhanced Trust Region Bayesian Optimization with Exploration Regularization:** This algorithm synergistically combines the strengths of GADETBO and TREDABO, incorporating gradient information, diversity enhancement, trust region adaptation, and exploration regularization to achieve a robust and efficient optimization process. It uses a Gaussian Process Regression (GPR) model with a Matérn kernel for surrogate modeling. The acquisition function is a combination of Expected Improvement (EI), a gradient-based exploration term, a distance-based diversity term, and an exploration regularization term, all evaluated within an adaptive trust region. The trust region size is adjusted based on the agreement between the GPR model and the true objective function. Clustering is used to identify diverse regions for sampling. The exploration regularization term dynamically adjusts the exploration-exploitation balance based on the uncertainty of the GPR model and the trust region size.


- AGATBO: 0.1767, 510.80 seconds, **AGATBO: Adaptive Gradient-Aware Trust Region Bayesian Optimization:** This algorithm combines the strengths of AGABO and ATRBO, incorporating gradient information into the acquisition function within an adaptive trust region framework. It uses a Gaussian Process Regression (GPR) model with a Matérn kernel for surrogate modeling. The acquisition function is a combination of Expected Improvement (EI) and a gradient-based exploration term, evaluated within a trust region. The size of the trust region is adaptively adjusted based on the agreement between the GPR model and the true objective function. Halton sequence is used for initial exploration, and Sobol sequence is used for sampling within the trust region.


- GADETBO_RBF: 0.1755, 111.20 seconds, **GADETBO-RBF: Gradient-Aware Diversity Enhanced Trust Region Bayesian Optimization with RBF Kernel:** This algorithm builds upon GADETBO by replacing the Matérn kernel with a Radial Basis Function (RBF) kernel in the Gaussian Process Regression (GPR) model. The RBF kernel's smoothness can be beneficial for certain types of black-box functions. Additionally, the gradient prediction is enhanced by using a more accurate central difference method. The diversity term is also modified to be more robust to outliers by using the median distance instead of the minimum distance. Finally, the trust region update is modified to use a more robust agreement check based on the Spearman correlation.


- GRADEBO: 0.1745, 303.07 seconds, **GRADEBO: Gradient-Regularized Adaptive Diversity Enhanced Bayesian Optimization:** This algorithm combines gradient information, adaptive regularization, and diversity enhancement within a trust region framework to improve Bayesian optimization. It leverages a Gaussian Process Regression (GPR) model with a Matérn kernel for surrogate modeling. The acquisition function integrates Expected Improvement (EI), a gradient-based exploration term, an adaptive regularization term, and a distance-based diversity term. The trust region size is adaptively adjusted based on the agreement between the GPR model and the true objective function. The gradient is estimated efficiently using finite differences. An adaptive batch size strategy is also incorporated to dynamically balance exploration and exploitation.


- AGAPTRBO: 0.1727, 512.47 seconds, **AGAPTRBO: Adaptive Gradient-Aware Pareto Trust Region Bayesian Optimization:** This algorithm combines the strengths of AGATBO and APTRBO by integrating gradient information into the Pareto-based multi-acquisition function approach within an adaptive trust region framework. It uses a Gaussian Process Regression (GPR) model with a Matérn kernel for surrogate modeling. The acquisition function is a Pareto front of Expected Improvement (EI), Probability of Improvement (PI), Upper Confidence Bound (UCB), and a gradient-based exploration term, evaluated within a trust region. The size of the trust region is adaptively adjusted based on the agreement between the GPR model and the true objective function. The UCB's kappa parameter is dynamically adjusted based on the trust region size to balance exploration and exploitation.


- APTRBO: 0.1724, 104.60 seconds, **Adaptive Pareto Trust Region Bayesian Optimization (APTRBO):** This algorithm combines the strengths of Adaptive Trust Region Bayesian Optimization (ATRBO) and Pareto-based Acquisition Steering Bayesian Optimization (PaSBO) to achieve a more robust and efficient optimization process. It adaptively adjusts the trust region based on model agreement, uses a Pareto-based approach to manage multiple acquisition functions (EI, PI, UCB) within the trust region, and incorporates a dynamic exploration-exploitation balance by adjusting the UCB kappa parameter based on the trust region size. A Matern kernel is used for the Gaussian Process Regression (GPR) model.


- ARPTRBO: 0.1723, 288.39 seconds, **Adaptive Regularized Pareto Trust Region Bayesian Optimization (ARPTRBO):** This algorithm synergistically integrates the strengths of APTRBO and ARBO. It employs a trust region, adaptively adjusted based on model agreement, and utilizes a Pareto-based approach to manage multiple acquisition functions (EI, PI, UCB) within the trust region, similar to APTRBO. Additionally, it incorporates the adaptive regularization term and dynamic batch size strategy from ARBO to enhance exploration and exploitation. The regularization term is modified to include uncertainty from the Gaussian Process, and an exploration factor is added to the acquisition function. This combination aims to balance exploration and exploitation more effectively, leading to improved performance.




The selected solution to update is:
**Adaptive Regularized Pareto Trust Region Bayesian Optimization (ARPTRBO):** This algorithm synergistically integrates the strengths of APTRBO and ARBO. It employs a trust region, adaptively adjusted based on model agreement, and utilizes a Pareto-based approach to manage multiple acquisition functions (EI, PI, UCB) within the trust region, similar to APTRBO. Additionally, it incorporates the adaptive regularization term and dynamic batch size strategy from ARBO to enhance exploration and exploitation. The regularization term is modified to include uncertainty from the Gaussian Process, and an exploration factor is added to the acquisition function. This combination aims to balance exploration and exploitation more effectively, leading to improved performance.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.optimize import minimize

class ARPTRBO:
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
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 2.0
        self.model_agreement_threshold = 0.75
        self.best_x = None
        self.best_y = float('inf')
        self.acquisition_functions = ['ei', 'pi', 'ucb']
        self.ucb_kappa = 2.0 # Initial value
        self.ucb_kappa_max = 5.0
        self.ucb_kappa_min = 1.0
        self.reg_weight = 0.1 # Initial weight for the regularization term
        self.exploration_factor = 0.01 # Add an exploration factor

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
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X, model, iteration, acq_type='ei'):
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

            if acq_type == 'ei':
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma <= 1e-6] = 0.0  # handle zero sigma
            elif acq_type == 'pi':
                ei = norm.cdf(Z)
            elif acq_type == 'ucb':
                ei = mu + self.ucb_kappa * sigma
            else:
                raise ValueError("Invalid acquisition function type.")

        # Adaptive Regularization
        reg_weight = self.reg_weight * (1 - (iteration / self.budget))
        regularization_term = -reg_weight * np.linalg.norm(mu / (sigma + 1e-6), axis=1, keepdims=True)**2 # Uncertainty aware regularization
        ei = ei + regularization_term + self.exploration_factor * sigma # Add exploration factor

        return ei

    def _is_pareto_efficient(self, points):
        """
        Find the pareto-efficient points
        :param points: An n by m matrix of points
        :return: A boolean array of length n, True for pareto-efficient points, False otherwise
        """
        is_efficient = np.ones(points.shape[0], dtype=bool)
        for i, c in enumerate(points):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(points[is_efficient] > c, axis=1) | (points[is_efficient] == c).all(axis=1)
                is_efficient[i] = True  # Keep current point
        return is_efficient

    def _select_next_points(self, batch_size, model, iteration, trust_region_center):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # return array of shape (batch_size, n_dims)

        # Sample within the trust region using Sobol sequence
        sobol_engine = qmc.Sobol(d=self.dim, seed=42)
        samples = sobol_engine.random(n=100 * batch_size)

        # Scale and shift samples to be within the trust region
        scaled_samples = trust_region_center + self.trust_region_radius * (2 * samples - 1)

        # Clip samples to stay within the problem bounds
        scaled_samples = np.clip(scaled_samples, self.bounds[0], self.bounds[1])

        acquisition_values = np.zeros((scaled_samples.shape[0], len(self.acquisition_functions)))

        for i, acq_type in enumerate(self.acquisition_functions):
            acquisition_values[:, i] = self._acquisition_function(scaled_samples, model, iteration, acq_type).flatten()

        # Find Pareto front
        is_efficient = self._is_pareto_efficient(acquisition_values)
        pareto_points = scaled_samples[is_efficient]

        # Select top batch_size points from Pareto front
        if len(pareto_points) > batch_size:
            # Randomly select if more than batch_size
            indices = np.random.choice(len(pareto_points), batch_size, replace=False)
            selected_points = pareto_points[indices]
        else:
            # Use all Pareto points if less than or equal to batch_size
            selected_points = pareto_points

            # If still less than batch_size, sample randomly
            if len(selected_points) < batch_size:
                remaining = batch_size - len(selected_points)
                random_points = self._sample_points(remaining)
                selected_points = np.vstack((selected_points, random_points))

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
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        
        trust_region_center = self.X[np.argmin(self.y)] # Initialize trust region center

        iteration = self.n_init
        while self.n_evals < self.budget:
            # Adaptive batch size
            batch_size = max(1, int(5 * (1 - self.n_evals / self.budget))) # Linearly decreasing batch size

            # Optimization
            model = self._fit_model(self.X, self.y)
            
            # select points by acquisition function
            next_X = self._select_next_points(batch_size, model, iteration, trust_region_center)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Model agreement check (simplified)
            predicted_y = model.predict(next_X)
            agreement = np.corrcoef(next_y.flatten(), predicted_y.flatten())[0, 1]

            # Adjust trust region size
            if np.isnan(agreement) or agreement < self.model_agreement_threshold:
                self.trust_region_radius *= self.trust_region_shrink_factor
                self.ucb_kappa = min(self.ucb_kappa * 1.1, self.ucb_kappa_max) # Increase kappa for exploration
            else:
                self.trust_region_radius *= self.trust_region_expand_factor
                self.trust_region_radius = min(self.trust_region_radius, 5.0) # Limit expansion
                self.ucb_kappa = max(self.ucb_kappa * 0.9, self.ucb_kappa_min) # Decrease kappa for exploitation

            # Update trust region center
            trust_region_center = self.X[np.argmin(self.y)]
            iteration += batch_size

        return self.best_y, self.best_x

```
The algorithm ARPTRBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1723 with standard deviation 0.1080.

took 288.39 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

