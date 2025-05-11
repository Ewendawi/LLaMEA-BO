You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- DEBO: 0.1705, 63.62 seconds, **DEBO: Diversity Enhanced Bayesian Optimization:** This algorithm focuses on enhancing diversity in Bayesian Optimization using a combination of techniques. It employs a Gaussian Process Regression (GPR) model with a Radial Basis Function (RBF) kernel for surrogate modeling. The acquisition function is a combination of Expected Improvement (EI) and a distance-based diversity term. This diversity term encourages the selection of points that are far away from previously evaluated points. A clustering-based approach is used to identify diverse regions in the search space. The initial exploration is performed using a Sobol sequence to ensure good coverage of the search space.


- ATRBO: 0.1639, 99.40 seconds, **Adaptive Trust Region Bayesian Optimization (ATRBO):** This algorithm uses a Gaussian Process Regression (GPR) model with a Matérn kernel for surrogate modeling. It employs a trust region approach, adaptively adjusting the size of the trust region based on the agreement between the GPR model and the true objective function. The acquisition function is the Expected Improvement (EI) within the trust region. The initial exploration is performed using Latin Hypercube Sampling (LHS). To enhance diversity, a Sobol sequence is used for sampling within the trust region.


- ReBO: 0.1632, 51.69 seconds, **ReBO: Regularized Bayesian Optimization:** This algorithm introduces a regularization term to the acquisition function to prevent overfitting and promote exploration in Bayesian Optimization. It uses a Gaussian Process Regression (GPR) model with an RBF kernel for surrogate modeling. The acquisition function is a combination of Expected Improvement (EI) and a regularization term based on the L2 norm of the GPR's predicted mean. This regularization encourages the algorithm to explore regions where the model's predictions are less extreme, thus mitigating the risk of prematurely converging to a local optimum. Additionally, an adaptive scaling factor is introduced for the regularization term, which adjusts based on the iteration number, gradually decreasing the regularization as the algorithm progresses. The initial exploration is performed using Latin Hypercube Sampling (LHS).


- PaSBO: 0.1614, 55.38 seconds, **PaSBO: Pareto-based Acquisition Steering Bayesian Optimization:** This algorithm introduces a Pareto-based approach to manage multiple acquisition functions in Bayesian Optimization. It uses a Gaussian Process Regression (GPR) model with an RBF kernel for surrogate modeling. Instead of relying on a single acquisition function, PaSBO maintains a set of acquisition functions, including Expected Improvement (EI), Probability of Improvement (PI), and Upper Confidence Bound (UCB). A Pareto-based selection mechanism is employed to identify a set of non-dominated points based on these multiple acquisition functions. This allows the algorithm to explore the search space more effectively by balancing exploration and exploitation from different perspectives. The initial exploration is performed using a Latin Hypercube Sampling (LHS).


- EHBBO: 0.1592, 151.03 seconds, **Efficient Hybrid Bayesian Optimization (EHBBO):** This algorithm combines Gaussian Process Regression (GPR) with Expected Improvement (EI) acquisition, Latin Hypercube Sampling (LHS) for initial exploration, and a local search refinement step using a simple gradient-based method to enhance exploitation. A batch selection strategy based on Thompson Sampling is used to select multiple points for evaluation in each iteration, balancing exploration and exploitation.


- AGABO: 0.1411, 467.95 seconds, **AGABO: Asynchronous Gradient-Aware Bayesian Optimization:** This algorithm introduces an asynchronous parallelization strategy coupled with gradient information to enhance Bayesian Optimization. It employs a Gaussian Process Regression (GPR) model with a Matérn kernel for surrogate modeling. The acquisition function is a combination of Expected Improvement (EI) and a gradient-based exploration term. This term leverages the gradient of the GPR's predicted mean to guide exploration towards promising regions with steeper descent. The algorithm uses an asynchronous parallelization strategy, where multiple workers evaluate points concurrently, and the GPR model is updated asynchronously with the newly acquired data. This allows the algorithm to explore the search space more efficiently by mitigating the bottleneck of sequential evaluation. The initial exploration is performed using a Halton sequence to ensure good coverage of the search space. A key aspect is the asynchronous update of the GP model, which requires careful handling to avoid instability.


- CGHBO: 0.0000, 0.00 seconds, **CGHBO: Covariance Guided Hybrid Bayesian Optimization:** This algorithm introduces a novel approach to Bayesian Optimization by incorporating a covariance-based exploration strategy. It uses a Gaussian Process Regression (GPR) model with a dynamic kernel for surrogate modeling. The acquisition function is a hybrid of Expected Improvement (EI) and a covariance-guided exploration term. This term leverages the covariance matrix of the GPR posterior predictive distribution to identify regions of high uncertainty and potential improvement, promoting exploration in areas where the model is less confident. The algorithm also uses a combination of Latin Hypercube Sampling (LHS) and a covariance-guided sampling strategy for initial exploration. A gradient-based method is used for local refinement.


- DABO: 0.0000, 0.00 seconds, **DABO: Distribution-Aware Bayesian Optimization:** This algorithm introduces a distribution-aware exploration strategy within Bayesian Optimization. It uses a Gaussian Process Regression (GPR) model for surrogate modeling. The acquisition function combines Expected Improvement (EI) with a distribution matching term. This term encourages sampling from a distribution that is close to the distribution of promising regions identified by the GPR model. Kernel Density Estimation (KDE) is used to estimate the distribution of promising regions. A modified sampling strategy is employed to sample points from the learned distribution. The initial exploration is performed using Latin Hypercube Sampling (LHS).




The selected solutions to update are:
## ATRBO
**Adaptive Trust Region Bayesian Optimization (ATRBO):** This algorithm uses a Gaussian Process Regression (GPR) model with a Matérn kernel for surrogate modeling. It employs a trust region approach, adaptively adjusting the size of the trust region based on the agreement between the GPR model and the true objective function. The acquisition function is the Expected Improvement (EI) within the trust region. The initial exploration is performed using Latin Hypercube Sampling (LHS). To enhance diversity, a Sobol sequence is used for sampling within the trust region.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
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
        self.n_init = 2 * dim
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 2.0
        self.model_agreement_threshold = 0.75
        self.best_x = None
        self.best_y = float('inf')

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

    def _acquisition_function(self, X, model):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if self.best_y is None:
            return np.zeros_like(mu)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # handle zero sigma

        return ei

    def _select_next_points(self, batch_size, model, trust_region_center):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Sample within the trust region using Sobol sequence
        sobol_engine = qmc.Sobol(d=self.dim, seed=42)
        samples = sobol_engine.random(n=100 * batch_size)

        # Scale and shift samples to be within the trust region
        scaled_samples = trust_region_center + self.trust_region_radius * (2 * samples - 1)

        # Clip samples to stay within the problem bounds
        scaled_samples = np.clip(scaled_samples, self.bounds[0], self.bounds[1])

        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(scaled_samples, model)

        # Select top batch_size points based on acquisition function values
        indices = np.argsort(-acquisition_values.flatten())[:batch_size]
        selected_points = scaled_samples[indices]

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

        batch_size = 5
        while self.n_evals < self.budget:
            # Optimization
            model = self._fit_model(self.X, self.y)
            
            # select points by acquisition function
            next_X = self._select_next_points(batch_size, model, trust_region_center)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Model agreement check (simplified)
            predicted_y = model.predict(next_X)
            agreement = np.corrcoef(next_y.flatten(), predicted_y.flatten())[0, 1]

            # Adjust trust region size
            if np.isnan(agreement) or agreement < self.model_agreement_threshold:
                self.trust_region_radius *= self.trust_region_shrink_factor
            else:
                self.trust_region_radius *= self.trust_region_expand_factor
                self.trust_region_radius = min(self.trust_region_radius, 5.0) # Limit expansion

            # Update trust region center
            trust_region_center = self.X[np.argmin(self.y)]

        return self.best_y, self.best_x

```
The algorithm ATRBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1639 with standard deviation 0.1008.

took 99.40 seconds to run.

## PaSBO
**PaSBO: Pareto-based Acquisition Steering Bayesian Optimization:** This algorithm introduces a Pareto-based approach to manage multiple acquisition functions in Bayesian Optimization. It uses a Gaussian Process Regression (GPR) model with an RBF kernel for surrogate modeling. Instead of relying on a single acquisition function, PaSBO maintains a set of acquisition functions, including Expected Improvement (EI), Probability of Improvement (PI), and Upper Confidence Bound (UCB). A Pareto-based selection mechanism is employed to identify a set of non-dominated points based on these multiple acquisition functions. This allows the algorithm to explore the search space more effectively by balancing exploration and exploitation from different perspectives. The initial exploration is performed using a Latin Hypercube Sampling (LHS).


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.linalg import cholesky, solve_triangular

class PaSBO:
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
        self.acquisition_functions = ['ei', 'pi', 'ucb']
        self.ucb_kappa = 2.0
        self.best_x = None
        self.best_y = float('inf')

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
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X, model, acq_type='ei'):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if self.best_y is None:
            return np.zeros_like(mu)

        imp = self.best_y - mu
        Z = imp / sigma

        if acq_type == 'ei':
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0  # handle zero sigma
            return ei
        elif acq_type == 'pi':
            pi = norm.cdf(Z)
            return pi
        elif acq_type == 'ucb':
            ucb = mu + self.ucb_kappa * sigma
            return ucb
        else:
            raise ValueError("Invalid acquisition function type.")

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

    def _select_next_points(self, batch_size, model):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # return array of shape (batch_size, n_dims)

        candidate_points = self._sample_points(100 * batch_size)
        acquisition_values = np.zeros((candidate_points.shape[0], len(self.acquisition_functions)))

        for i, acq_type in enumerate(self.acquisition_functions):
            acquisition_values[:, i] = self._acquisition_function(candidate_points, model, acq_type).flatten()

        # Find Pareto front
        is_efficient = self._is_pareto_efficient(acquisition_values)
        pareto_points = candidate_points[is_efficient]

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
        
        batch_size = 5
        while self.n_evals < self.budget:
            # Optimization
            model = self._fit_model(self.X, self.y)
            
            # select points by acquisition function
            next_X = self._select_next_points(batch_size, model)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x

```
The algorithm PaSBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1614 with standard deviation 0.0951.

took 55.38 seconds to run.

Combine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it. If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.




Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

