# Description
**Efficient Hybrid Bayesian Optimization (EHBBO):** This algorithm combines an efficient space-filling initial design with a Gaussian Process (GP) surrogate model and a hybrid acquisition function. The initial design uses Latin Hypercube Sampling (LHS) to ensure good coverage of the search space. The GP model is used to predict the objective function value and its uncertainty. The acquisition function balances exploration and exploitation by combining the Expected Improvement (EI) and the Upper Confidence Bound (UCB). To improve computational efficiency, the algorithm employs a batch selection strategy using a simple greedy approach to select multiple points for evaluation in each iteration.

# Justification
1.  **Space-Filling Initial Design (LHS):** LHS is used to generate an initial set of diverse samples. This helps to build a good initial GP model and avoids premature convergence to local optima.
2.  **Gaussian Process Surrogate Model:** GPs are well-suited for modeling black-box functions because they provide both a prediction of the function value and an estimate of the uncertainty. This uncertainty is crucial for balancing exploration and exploitation.
3.  **Hybrid Acquisition Function (EI + UCB):** Combining EI and UCB promotes a balance between exploring regions with high potential for improvement (EI) and regions with high uncertainty (UCB). This helps to avoid getting stuck in local optima and to efficiently explore the search space.
4.  **Efficient Batch Selection:** A greedy approach is used for batch selection of points to evaluate. This is computationally efficient and allows the algorithm to make progress even when the function evaluation is expensive.
5.  **Computational Efficiency:** The algorithm is designed to be computationally efficient by using a relatively small number of initial samples, a simple GP model, and a greedy batch selection strategy. This makes it suitable for optimizing functions with a limited budget of evaluations.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

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
        self.n_init = 2 * dim # number of initial samples

        # Do not add any other arguments without a default value
        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.acq_strategy = "EI+UCB"
        self.exploration_weight = 0.2
        self.batch_size = 2
    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        samples = sampler.random(n=n_points)
        return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.acq_strategy == "EI":
            return self._expected_improvement(X)
        elif self.acq_strategy == "UCB":
            return self._upper_confidence_bound(X)
        elif self.acq_strategy == "EI+UCB":
            ei = self._expected_improvement(X)
            ucb = self._upper_confidence_bound(X)
            return ei + self.exploration_weight * ucb
        else:
            raise ValueError("Invalid acquisition function strategy.")

    def _expected_improvement(self, X):
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-6)  # avoid division by zero
        gamma = (self.best_y - mu) / sigma
        ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return ei.reshape(-1, 1)

    def _upper_confidence_bound(self, X):
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu.reshape(-1, 1) + 2 * sigma.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        selected_points = []
        candidates = self._sample_points(100 * self.dim) # Generate a larger candidate set
        
        for _ in range(batch_size):
            acq_values = self._acquisition_function(candidates)
            best_index = np.argmax(acq_values)
            selected_point = candidates[best_index]
            selected_points.append(selected_point)
            
            # Remove the selected point from the candidates to avoid duplicates in the batch
            candidates = np.delete(candidates, best_index, axis=0)
        
        return np.array(selected_points)

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
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))
            
        # Update best seen value
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Select points by acquisition function
            next_X = self._select_next_points(min(self.batch_size, self.budget - self.n_evals))

            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm EHBBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1394 with standard deviation 0.1024.

took 17.39 seconds to run.