# Description
Efficient Bayesian Optimization with Expected Improvement and Adaptive Batch Size (EI-ABS-BO). This algorithm uses a Gaussian Process (GP) as a surrogate model, Expected Improvement (EI) as the acquisition function, and an adaptive batch size strategy to balance exploration and exploitation. The batch size is adjusted based on the uncertainty of the GP predictions. QMC sampling is used for initial sampling and acquisition function optimization.

# Justification
*   **Gaussian Process Surrogate:** GPs provide a probabilistic model of the objective function, allowing for uncertainty quantification. This is crucial for effective exploration-exploitation trade-off.
*   **Expected Improvement Acquisition:** EI balances exploration and exploitation by considering both the predicted value and the uncertainty of the GP. It favors regions with high potential for improvement over the current best.
*   **Adaptive Batch Size:** The batch size is dynamically adjusted based on the GP's uncertainty. In regions of high uncertainty (exploration phase), a larger batch size is used to gather more information. In regions of low uncertainty (exploitation phase), a smaller batch size is used to refine the search around promising areas. This adaptive strategy improves the efficiency of the optimization process.
*   **QMC Sampling:** QMC sampling provides a more uniform coverage of the search space compared to random sampling, especially in high-dimensional spaces. This leads to a better initial exploration and a more accurate estimation of the acquisition function.
*   **Computational Efficiency:** The algorithm is designed to be computationally efficient by using a relatively simple GP model (e.g., RBF kernel) and by carefully managing the batch size to avoid unnecessary function evaluations.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class EI_ABS_BO:
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

        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.batch_size = 1

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        points = sampler.random(n=n_points)
        return qmc.scale(points, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0 # avoid nan values
        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Optimize the acquisition function using multiple random restarts
        best_x = None
        best_ei = -np.inf
        
        for _ in range(10):  # Multiple restarts
            x0 = self._sample_points(1)  # Random initial point
            
            # Define the objective function to minimize (negative EI)
            def objective(x):
                x = x.reshape(1, -1)
                return -self._acquisition_function(x)[0, 0]
            
            # Perform the optimization
            res = minimize(objective, x0, bounds=[(-5, 5)] * self.dim, method='L-BFGS-B')
            
            # Check if this restart found a better solution
            if -res.fun > best_ei:
                best_ei = -res.fun
                best_x = res.x
        
        return best_x.reshape(1, -1)

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
        if np.min(new_y) < self.best_y:
            self.best_y = np.min(new_y)
            self.best_x = new_X[np.argmin(new_y)]
    
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
            
            # Adaptive batch size
            _, std = self.gp.predict(self.X, return_std=True)
            mean_std = np.mean(std)
            self.batch_size = max(1, int(self.dim / (1 + mean_std * 10))) # Adjust batch size based on uncertainty
            self.batch_size = min(self.batch_size, self.budget - self.n_evals) # Ensure not exceeding budget

            # Select next points by acquisition function
            next_X = np.vstack([self._select_next_points(1) for _ in range(self.batch_size)])

            # Evaluate the points
            next_y = self._evaluate_points(func, next_X)
            
            # Update evaluated points
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm EI_ABS_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1616 with standard deviation 0.1030.

took 745.98 seconds to run.