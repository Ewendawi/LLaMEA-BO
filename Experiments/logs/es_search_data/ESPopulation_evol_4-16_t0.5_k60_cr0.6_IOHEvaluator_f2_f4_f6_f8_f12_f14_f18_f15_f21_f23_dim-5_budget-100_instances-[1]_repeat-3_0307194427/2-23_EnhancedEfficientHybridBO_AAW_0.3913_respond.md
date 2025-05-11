# Description
**EnhancedEfficientHybridBO with Adaptive Acquisition Weighting (EEHBBO-AAW)**: This algorithm improves upon the EnhancedEfficientHybridBO by introducing an adaptive weighting mechanism to the Expected Improvement (EI) acquisition function. This weighting balances exploration and exploitation more effectively by considering both the uncertainty (sigma) and the improvement potential (gamma) in a dynamic manner. Specifically, a weight is calculated based on the ratio of the average EI to the average uncertainty. This weight is then applied to the EI, encouraging exploration when uncertainty is high and exploitation when potential improvements are significant. The kernel optimization is also adjusted to use a larger number of restarts to find a better kernel.

# Justification
The core idea is to dynamically adjust the exploration-exploitation balance. The original EEHBBO uses a fixed EI acquisition function, which might not be optimal across different stages of the optimization or for different functions. By adaptively weighting the EI, the algorithm can prioritize exploration when the model is uncertain and exploitation when promising regions are identified. The adaptive weight is calculated based on the average EI and the average uncertainty, which provides a measure of the overall exploration-exploitation trade-off. The increase in kernel optimization restarts will help the algorithm to better fit the GP model to the data.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

class EnhancedEfficientHybridBO_AAW:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * (dim + 1)
        self.length_scale = 1.0
        self.kernel_optim_interval = 5

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
        kernel = C(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.length_scale, length_scale_bounds=(1e-2, 10))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-5)
        gp.fit(X, y)
        return gp

    def _optimize_kernel(self):
        def obj(length_scale):
            kernel = C(1.0, constant_value_bounds="fixed") * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 10))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-5)
            gp.fit(self.X, self.y)
            return -gp.log_marginal_likelihood()

        res = minimize(obj, x0=self.length_scale, method='L-BFGS-B', bounds=[(1e-2, 10)])
        self.length_scale = res.x[0]

    def _acquisition_function(self, X, gp, y_best):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)
        gamma = (y_best - mu) / sigma
        ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))

        # Adaptive weighting
        avg_ei = np.mean(ei)
        avg_sigma = np.mean(sigma)
        if avg_sigma > 0:
            weight = avg_ei / avg_sigma
        else:
            weight = 1.0  # Avoid division by zero

        weighted_ei = weight * ei
        return weighted_ei

    def _select_next_points(self, gp, y_best, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        candidates = self._sample_points(100 * self.dim)
        ei = self._acquisition_function(candidates, gp, y_best)
        
        # Select the top batch_size candidates based on EI
        selected_indices = np.argsort(ei)[-batch_size:]
        selected_points = candidates[selected_indices]

        # Ensure diversity by penalizing points that are too close to existing points
        if self.X is not None:
            distances = cdist(selected_points, self.X)
            min_distances = np.min(distances, axis=1)
            # Only select points that are sufficiently far away from existing points
            selected_points = selected_points[min_distances > 0.1]
            if len(selected_points) < batch_size:
              remaining_needed = batch_size - len(selected_points)
              additional_indices = np.argsort(ei)[:-batch_size-1:-1][:remaining_needed]
              additional_points = candidates[additional_indices]
              selected_points = np.concatenate([selected_points, additional_points], axis=0)

        return selected_points[:batch_size]

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
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        
        iteration = 0
        while self.n_evals < self.budget:
            # Fit the Gaussian Process model
            gp = self._fit_model(self.X, self.y)

            # Optimize kernel lengthscale periodically
            if iteration % self.kernel_optim_interval == 0:
                self._optimize_kernel()
                gp = self._fit_model(self.X, self.y)

            # Dynamic batch size
            _, sigma = gp.predict(self.X, return_std=True)
            avg_sigma = np.mean(sigma)
            batch_size = min(self.budget - self.n_evals, int(np.ceil(5 * (avg_sigma / np.std(self.bounds))))) if np.std(self.bounds) > 0 else min(self.budget - self.n_evals, 5)
            batch_size = max(1, batch_size)

            # Select the next points to evaluate
            next_X = self._select_next_points(gp, best_y, batch_size)

            # Evaluate the points
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Update the best solution
            best_idx = np.argmin(self.y)
            best_y = self.y[best_idx][0]
            best_x = self.X[best_idx]

            iteration += 1

        return best_y, best_x
```
## Feedback
 The algorithm EnhancedEfficientHybridBO_AAW got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1682 with standard deviation 0.1028.

took 463.93 seconds to run.