# Description
**Budget-Aware Efficient Hybrid Bayesian Optimization with Adaptive Kernel and Exploration-Exploitation Balance (BAEHBBO-AKEEB):** This algorithm builds upon the BAEHBBO framework by incorporating an adaptive kernel for the Gaussian Process Regressor (GPR) and a dynamic exploration-exploitation balance in the acquisition function. The kernel adapts its length scale during the optimization process to better capture the function's characteristics. The exploration-exploitation balance is controlled by a parameter that adjusts the weight of exploration versus exploitation based on the optimization progress.

# Justification
1.  **Adaptive Kernel:** The original BAEHBBO uses a fixed RBF kernel. However, the optimal length scale of the RBF kernel can vary significantly depending on the function being optimized. By allowing the kernel's length scale to adapt during the optimization process, the GPR can better model the function's landscape and improve the accuracy of its predictions. We use a simple heuristic: after a certain number of evaluations, we re-estimate the length scale using the median distance between points.
2.  **Dynamic Exploration-Exploitation Balance:** Balancing exploration and exploitation is crucial in Bayesian Optimization. The original BAEHBBO uses a fixed Expected Improvement (EI) acquisition function. This can lead to premature convergence if the initial samples are not representative of the entire search space. To address this, we introduce a parameter that adjusts the weight of exploration versus exploitation in the EI acquisition function. This parameter is dynamically updated during the optimization process based on the optimization progress. Specifically, we start with a higher exploration bias and gradually shift towards exploitation as the optimization progresses.
3.  **Computational Efficiency:** The adaptive kernel is implemented efficiently by re-estimating the length scale only periodically. The dynamic exploration-exploitation balance is implemented by simply adjusting a parameter in the acquisition function, which has minimal computational overhead.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize

class BAEHBBO_AKEEB:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.exploration_weight = 0.2  # Initial exploration weight

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        samples = sampler.random(n=n_points)
        return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Adaptive kernel length scale estimation
        if len(X) > self.n_init:
            distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
            distances = np.triu(distances, k=1)
            median_distance = np.median(distances[distances > 0])
            length_scale = median_distance
        else:
            length_scale = 1.0  # Initial length scale

        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=length_scale, length_scale_bounds="fixed")
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, gp, y_best):
        mu, sigma = gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)
        imp = mu - y_best
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        # Exploration component (using UCB)
        ucb = mu + self.exploration_weight * sigma

        # Combine EI and UCB
        acquisition = ei + self.exploration_weight * ucb

        return acquisition

    def _select_next_points(self, batch_size, gp, y_best):
        best_x = None
        best_acq = -np.inf
        for _ in range(10 * batch_size):
            x = self._sample_points(1)
            acq = self._acquisition_function(x, gp, y_best)[0]
            if acq > best_acq:
                best_acq = acq
                best_x = x

        return best_x

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)
    
    def _update_eval_points(self, new_X, new_y):
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
        
        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        while self.n_evals < self.budget:
            # Fit the GP model
            gp = self._fit_model(self.X, self.y)
            
            # Determine the batch size adaptively
            remaining_evals = self.budget - self.n_evals
            batch_size = min(max(1, remaining_evals // 5), 5)  # Adaptive batch size
            
            # Select the next points using EI
            next_X = self._select_next_points(batch_size, gp, best_y)
            
            # Evaluate the next points
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Update the best solution
            best_idx = np.argmin(self.y)
            best_y = self.y[best_idx][0]
            best_x = self.X[best_idx]

            # Local search refinement using the surrogate model
            def surrogate_objective(x):
                return gp.predict(x.reshape(1, -1))[0]

            # Limit the number of iterations based on remaining budget
            max_iter = min(5, remaining_evals)  # Limit iterations
            if max_iter > 0:
                res = minimize(surrogate_objective, best_x, method='Nelder-Mead', bounds=list(zip(self.bounds[0], self.bounds[1])), options={'maxiter': max_iter, 'maxfev': max_iter})  # Limit function evaluations
                
                # Evaluate the result of the local search with the real function
                if self.n_evals < self.budget:
                    refined_y = self._evaluate_points(func, res.x.reshape(1, -1))[0][0]
                    if refined_y < best_y:
                        best_y = refined_y
                        best_x = res.x
            
            # Update exploration weight (linear decay)
            self.exploration_weight = max(0.01, self.exploration_weight - 0.01)

        return best_y, best_x
```
## Feedback
 The algorithm BAEHBBO_AKEEB got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1374 with standard deviation 0.1040.

took 41.28 seconds to run.