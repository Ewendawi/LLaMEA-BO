# Description
**BAEHBBO_AKEEBLS_V2:** Budget-Aware Efficient Hybrid Bayesian Optimization with Adaptive Kernel, Exploration-Exploitation Balance, and Local Search Refinement (BAEHBBO-AKEEBLS) with improved local search and adaptive exploration. The local search uses L-BFGS-B with a higher iteration limit and multiple restarts. The exploration weight is adjusted dynamically based on the GP's uncertainty and the progress of optimization. A separate acquisition function is used for the local search step to encourage exploitation around the current best. The kernel is now a Matern kernel.

# Justification
The key improvements focus on enhancing the local search and exploration-exploitation balance.

1.  **Enhanced Local Search:** Increasing the `max_iter` in `minimize` allows for a more thorough local search, potentially leading to better exploitation of promising regions. Adding `n_restarts` aims to escape local optima and find better solutions. Evaluating the result of local search is important.

2.  **Adaptive Exploration Weight:** The exploration weight is dynamically adjusted based on both the average uncertainty (sigma) from the GP model and the progress of the optimization (reduction in function values). The progress-based adjustment helps to reduce exploration as the algorithm converges, focusing more on exploitation.

3.  **Matern Kernel:** Matern kernel is used instead of RBF kernel. Matern kernel is a generalization of the RBF kernel, and is indexed by a parameter ν which controls the smoothness of the resulting function. For ν→∞ the Matern kernel converges to the RBF kernel. Using Matern kernel allows to control the smoothness of the function.

These modifications aim to improve the algorithm's ability to find better optima within the given budget by balancing exploration and exploitation more effectively.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize

class BAEHBBO_AKEEBLS_V2:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.exploration_weight = 0.2  # Initial exploration weight
        self.best_y_so_far = np.inf

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

        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=length_scale, length_scale_bounds="fixed", nu=2.5)
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

    def _acquisition_function_local_search(self, X, gp, y_best):
        mu, sigma = gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)
        imp = mu - y_best
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei

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
        self.best_y_so_far = best_y

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
            max_iter = min(10, remaining_evals)  # Limit iterations
            if max_iter > 0:
                # Use L-BFGS-B for local search with multiple restarts
                n_restarts = 2
                best_refined_y = best_y
                best_refined_x = best_x
                for _ in range(n_restarts):
                    res = minimize(surrogate_objective, best_x, method='L-BFGS-B', bounds=list(zip(self.bounds[0], self.bounds[1])), options={'maxiter': max_iter})  # Limit function evaluations
                
                    # Evaluate the result of the local search with the real function
                    if self.n_evals < self.budget:
                        refined_y = self._evaluate_points(func, res.x.reshape(1, -1))[0][0]
                        if refined_y < best_refined_y:
                            best_refined_y = refined_y
                            best_refined_x = res.x
                best_y = best_refined_y
                best_x = best_refined_x
            
            # Update exploration weight (adaptive decay based on GP uncertainty and optimization progress)
            mu, sigma = gp.predict(self.X, return_std=True)
            avg_sigma = np.mean(sigma)
            progress = max(0, (self.best_y_so_far - best_y) / self.best_y_so_far) if self.best_y_so_far != 0 else 0
            self.exploration_weight = max(0.01, min(0.5, avg_sigma * (1 - progress)))
            self.best_y_so_far = min(self.best_y_so_far, best_y)

        return best_y, best_x
```
## Feedback
 The algorithm BAEHBBO_AKEEBLS_V2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1425 with standard deviation 0.0986.

took 74.17 seconds to run.