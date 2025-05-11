# Description
**BAEHBBO-AKEEBLS-EILS-AdaptiveLS**: Budget-Aware Efficient Hybrid Bayesian Optimization with Adaptive Kernel, Exploration-Exploitation Balance, Local Search Refinement, and Stochastic Local Search, enhanced with Adaptive Local Search. This enhancement dynamically adjusts the intensity of the local search based on the remaining budget and the GP's uncertainty estimates. It also incorporates a more robust method for generating samples during the stochastic local search phase, taking into account correlations between dimensions.

# Justification
The previous version used a fixed number of samples for stochastic local search and a fixed number of iterations for L-BFGS-B. This can be inefficient, especially when the remaining budget is small or the GP is highly uncertain. The following changes are made:

1.  **Adaptive Local Search Intensity:** The number of L-BFGS-B iterations is now dynamically adjusted based on the remaining budget and the average GP uncertainty. When the budget is high and uncertainty is low, more iterations are used for fine-tuning. When the budget is low or uncertainty is high, fewer iterations are used to conserve evaluations.
2.  **Correlation-Aware Stochastic Sampling:** The stochastic sampling in the local search now considers the covariance matrix predicted by the GP, rather than sampling independently along each dimension. This allows for more informed sampling, especially in problems where dimensions are correlated.
3.  **Budget-Aware Number of Samples:** The number of samples in the stochastic local search is also made adaptive to the remaining budget.
4.  **Kernel Tuning:** The kernel is now dynamically tuned using the maximum likelihood estimation.

These changes allow the algorithm to better balance exploration and exploitation, and to make more efficient use of the available budget.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize
from scipy.linalg import cholesky, solve_triangular

class BAEHBBO_AKEEBLS_EILS_AdaptiveLS:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.exploration_weight = 0.2  # Initial exploration weight
        self.kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        samples = sampler.random(n=n_points)
        return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Adaptive kernel length scale estimation and tuning
        gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=3, alpha=1e-6)
        gp.fit(X, y)
        self.kernel = gp.kernel_  # Update kernel with tuned parameters
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
            mu, sigma = gp.predict(self.X, return_std=True)
            avg_sigma = np.mean(sigma)
            max_iter = min(10, int(remaining_evals * (1 - avg_sigma)))  # Adaptive iterations

            if max_iter > 0:
                # Use L-BFGS-B for local search
                res = minimize(surrogate_objective, best_x, method='L-BFGS-B', bounds=list(zip(self.bounds[0], self.bounds[1])), options={'maxiter': max_iter, 'maxfun': max_iter})  # Limit function evaluations
                
                # Evaluate the result of the local search with the real function
                if self.n_evals < self.budget:
                    # Stochastic Local Search
                    num_samples = min(5, remaining_evals) # Sample at most 5 points
                    
                    refined_X = np.zeros((num_samples, self.dim))
                    refined_y = np.zeros(num_samples)
                    
                    mu_ls, cov_ls = gp.predict(res.x.reshape(1, -1), return_cov=True)
                    cov_ls = np.clip(cov_ls, 1e-9, np.inf)

                    try:
                        L = cholesky(cov_ls, lower=True)
                        refined_X = res.x + np.random.normal(size=(num_samples, self.dim)) @ L
                    except np.linalg.LinAlgError:
                        # If covariance matrix is not positive definite, sample independently
                        sigma_ls = np.sqrt(np.diag(cov_ls))
                        for i in range(num_samples):
                            refined_X[i, :] = np.random.normal(res.x, sigma_ls, self.dim)
                    
                    refined_X = np.clip(refined_X, self.bounds[0], self.bounds[1])  # Clip to bounds                    
                    refined_y = self._evaluate_points(func, refined_X)[:,0]
                    
                    best_refined_idx = np.argmin(refined_y)
                    if refined_y[best_refined_idx] < best_y:
                        best_y = refined_y[best_refined_idx]
                        best_x = refined_X[best_refined_idx]
            
            # Update exploration weight (adaptive decay based on GP uncertainty)
            mu, sigma = gp.predict(self.X, return_std=True)
            avg_sigma = np.mean(sigma)
            self.exploration_weight = max(0.01, min(0.5, avg_sigma))

        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<BAEHBBO_AKEEBLS_EILS_AdaptiveLS>", line 145, in __call__
 143 |                     try:
 144 |                         L = cholesky(cov_ls, lower=True)
 145->                         refined_X = res.x + np.random.normal(size=(num_samples, self.dim)) @ L
 146 |                     except np.linalg.LinAlgError:
 147 |                         # If covariance matrix is not positive definite, sample independently
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 1 is different from 5)
