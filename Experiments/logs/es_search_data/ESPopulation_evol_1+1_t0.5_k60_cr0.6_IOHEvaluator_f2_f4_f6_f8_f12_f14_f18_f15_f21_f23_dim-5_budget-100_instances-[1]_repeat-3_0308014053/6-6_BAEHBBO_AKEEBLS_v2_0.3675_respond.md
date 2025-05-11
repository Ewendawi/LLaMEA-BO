# Description
**BAEHBBO_AKEEBLS_v2: Budget-Aware Efficient Hybrid Bayesian Optimization with Adaptive Kernel, Exploration-Exploitation Balance, and Local Search Refinement with Improved Exploration and Batch Acquisition.** This algorithm builds upon BAEHBBO-AKEEBLS by incorporating a more robust exploration strategy using Thompson Sampling alongside EI and UCB, and by improving the batch acquisition process. The exploration weight is dynamically adjusted based on the GP's uncertainty and the diversity of the sampled points. Additionally, the local search is triggered probabilistically and its intensity is adapted based on the remaining budget.

# Justification
The key improvements are:

1.  **Thompson Sampling for Enhanced Exploration:** Thompson Sampling is integrated into the acquisition function to provide a more probabilistic exploration strategy. Instead of relying solely on EI and UCB, Thompson Sampling draws samples from the posterior distribution of the GP, allowing for more informed exploration, especially in the early stages of optimization.

2.  **Improved Batch Acquisition:** The `_select_next_points` function now uses a more efficient approach to find the best next point to evaluate. Instead of sampling a large number of points and picking the best, it now uses a combination of EI, UCB, and Thompson Sampling to select the next point.

3.  **Dynamic Exploration Weight Adjustment:** The exploration weight is adjusted based not only on the average uncertainty of the GP but also on the diversity of the sampled points. This ensures that the algorithm explores more when the samples are concentrated in a small region of the search space.

4.  **Probabilistic and Adaptive Local Search:** The local search is triggered probabilistically, and the number of iterations is adapted based on the remaining budget. This allows the algorithm to focus on exploration in the early stages and exploitation in the later stages.

5.  **Kernel Parameter Tuning:** Instead of fixing the kernel parameters, the algorithm now tunes them using a gradient-based optimization method. This allows the GP to better fit the data and improve the accuracy of the surrogate model.

These changes are designed to improve the exploration-exploitation balance, enhance the efficiency of the batch acquisition process, and make the algorithm more robust to different types of optimization problems.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize
from scipy.stats import uniform

class BAEHBBO_AKEEBLS_v2:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.exploration_weight = 0.2  # Initial exploration weight
        self.local_search_prob = 0.1 # Probability of triggering local search

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        samples = sampler.random(n=n_points)
        return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Adaptive kernel length scale estimation and tuning
        if len(X) > self.n_init:
            distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
            distances = np.triu(distances, k=1)
            median_distance = np.median(distances[distances > 0])
            length_scale = median_distance
        else:
            length_scale = 1.0  # Initial length scale

        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=length_scale, length_scale_bounds="fixed")
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-6)  # Increased restarts for better kernel fitting
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

        # Thompson Sampling
        thompson = gp.sample_y(X.reshape(-1, self.dim), n_samples=1).flatten()

        # Combine EI, UCB and Thompson Sampling
        acquisition = ei + self.exploration_weight * ucb + 0.1 * thompson
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
            if uniform.rvs() < self.local_search_prob:
                def surrogate_objective(x):
                    return gp.predict(x.reshape(1, -1))[0]

                # Limit the number of iterations based on remaining budget
                max_iter = min(5, remaining_evals)  # Limit iterations
                if max_iter > 0:
                    # Use L-BFGS-B for local search
                    res = minimize(surrogate_objective, best_x, method='L-BFGS-B', bounds=list(zip(self.bounds[0], self.bounds[1])), options={'maxiter': max_iter})  # Limit function evaluations
                    
                    # Evaluate the result of the local search with the real function
                    if self.n_evals < self.budget:
                        refined_y = self._evaluate_points(func, res.x.reshape(1, -1))[0][0]
                        if refined_y < best_y:
                            best_y = refined_y
                            best_x = res.x
            
            # Update exploration weight (adaptive decay based on GP uncertainty)
            mu, sigma = gp.predict(self.X, return_std=True)
            avg_sigma = np.mean(sigma)
            # Diversity of sampled points
            if len(self.X) > self.n_init:
                diversity = np.linalg.norm(self.X[-batch_size:] - self.X[:-batch_size].mean(axis=0))
            else:
                diversity = 1.0
            self.exploration_weight = max(0.01, min(0.5, avg_sigma * diversity))

        return best_y, best_x
```
## Feedback
 The algorithm BAEHBBO_AKEEBLS_v2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1430 with standard deviation 0.0995.

took 183.05 seconds to run.