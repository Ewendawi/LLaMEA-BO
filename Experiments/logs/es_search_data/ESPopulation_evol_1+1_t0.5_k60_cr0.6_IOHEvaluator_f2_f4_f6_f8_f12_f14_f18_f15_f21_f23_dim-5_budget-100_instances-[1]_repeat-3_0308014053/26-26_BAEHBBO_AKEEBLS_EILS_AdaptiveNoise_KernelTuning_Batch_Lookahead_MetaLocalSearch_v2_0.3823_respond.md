# Description
**BAEHBBO-AKEEBLS-EILS-AdaptiveNoise-KernelTuning-Batch-Lookahead-MetaLocalSearch-v2**: Budget-Aware Efficient Hybrid Bayesian Optimization with Adaptive Kernel, Exploration-Exploitation Balance, Local Search Refinement, Stochastic Local Search, Adaptive Noise Handling, Kernel Tuning, Batch Evaluation, Lookahead Acquisition, and Meta-Local Search. This version enhances the previous one by improving the stochastic local search and meta-local search phases. It introduces a more robust sampling strategy around the L-BFGS-B solution in the stochastic local search, using a combination of GP uncertainty and a fixed exploration radius. The meta-local search is also refined by increasing the number of iterations when budget allows, and by re-evaluating the L-BFGS-B solution with the true objective function *before* comparing it to the current best. This ensures that the comparison is always against true function values, preventing the surrogate model from misleading the search. Finally, the adaptive exploration weight decay is made more aggressive to favor exploitation in the later stages of the budget.

# Justification
The key improvements target the local search phases, which are crucial for refining the solutions found by the global Bayesian optimization.

1.  **Improved Stochastic Local Search:** Instead of directly sampling from a normal distribution with standard deviation from the GP, the new stochastic local search samples within a defined radius of the L-BFGS-B solution. This radius is determined by a combination of the GP's uncertainty and a fixed exploration radius. The fixed exploration radius allows for exploration beyond what the GP uncertainty suggests, which can be beneficial in escaping local optima of the surrogate model.

2.  **Re-evaluate L-BFGS-B Solution in Meta-Local Search:** Before comparing the result of the meta-local search with the current best, the L-BFGS-B solution is evaluated using the true objective function. This ensures that the comparison is based on true function values and prevents the surrogate model from biasing the search.

3.  **More Aggressive Exploration Decay:** The exploration weight is decayed more aggressively to favor exploitation in the later stages of the budget. This is achieved by squaring the budget fraction, which causes the exploration weight to decrease more rapidly as the budget is consumed. This is based on the assumption that the GP model becomes more accurate as more data is collected, so the algorithm can rely more on the GP's predictions and less on exploration.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
import warnings

class BAEHBBO_AKEEBLS_EILS_AdaptiveNoise_KernelTuning_Batch_Lookahead_MetaLocalSearch_v2:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.exploration_weight = 0.2  # Initial exploration weight
        self.initial_exploration_weight = 0.2
        self.ls_exploration_radius = 0.5 # Fixed exploration radius for local search

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        samples = sampler.random(n=n_points)
        return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _estimate_noise(self, X, y, n_neighbors=5):
        """Estimates the noise level based on the variance of function values at nearby points."""
        n_neighbors = min(n_neighbors, len(X) - 1)
        if len(X) < 2 or n_neighbors < 1:
            return 1e-6  # Return a small default noise if not enough points

        knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree')
        knn.fit(X)
        distances, indices = knn.kneighbors(X)
        
        # Calculate the variance of y values for each point's neighbors
        noise_estimates = np.var(y[indices[:, 1:]], axis=1) # Exclude the point itself
        
        # Return the median noise estimate, scaled to avoid being too small
        noise = np.median(noise_estimates)
        return max(noise, 1e-6)

    def _fit_model(self, X, y):
        # Adaptive noise level estimation
        alpha = self._estimate_noise(X, y)
        
        # Kernel tuning: Optimize the kernel hyperparameters
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))  # Allow length scale to vary
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=alpha)  # Enable kernel optimization
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
        selected_X = []
        for _ in range(batch_size):
            best_x = None
            best_acq = -np.inf
            for _ in range(10 * batch_size):
                x = self._sample_points(1)
                acq = self._acquisition_function(x, gp, y_best)
                if acq > best_acq:
                    best_acq = acq
                    best_x = x

            selected_X.append(best_x.flatten())
        return np.array(selected_X)

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
                # Use L-BFGS-B for local search
                res = minimize(surrogate_objective, best_x, method='L-BFGS-B', bounds=list(zip(self.bounds[0], self.bounds[1])), options={'maxiter': max_iter, 'maxfun': max_iter})  # Limit function evaluations
                
                # Evaluate the result of the local search with the real function
                if self.n_evals < self.budget:
                    # Stochastic Local Search
                    num_samples = min(3, remaining_evals) # Sample at most 3 points
                    
                    refined_X = np.zeros((num_samples, self.dim))
                    refined_y = np.zeros(num_samples)
                    
                    mu_ls, sigma_ls = gp.predict(res.x.reshape(1, -1), return_std=True)
                    sigma_ls = np.clip(sigma_ls, 1e-9, np.inf)
                    
                    for i in range(num_samples):
                        # Sample around the L-BFGS-B solution, scaling by GP's uncertainty and a fixed radius
                        # The radius is the maximum of GP uncertainty and a fixed exploration radius
                        radius = max(sigma_ls, self.ls_exploration_radius)
                        sample = np.random.normal(res.x, radius, self.dim)
                        sample = np.clip(sample, self.bounds[0], self.bounds[1])  # Clip to bounds
                        refined_X[i, :] = sample
                    
                    refined_y = self._evaluate_points(func, refined_X)[:,0]
                    
                    best_refined_idx = np.argmin(refined_y)
                    if refined_y[best_refined_idx] < best_y:
                        best_y = refined_y[best_refined_idx]
                        best_x = refined_X[best_refined_idx]
                    
                    # Meta-Local Search
                    meta_max_iter = min(5, remaining_evals) # Further limit iterations
                    if meta_max_iter > 0:
                        meta_res = minimize(surrogate_objective, best_x, method='L-BFGS-B', bounds=list(zip(self.bounds[0], self.bounds[1])), options={'maxiter': meta_max_iter, 'maxfun': meta_max_iter})
                        
                        # Evaluate the result of the meta-local search with the real function
                        if self.n_evals < self.budget:
                            meta_x = meta_res.x
                            meta_y = self._evaluate_points(func, meta_x.reshape(1, -1))[0, 0]
                            if meta_y < best_y:
                                best_y = meta_y
                                best_x = meta_x
            
            # Update exploration weight (adaptive decay based on budget)
            budget_fraction = (self.budget - self.n_evals) / self.budget
            self.exploration_weight = max(0.01, min(0.5, 0.5 * budget_fraction**2))

        return best_y, best_x
```
## Feedback
 The algorithm BAEHBBO_AKEEBLS_EILS_AdaptiveNoise_KernelTuning_Batch_Lookahead_MetaLocalSearch_v2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1549 with standard deviation 0.0943.

took 50.67 seconds to run.