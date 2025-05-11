# Description
**BAEHBBO-AKEEBLS-EILS-AdaptiveNoise-KernelTuning-Batch-Lookahead-MetaLocalSearch-v2**: Budget-Aware Efficient Hybrid Bayesian Optimization with Adaptive Kernel, Exploration-Exploitation Balance, Local Search Refinement, Stochastic Local Search, Adaptive Noise Handling, Kernel Tuning, Batch Evaluation, Lookahead Acquisition, and Meta-Local Search. This version refines the previous one by incorporating a more robust adaptive exploration strategy and a refined meta-local search. The exploration weight is now dynamically adjusted based on both the budget remaining and the observed improvement rate. The meta-local search is enhanced by considering multiple starting points sampled from the GP's predictive distribution around the current best solution, aiming to escape local optima more effectively. Additionally, the kernel length scale bounds are tightened adaptively based on the data observed so far.

# Justification
The key improvements in this version are:

1.  **Adaptive Exploration Weight:** The exploration weight is crucial for balancing exploration and exploitation. Instead of solely relying on the remaining budget, the exploration weight now also considers the recent improvement rate. If the algorithm is making good progress, the exploration weight is reduced to focus on exploitation. Conversely, if the improvement rate is low, the exploration weight is increased to encourage exploration. This adaptive strategy helps the algorithm to dynamically adjust its search behavior based on the characteristics of the objective function. The improvement rate is calculated based on the change in the best observed function value over a recent window of evaluations.

2.  **Enhanced Meta-Local Search:** The meta-local search is refined to consider multiple starting points. Instead of just starting from the best stochastic local search point, the algorithm now samples a small number of points from the GP's predictive distribution around the current best solution. These samples serve as starting points for multiple L-BFGS-B optimizations. The best solution found among these optimizations is then selected. This multi-start approach helps the meta-local search to escape local optima more effectively and find better solutions.

3. **Adaptive Kernel Length Scale Bounds:** The bounds on the kernel length scale are now tightened adaptively based on the observed data. Specifically, we compute the median distance between data points and use this as a guide for setting the upper bound on the length scale. This helps to prevent the kernel from becoming too smooth or too rough, which can improve the accuracy of the GP model.

4. **Budget Allocation**: The budget allocation for initial sampling, main loop, and local search steps is carefully managed. The initial sampling uses a small fraction of the budget, and the remaining budget is allocated to the main loop and local search steps based on the remaining evaluations.

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
        self.improvement_window = 5 # Window size for improvement rate calculation
        self.previous_best_y = np.inf
        self.improvement_rate = 0.0

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
        
        # Adaptive kernel length scale bounds
        median_distance = np.median(np.linalg.norm(X[1:] - X[:-1], axis=1)) if len(X) > 1 else 1.0
        length_scale_upper_bound = max(1.0, 10 * median_distance)

        # Kernel tuning: Optimize the kernel hyperparameters
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(1e-2, length_scale_upper_bound))  # Allow length scale to vary
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
            current_best_y = self.y[best_idx][0]
            best_x = self.X[best_idx]

            # Calculate improvement rate
            if self.n_evals > self.n_init:
                self.improvement_rate = (self.previous_best_y - current_best_y) / self.previous_best_y
                self.improvement_rate = max(0, self.improvement_rate) # Ensure non-negative

            self.previous_best_y = current_best_y
            best_y = current_best_y

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
                        # Sample around the L-BFGS-B solution, scaling by GP's uncertainty
                        sample = np.random.normal(res.x, sigma_ls, self.dim)
                        sample = np.clip(sample, self.bounds[0], self.bounds[1])  # Clip to bounds
                        refined_X[i, :] = sample
                    
                    refined_y = self._evaluate_points(func, refined_X)[:,0]
                    
                    best_refined_idx = np.argmin(refined_y)
                    if refined_y[best_refined_idx] < best_y:
                        best_y = refined_y[best_refined_idx]
                        best_x = refined_X[best_refined_idx]
                    
                    # Meta-Local Search: Multi-start approach
                    meta_num_starts = min(3, remaining_evals) # Use at most 3 starts
                    meta_best_y = best_y
                    meta_best_x = best_x
                    
                    for _ in range(meta_num_starts):
                        # Sample a starting point from GP's predictive distribution
                        meta_x0 = np.random.normal(best_x, sigma_ls, self.dim)
                        meta_x0 = np.clip(meta_x0, self.bounds[0], self.bounds[1])
                        
                        meta_max_iter = min(3, remaining_evals) # Further limit iterations
                        if meta_max_iter > 0:
                            meta_res = minimize(surrogate_objective, meta_x0, method='L-BFGS-B', bounds=list(zip(self.bounds[0], self.bounds[1])), options={'maxiter': meta_max_iter, 'maxfun': meta_max_iter})
                            
                            # Evaluate the result of the meta-local search with the real function
                            if self.n_evals < self.budget:
                                meta_x = meta_res.x
                                meta_y = self._evaluate_points(func, meta_x.reshape(1, -1))[0, 0]
                                if meta_y < meta_best_y:
                                    meta_best_y = meta_y
                                    meta_best_x = meta_x
                    
                    if meta_best_y < best_y:
                        best_y = meta_best_y
                        best_x = meta_best_x
            
            # Update exploration weight (adaptive decay based on budget and improvement rate)
            budget_fraction = (self.budget - self.n_evals) / self.budget
            self.exploration_weight = max(0.01, min(0.5, 0.5 * budget_fraction * (1 + 2 * (0.1 - self.improvement_rate)))) # Increase exploration if improvement is slow

        return best_y, best_x
```
## Feedback
 The algorithm BAEHBBO_AKEEBLS_EILS_AdaptiveNoise_KernelTuning_Batch_Lookahead_MetaLocalSearch_v2 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1541 with standard deviation 0.1006.

took 41.26 seconds to run.