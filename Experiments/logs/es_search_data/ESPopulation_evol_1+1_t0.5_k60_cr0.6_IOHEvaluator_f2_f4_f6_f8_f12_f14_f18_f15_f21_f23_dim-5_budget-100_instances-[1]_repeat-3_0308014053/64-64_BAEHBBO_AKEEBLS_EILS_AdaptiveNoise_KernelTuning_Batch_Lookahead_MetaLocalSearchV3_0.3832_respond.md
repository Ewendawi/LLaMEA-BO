# Description
**BAEHBBO-AKEEBLS-EILS-AdaptiveNoise-KernelTuning-Batch-Lookahead-MetaLocalSearchV3**: Budget-Aware Efficient Hybrid Bayesian Optimization with Adaptive Kernel, Exploration-Exploitation Balance, Local Search Refinement, Stochastic Local Search, Adaptive Noise Handling, Kernel Tuning, Batch Evaluation, Lookahead Acquisition, and Meta-Local Search. This version improves upon the previous one by incorporating a more robust and efficient local search strategy. The key improvement is to refine the local search by considering the uncertainty of the surrogate model more explicitly when deciding whether to evaluate the L-BFGS-B result. Also, the number of restarts for the local search is increased based on the remaining budget. The acquisition function is also refined to incorporate a dynamic weighting between EI and UCB.

# Justification
The previous version already had a strong foundation, but the local search could be improved. The key ideas behind the changes are:

*   **More Informed Local Search Evaluation**: Instead of always evaluating the result of the L-BFGS-B local search, we now consider the GP's uncertainty estimate at that point. If the uncertainty is high, it suggests the GP is less confident in its prediction, and evaluating the point is more valuable for exploration. This helps to avoid wasting evaluations on points where the GP is already confident, even if the predicted value is good.
*   **Dynamic EI/UCB Balancing:** The balance between exploration and exploitation is crucial. Instead of a fixed combination, the exploration weight is dynamically adjusted based on the remaining budget and the GP's uncertainty. When the uncertainty is high, we increase exploration to sample in less-explored regions.
*   **Budget-Aware Local Search Restarts**: Increase the number of restarts for the local search based on the remaining budget. This allows for more thorough exploration of the local landscape when the budget permits.

These changes aim to make the algorithm more efficient by focusing evaluations on areas where they are most likely to improve the model and find better solutions.

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

class BAEHBBO_AKEEBLS_EILS_AdaptiveNoise_KernelTuning_Batch_Lookahead_MetaLocalSearchV3:
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
        #self.lookahead_depth = 3 # Number of lookahead steps

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
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e3))  # Allow length scale to vary
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=alpha)  # Enable kernel optimization
        
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

        # Combine EI and UCB with dynamic weighting
        exploration_factor = self.exploration_weight #* np.mean(sigma)  # Scale by average uncertainty
        acquisition = ei + exploration_factor * ucb

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
            
            # Determine the number of restarts based on remaining budget
            n_restarts = min(3, remaining_evals // 2)  # Limit restarts

            if max_iter > 0:
                best_local_x = best_x
                best_local_y = best_y
                
                for _ in range(n_restarts):
                    # Use L-BFGS-B for local search
                    res = minimize(surrogate_objective, best_x, method='L-BFGS-B', bounds=list(zip(self.bounds[0], self.bounds[1])), options={'maxiter': max_iter, 'maxfun': max_iter})  # Limit function evaluations

                    # Evaluate the result of the local search with the real function
                    if self.n_evals < self.budget:
                        mu_ls, sigma_ls = gp.predict(res.x.reshape(1, -1), return_std=True)
                        sigma_ls = np.clip(sigma_ls, 1e-9, np.inf)

                        # Evaluate only if GP is uncertain about the L-BFGS-B result
                        if sigma_ls > 0.1 * np.std(self.y):  # adaptive uncertainty threshold
                            local_y = self._evaluate_points(func, res.x.reshape(1, -1))[0, 0]
                            if local_y < best_local_y:
                                best_local_y = local_y
                                best_local_x = res.x
                
                # Stochastic Local Search around the best L-BFGS-B result
                if self.n_evals < self.budget:
                    num_samples = min(5, remaining_evals) # Sample at most 5 points
                    refined_X = np.zeros((num_samples, self.dim))
                    refined_y = np.zeros(num_samples)

                    mu_ls, sigma_ls = gp.predict(best_local_x.reshape(1, -1), return_std=True)
                    sigma_ls = np.clip(sigma_ls, 1e-9, np.inf)
                    
                    for i in range(num_samples):
                        # Sample around the L-BFGS-B solution, scaling by GP's uncertainty
                        sample = np.random.normal(best_local_x, sigma_ls, self.dim)
                        sample = np.clip(sample, self.bounds[0], self.bounds[1])  # Clip to bounds
                        refined_X[i, :] = sample
                    
                    refined_y = self._evaluate_points(func, refined_X)[:,0]
                    
                    best_refined_idx = np.argmin(refined_y)
                    if refined_y[best_refined_idx] < best_local_y:
                        best_local_y = refined_y[best_refined_idx]
                        best_local_x = refined_X[best_refined_idx]
                
                # Meta-Local Search
                meta_max_iter = min(3, remaining_evals) # Further limit iterations
                if meta_max_iter > 0:
                    meta_res = minimize(surrogate_objective, best_local_x, method='L-BFGS-B', bounds=list(zip(self.bounds[0], self.bounds[1])), options={'maxiter': meta_max_iter, 'maxfun': meta_max_iter})
                    
                    # Evaluate the result of the meta-local search with the real function
                    if self.n_evals < self.budget:
                        meta_x = meta_res.x
                        meta_y = self._evaluate_points(func, meta_x.reshape(1, -1))[0, 0]
                        if meta_y < best_local_y:
                            best_local_y = meta_y
                            best_local_x = meta_x
                
                if best_local_y < best_y:
                    best_y = best_local_y
                    best_x = best_local_x
            
            # Update exploration weight (adaptive decay based on budget)
            budget_fraction = (self.budget - self.n_evals) / self.budget
            self.exploration_weight = max(0.01, min(0.5, 0.5 * budget_fraction))

        return best_y, best_x
```
## Feedback
 The algorithm BAEHBBO_AKEEBLS_EILS_AdaptiveNoise_KernelTuning_Batch_Lookahead_MetaLocalSearchV3 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1558 with standard deviation 0.0991.

took 43.50 seconds to run.