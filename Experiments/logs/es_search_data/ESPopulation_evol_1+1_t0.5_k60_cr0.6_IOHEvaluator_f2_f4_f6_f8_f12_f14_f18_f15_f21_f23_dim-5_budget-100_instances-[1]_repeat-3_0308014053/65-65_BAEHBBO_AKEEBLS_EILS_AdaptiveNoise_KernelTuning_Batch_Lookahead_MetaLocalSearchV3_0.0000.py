from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
import warnings
from scipy.linalg import solve

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
        
        # Calculate the variance of y values for each point's neighbors, weighted by distance
        noise_estimates = []
        for i in range(len(X)):
            neighbor_ys = y[indices[i, 1:]]
            neighbor_distances = distances[i, 1:]
            
            # Weight inversely by distance, adding a small constant to avoid division by zero
            weights = 1.0 / (neighbor_distances + 1e-6)
            weights /= np.sum(weights)  # Normalize weights
            
            weighted_variance = np.sum(weights * (neighbor_ys - np.average(neighbor_ys, weights=weights))**2)
            noise_estimates.append(weighted_variance)
        
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

    def _acquisition_function(self, X, gp, y_best, exploration_weight):
        mu, sigma = gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)
        imp = mu - y_best
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        # Exploration component (using UCB)
        ucb = mu + exploration_weight * sigma

        # Combine EI and UCB
        acquisition = ei + exploration_weight * ucb

        return acquisition

    def _select_next_points(self, batch_size, gp, y_best, exploration_weight):
        selected_X = []
        for _ in range(batch_size):
            best_x = None
            best_acq = -np.inf
            for _ in range(10 * batch_size):
                x = self._sample_points(1)
                acq = self._acquisition_function(x, gp, y_best, exploration_weight)
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
            
            # Adaptive exploration weight
            mu_global, sigma_global = gp.predict(self.X, return_std=True)
            avg_sigma = np.mean(sigma_global)  # Average uncertainty across all points
            budget_fraction = (self.budget - self.n_evals) / self.budget
            exploration_weight = max(0.01, min(0.5, avg_sigma * budget_fraction))  # Dynamic exploration weight

            # Select the next points using EI
            next_X = self._select_next_points(batch_size, gp, best_y, exploration_weight)
            
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
                    mu_ls, sigma_ls = gp.predict(res.x.reshape(1, -1), return_std=True)
                    sigma_ls = np.clip(sigma_ls, 1e-9, np.inf)
                    
                    # Adaptive number of samples based on GP's uncertainty
                    num_samples = min(int(5 * sigma_ls + 1), remaining_evals)  # More samples if uncertain

                    refined_X = np.zeros((num_samples, self.dim))
                    refined_y = np.zeros(num_samples)
                    
                    # Compute the gradient of the surrogate model
                    dmu_dx = self._gradient(gp, res.x)

                    for i in range(num_samples):
                        # Sample around the L-BFGS-B solution, scaling by GP's uncertainty and biased by the gradient
                        # Biasing towards negative gradient
                        sample = np.random.normal(res.x - 0.1 * dmu_dx, sigma_ls, self.dim)
                        sample = np.clip(sample, self.bounds[0], self.bounds[1])  # Clip to bounds
                        refined_X[i, :] = sample
                    
                    refined_y = self._evaluate_points(func, refined_X)[:,0]
                    
                    best_refined_idx = np.argmin(refined_y)
                    if refined_y[best_refined_idx] < best_y:
                        best_y = refined_y[best_refined_idx]
                        best_x = refined_X[best_refined_idx]
                    
                    # Meta-Local Search
                    meta_max_iter = min(3, remaining_evals) # Further limit iterations
                    if meta_max_iter > 0:
                        meta_res = minimize(surrogate_objective, best_x, method='L-BFGS-B', bounds=list(zip(self.bounds[0], self.bounds[1])), options={'maxiter': meta_max_iter, 'maxfun': meta_max_iter})
                        
                        # Evaluate the result of the meta-local search with the real function
                        if self.n_evals < self.budget:
                            meta_x = meta_res.x
                            meta_y = self._evaluate_points(func, meta_x.reshape(1, -1))[0, 0]
                            if meta_y < best_y:
                                best_y = meta_y
                                best_x = meta_x
            

        return best_y, best_x

    def _gradient(self, gp, x):
        """Calculates the gradient of the GP's mean function at a given point x."""
        x = x.reshape(1, -1)  # Ensure x is a row vector

        # Get the kernel and its parameters
        kernel = gp.kernel_
        K = kernel(self.X, x)  # Covariance between training points and x
        alpha = gp.alpha_  # Dual coefficients

        # Calculate the gradient
        dmu_dx = alpha.T @ gp.kernel_.gradient_x(x, self.X).reshape(self.dim, -1)

        return dmu_dx.flatten()
