from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
import warnings
from scipy.stats import trim_mean

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
        self.stagnation_counter = 0
        self.max_stagnation = 10
        self.best_y = np.inf

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        samples = sampler.random(n=n_points)
        return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _estimate_noise(self, X, y, n_neighbors=10, trim_proportion=0.2):
        """Estimates the noise level based on the variance of function values at nearby points using a trimmed mean."""
        n_neighbors = min(n_neighbors, len(X) - 1)
        if len(X) < 2 or n_neighbors < 1:
            return 1e-6  # Return a small default noise if not enough points

        knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree')
        knn.fit(X)
        distances, indices = knn.kneighbors(X)
        
        # Calculate the variance of y values for each point's neighbors
        noise_estimates = np.var(y[indices[:, 1:]], axis=1) # Exclude the point itself
        
        # Return the trimmed mean of noise estimates, scaled to avoid being too small
        noise = max(trim_mean(noise_estimates, trim_proportion), 1e-6)
        return noise

    def _fit_model(self, X, y):
        # Adaptive noise level estimation
        alpha = self._estimate_noise(X, y)
        
        # Kernel selection: Choose between RBF and Matern based on data characteristics
        if len(X) > self.dim + 1:
            try:
                # Fit with RBF kernel
                kernel_rbf = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
                gp_rbf = GaussianProcessRegressor(kernel=kernel_rbf, n_restarts_optimizer=3, alpha=alpha)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gp_rbf.fit(X, y)

                # Fit with Matern kernel
                kernel_matern = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
                gp_matern = GaussianProcessRegressor(kernel=kernel_matern, n_restarts_optimizer=3, alpha=alpha)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gp_matern.fit(X, y)

                # Choose the kernel with the higher log-likelihood
                if gp_rbf.log_marginal_likelihood() > gp_matern.log_marginal_likelihood():
                    gp = gp_rbf
                else:
                    gp = gp_matern
            except:
                kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=alpha)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gp.fit(X, y)
        else:
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=alpha)
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
        self.best_y = best_y

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
            best_y_current = self.y[best_idx][0]
            best_x_current = self.X[best_idx]

            if best_y_current < self.best_y:
                self.best_y = best_y_current
                best_x = best_x_current
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

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
                    if refined_y[best_refined_idx] < self.best_y:
                        self.best_y = refined_y[best_refined_idx]
                        best_x = refined_X[best_refined_idx]
                        self.stagnation_counter = 0
                    
                    # Meta-Local Search
                    meta_max_iter = min(3, remaining_evals) # Further limit iterations
                    if meta_max_iter > 0:
                        # Dynamic step size based on GP's uncertainty
                        _, sigma_meta = gp.predict(best_x.reshape(1, -1), return_std=True)
                        step_size = np.clip(sigma_meta, 1e-3, 0.5)  # Clip step size
                        
                        meta_res = minimize(surrogate_objective, best_x, method='L-BFGS-B', bounds=list(zip(self.bounds[0], self.bounds[1])), options={'maxiter': meta_max_iter, 'maxfun': meta_max_iter})
                        
                        # Evaluate the result of the meta-local search with the real function
                        if self.n_evals < self.budget:
                            meta_x = meta_res.x
                            meta_y = self._evaluate_points(func, meta_x.reshape(1, -1))[0, 0]
                            if meta_y < self.best_y:
                                self.best_y = meta_y
                                best_x = meta_x
                                self.stagnation_counter = 0
            
            # Update exploration weight (adaptive decay based on budget and stagnation)
            budget_fraction = (self.budget - self.n_evals) / self.budget
            self.exploration_weight = max(0.01, min(0.5, 0.5 * budget_fraction * (1 + self.stagnation_counter / self.max_stagnation)))

            if self.stagnation_counter > self.max_stagnation:
                # Restart mechanism
                new_X = self._sample_points(self.n_init)
                new_y = self._evaluate_points(func, new_X)
                self._update_eval_points(new_X, new_y)
                
                best_idx = np.argmin(self.y)
                self.best_y = self.y[best_idx][0]
                best_x = self.X[best_idx]
                self.stagnation_counter = 0

        return self.best_y, best_x
