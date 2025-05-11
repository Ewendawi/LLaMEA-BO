from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize

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
        self.best_y_history = []
        self.best_x = None

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
        self.best_x = best_x
        self.best_y_history.append(best_y)

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
                # Sample points around the current best for local search initialization
                local_search_points = self._sample_points(3) * 0.1 + best_x  # Sample around best_x
                local_search_points = np.clip(local_search_points, self.bounds[0], self.bounds[1])

                best_refined_y = best_y
                best_refined_x = best_x

                for x0 in local_search_points:
                    # Use L-BFGS-B for local search
                    res = minimize(surrogate_objective, x0, method='L-BFGS-B', bounds=list(zip(self.bounds[0], self.bounds[1])), options={'maxiter': max_iter})  # Limit function evaluations
                    
                    # Evaluate the result of the local search with the real function
                    if self.n_evals < self.budget:
                        refined_y = self._evaluate_points(func, res.x.reshape(1, -1))[0][0]
                        if refined_y < best_refined_y:
                            best_refined_y = refined_y
                            best_refined_x = res.x
                
                if best_refined_y < best_y:
                    best_y = best_refined_y
                    best_x = best_refined_x
            
            # Update exploration weight (adaptive decay based on GP uncertainty)
            mu, sigma = gp.predict(self.X, return_std=True)
            avg_sigma = np.mean(sigma)

            # Adjust exploration based on improvement rate
            if len(self.best_y_history) > 5:
                improvement_rate = (self.best_y_history[-5] - best_y) / 5
                if improvement_rate < 1e-3:  # If improvement is slow
                    self.exploration_weight = min(0.5, self.exploration_weight * 1.2)  # Increase exploration
                else:
                    self.exploration_weight = max(0.01, self.exploration_weight * 0.8)  # Decrease exploration
            
            self.exploration_weight = max(0.01, min(0.5, avg_sigma * self.exploration_weight))
            self.best_y_history.append(best_y)
            self.best_x = best_x

        return best_y, best_x
