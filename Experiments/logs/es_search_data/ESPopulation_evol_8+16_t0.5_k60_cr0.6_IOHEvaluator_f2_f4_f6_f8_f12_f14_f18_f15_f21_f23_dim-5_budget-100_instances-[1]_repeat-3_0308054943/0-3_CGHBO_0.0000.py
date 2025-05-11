from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from scipy.linalg import cholesky, solve_triangular

class CGHBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim
        self.exploration_weight = 0.1
        self.best_x = None
        self.best_y = float('inf')
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))


        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, method='lhs'):
        # sample points
        # return array of shape (n_points, n_dims)
        if method == 'lhs':
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        elif method == 'covariance':
            if self.X is None:
                return self._sample_points(n_points, method='lhs')
            else:
                # Sample from the GPR posterior using the covariance matrix
                model = self._fit_model(self.X, self.y)
                mu, cov = model.predict(self.X, return_cov=True)
                try:
                    L = cholesky(cov, lower=True)
                    z = np.random.normal(size=(n_points, self.X.shape[0]))
                    samples = mu + L @ z.T
                    samples = samples.T
                    # Clip samples to stay within the problem bounds
                    samples = np.clip(samples, self.bounds[0], self.bounds[1])
                    return samples
                except np.linalg.LinAlgError:
                    # If covariance matrix is not positive definite, return LHS samples
                    return self._sample_points(n_points, method='lhs')
        else:
            raise ValueError("Invalid sampling method.")

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        self.kernel = model.kernel_ # Update kernel with optimized parameters
        return model

    def _acquisition_function(self, X, model):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if self.best_y is None:
            ei = np.zeros_like(mu)
        else:
            imp = self.best_y - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0  # handle zero sigma

        # Add covariance-guided exploration term
        if self.X is not None:
            _, cov = model.predict(X, return_cov=True)
            if len(cov.shape) == 0:
                cov = np.array([[cov]])
            exploration_term = np.diag(cov).reshape(-1, 1)
            ei = ei + self.exploration_weight * exploration_term

        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # return array of shape (batch_size, n_dims)
        
        # Sample candidate points using a combination of LHS and covariance-guided sampling
        n_candidates_lhs = batch_size // 2
        n_candidates_covariance = batch_size - n_candidates_lhs
        
        candidate_points_lhs = self._sample_points(n_candidates_lhs, method='lhs')
        candidate_points_covariance = self._sample_points(n_candidates_covariance, method='covariance')
        candidate_points = np.vstack((candidate_points_lhs, candidate_points_covariance))

        # Calculate acquisition function values
        model = self._fit_model(self.X, self.y)
        acquisition_values = self._acquisition_function(candidate_points, model)

        # Select top batch_size points based on acquisition function values
        indices = np.argsort(-acquisition_values.flatten())[:batch_size]
        selected_points = candidate_points[indices]

        # Local refinement
        refined_points = []
        for point in selected_points:
            res = minimize(lambda x: model.predict(x.reshape(1, -1))[0], point, bounds=[(-5, 5)] * self.dim, method='L-BFGS-B')
            refined_points.append(res.x)
        
        return np.array(refined_points)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y
    
    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        # Do not change the function signature
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))
        
        # Update best seen value
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        initial_X = self._sample_points(self.n_init, method='lhs')
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        
        batch_size = 5
        while self.n_evals < self.budget:
            # Optimization
            model = self._fit_model(self.X, self.y)
            
            # select points by acquisition function
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
