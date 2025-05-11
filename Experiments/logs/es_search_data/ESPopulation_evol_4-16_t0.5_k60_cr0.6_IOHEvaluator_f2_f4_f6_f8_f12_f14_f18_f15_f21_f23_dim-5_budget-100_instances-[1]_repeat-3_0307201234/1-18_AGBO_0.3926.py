from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize

class AGBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * self.dim # Initial samples
        self.exploration_factor = 2.0
        self.alpha = 0.5  # Weighting between UCB and EI

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        samples = sampler.random(n=n_points)
        return qmc.scale(samples, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        # Check for NaN values and impute if necessary
        if np.isnan(X).any():
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)

        model = HistGradientBoostingRegressor(random_state=0)
        model.fit(X, y.ravel())  # HistGradientBoostingRegressor expects y to be 1D
        return model

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu = self.model.predict(X).reshape(-1, 1)
        # Gradient boosting does not directly provide uncertainty estimates
        # Using a constant value for sigma as a proxy for uncertainty
        sigma = np.std(self.y) * np.ones_like(mu) if self.y is not None and len(self.y) > 1 else np.ones_like(mu)

        # Upper Confidence Bound
        ucb = mu + self.exploration_factor * sigma
        
        # Expected Improvement
        best_y = np.min(self.y) if self.y is not None else 0.0
        imp = best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0] = 0  # Handle cases with zero variance

        # Adaptive weighting
        f = self.alpha * ucb + (1 - self.alpha) * ei
        return -f # minimize -f

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Local search around the best point
        if self.X is not None and len(self.X) > 0:
            best_idx = np.argmin(self.y)
            best_x = self.X[best_idx]
            
            # Gaussian mutation
            mutation_scale = 0.1 * (self.bounds[1] - self.bounds[0])
            X_local = np.clip(best_x + np.random.normal(0, mutation_scale, size=(batch_size, self.dim)), self.bounds[0], self.bounds[1])
        else:
            X_local = self._sample_points(batch_size)

        # Global search based on acquisition function
        def obj(x):
            return self._acquisition_function(x.reshape(1, -1)).ravel()

        X_global = []
        for _ in range(batch_size):
            x0 = self._sample_points(1)  # Initial guess
            res = minimize(obj, x0, bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)])
            X_global.append(res.x)
        X_global = np.array(X_global)

        # Combine local and global search
        if np.random.rand() < 0.5: # 50% chance of local vs global
            return X_local
        else:
            return X_global
        

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
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        self.model = self._fit_model(self.X, self.y)
        
        while self.n_evals < self.budget:
            # Optimization
            batch_size = 1
            X_next = self._select_next_points(batch_size)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

            self.model = self._fit_model(self.X, self.y)

            # Update exploration factor and alpha
            self.exploration_factor = 1.0 + (self.budget - self.n_evals) / self.budget
            self.alpha = (self.budget - self.n_evals) / self.budget  # Adapt alpha

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        return best_y, best_x
