# Description
**Gradient-Boosted Trust Region Bayesian Optimization (GBTRBO):** This algorithm combines the strengths of Gradient-Boosted surrogate models and Trust Region methods. It uses a HistGradientBoostingRegressor for efficient and accurate modeling of the objective function, particularly for non-smooth or discontinuous functions. It incorporates a trust region approach to balance exploration and exploitation, adapting the trust region size based on the model's performance. The acquisition function utilizes the Upper Confidence Bound (UCB), and the exploration factor is dynamically adjusted. A Sobol sequence is used for initial sampling. To improve the next point selection, the acquisition function is optimized within the trust region using a multi-start L-BFGS-B approach.

# Justification
The algorithm combines the strengths of GBO and ATRBO.
*   **Gradient Boosting:** GBO uses HistGradientBoostingRegressor, which is computationally efficient and can capture complex relationships in the data, especially non-smoothness.
*   **Trust Region:** ATRBO uses a trust region approach, which helps to balance exploration and exploitation. The trust region size is adapted based on the model's performance.
*   **Acquisition Function Optimization:** To select the next points, the acquisition function is optimized within the trust region using a multi-start L-BFGS-B approach. This is more efficient than random sampling and helps to find better points.
*   **Initial Sampling:** Sobol sequence is used for initial sampling, which provides better space-filling properties than LHS.
*   **Dynamic Exploration Factor:** The exploration factor is dynamically adjusted to balance exploration and exploitation.
*   **Computational Efficiency:** HistGradientBoostingRegressor is computationally efficient, and the multi-start L-BFGS-B approach is also relatively efficient.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize

class GBTRBO:
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
        self.trust_region_size = 2.0

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.Sobol(d=self.dim, scramble=False)
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
        sigma = np.zeros_like(mu) # Gradient boosting does not directly provide uncertainty estimates

        # Upper Confidence Bound
        ucb = mu + self.exploration_factor * sigma
        return -ucb # minimize -ucb

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Optimize the acquisition function within the trust region using L-BFGS-B
        best_idx = np.argmin(self.y)
        best_x = self.X[best_idx]
        
        x_starts = best_x + np.random.normal(0, 0.1, size=(batch_size, self.dim))
        x_starts = np.clip(x_starts, self.bounds[0], self.bounds[1])

        candidates = []
        values = []
        for x_start in x_starts:
            # Define trust region bounds
            lower_bound = np.maximum(x_start - self.trust_region_size / 2, self.bounds[0])
            upper_bound = np.minimum(x_start + self.trust_region_size / 2, self.bounds[1])
            
            res = minimize(lambda x: self._acquisition_function(x.reshape(1, -1)),
                           x_start,
                           bounds=np.array([lower_bound, upper_bound]).T,
                           method="L-BFGS-B")
            candidates.append(res.x)
            values.append(res.fun)
        
        return np.array(candidates)

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

            # Adaptive trust region adjustment - simplified version
            agreement = np.var(y_next)  # Use variance as a proxy for model accuracy
            if agreement < 0.1:  # If variance is low, trust the model more
                self.trust_region_size *= 1.1
            else:
                self.trust_region_size *= 0.9
            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0)

            # Update exploration factor
            self.exploration_factor = 1.0 + (self.budget - self.n_evals) / self.budget
            
            self.model = self._fit_model(self.X, self.y)
            
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        return best_y, best_x
```
## Feedback
 The algorithm GBTRBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1604 with standard deviation 0.0941.

took 91.35 seconds to run.