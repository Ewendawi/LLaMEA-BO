# Description
**Adaptive Trust Region Bayesian Optimization (ATRBO):** This algorithm employs a Gaussian Process (GP) surrogate model with a Matérn kernel for enhanced flexibility in modeling functions with varying degrees of smoothness. It uses a trust-region approach, where the size of the trust region is adaptively adjusted based on the agreement between the GP model's predictions and the actual function evaluations. The acquisition function is based on the Lower Confidence Bound (LCB), which balances exploration and exploitation. To further improve exploration, especially in early stages, the algorithm incorporates a dynamic exploration factor in the LCB. The initial design uses a Sobol sequence for better space-filling properties compared to LHS.

# Justification
The ATRBO algorithm diverges from EHBBO in several key aspects:

1.  **Surrogate Model:** Instead of a fixed RBF kernel, ATRBO uses a Matérn kernel in the GP, allowing the model to adapt to functions with different smoothness levels. This is crucial for handling the variety of functions in the BBOB suite.
2.  **Trust Region:** ATRBO introduces a trust-region mechanism. This helps to prevent the algorithm from taking steps that are too large when the GP model is uncertain, thus improving the robustness and stability of the optimization process. The adaptive adjustment of the trust region size based on model accuracy ensures efficient exploration and exploitation.
3.  **Acquisition Function:** ATRBO uses the LCB acquisition function with a dynamic exploration factor. LCB is known to be less sensitive to the hyperparameters of the GP compared to EI, which can make the optimization more stable. The dynamic exploration factor further enhances exploration early in the optimization.
4.  **Initial Design:** ATRBO utilizes a Sobol sequence for the initial design. Sobol sequences are known to have better space-filling properties than Latin Hypercube Sampling, especially in higher dimensions, which can lead to a better initial GP model.
5.  **Error Analysis of EHBBO:** The EHBBO algorithm, with its fixed RBF kernel and EI acquisition, might struggle with functions that have different smoothness properties or are highly multimodal. The local search in EHBBO, while helpful, might not be sufficient to escape local optima. ATRBO addresses these potential issues with its adaptive kernel, trust-region mechanism, and dynamic exploration.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.optimize import minimize

class ATRBO:
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
        self.trust_region_size = 2.0  # Initial trust region size
        self.exploration_factor = 2.0 # Initial exploration factor

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
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, nu=2.5)
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Lower Confidence Bound
        lcb = mu - self.exploration_factor * sigma
        return lcb

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

            # Adaptive trust region adjustment
            y_pred, sigma = self.model.predict(X_next, return_std=True)
            y_pred = y_pred.reshape(-1, 1)
            
            # Agreement between prediction and actual value
            agreement = np.abs(y_pred - y_next) / sigma.reshape(-1, 1)
            
            if np.mean(agreement) < 1.0:
                self.trust_region_size *= 1.1  # Increase trust region if model is accurate
            else:
                self.trust_region_size *= 0.9  # Decrease trust region if model is inaccurate
            
            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0) # Clip trust region size

            # Dynamic exploration factor adjustment
            self.exploration_factor = 1.0 + (self.budget - self.n_evals) / self.budget # Reduce exploration over time
            
            self.model = self._fit_model(self.X, self.y) # Refit the model with new data
            
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        return best_y, best_x
```
## Feedback
 The algorithm ATRBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1823 with standard deviation 0.1030.

took 194.42 seconds to run.