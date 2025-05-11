# Description
**Adaptive Temperature Hybrid Bayesian Optimization (ATHBBO):** This algorithm refines the EHBBO by introducing an adaptive temperature schedule for Thompson Sampling, a more robust local search strategy, and dynamic kernel parameter tuning for the Gaussian Process. The temperature adaptation is based on both the best function value seen so far and the diversity of the evaluated points. The local search uses a multi-start approach with a combination of L-BFGS-B and random restarts. The kernel parameters are re-tuned periodically to better fit the observed data.

# Justification
The original EHBBO algorithm uses a simple temperature schedule that depends only on the best function value. This can lead to premature convergence if the initial best value is not representative of the global optimum. By incorporating the diversity of the evaluated points into the temperature schedule, the algorithm can maintain exploration for longer, especially when the evaluated points are clustered in a small region of the search space. The multi-start local search strategy helps to escape local optima and find better solutions. The dynamic kernel parameter tuning allows the Gaussian Process to adapt to the changing landscape of the objective function, improving the accuracy of the surrogate model and the effectiveness of the acquisition function. The adaptive batch size ensures that the algorithm efficiently uses the available budget, balancing exploration and exploitation based on the remaining evaluations.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize

class ATHBBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * (dim + 1) # Number of initial samples, increased for higher dimensions
        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, nu=2.5)
        self.diversity_threshold = 0.1  # Threshold for diversity-based temperature adjustment
        self.kernel_update_interval = 5 * dim # Update kernel every this many evaluations

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5, alpha=1e-6)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)

        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)

        # Thompson Sampling with adaptive temperature
        temperature = 1.0 / (1.0 + np.exp(- (self.best_y + 1e-9)))

        # Diversity-based temperature adjustment
        if self.X is not None and len(self.X) > self.dim:
            distances = np.linalg.norm(self.X - np.mean(self.X, axis=0), axis=1)
            diversity = np.std(distances) / (np.max(distances) - np.min(distances) + 1e-9)
            if diversity < self.diversity_threshold:
                temperature *= 2  # Increase temperature if diversity is low

        samples = np.random.normal(mu, temperature * sigma)
        return samples.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Optimize the acquisition function using a multi-start local search strategy
        x_starts = self._sample_points(batch_size)
        x_next = []
        for x_start in x_starts:
            # L-BFGS-B optimization
            res = minimize(lambda x: -self._acquisition_function(x.reshape(1, -1))[0,0], 
                           x_start, 
                           bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)],
                           method='L-BFGS-B')
            
            best_x = res.x
            best_acq = -res.fun

            # Random restart
            for _ in range(2): # Number of random restarts
                x_restart = self._sample_points(1).flatten()
                res_restart = minimize(lambda x: -self._acquisition_function(x.reshape(1, -1))[0,0],
                                       x_restart,
                                       bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)],
                                       method='L-BFGS-B')
                
                if -res_restart.fun > best_acq:
                    best_x = res_restart.x
                    best_acq = -res_restart.fun
            
            x_next.append(best_x)
        return np.array(x_next)

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
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Optimization loop
        while self.n_evals < self.budget:

            # Update kernel parameters periodically
            if self.n_evals % self.kernel_update_interval == 0:
                # Tune kernel parameters based on current data
                self.kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, nu=2.5)
                self._fit_model(self.X, self.y)

            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Select next points by acquisition function
            batch_size = min(self.budget - self.n_evals, max(1, self.dim // 2)) # Adaptive batch size
            X_next = self._select_next_points(batch_size)

            # Evaluate the points
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ATHBBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1438 with standard deviation 0.0995.

took 646.71 seconds to run.