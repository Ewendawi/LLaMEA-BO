# Description: Bayesian Optimization algorithm using Gaussian Process as the surrogate model, Expected Improvement as the acquisition function, Latin Hypercube Sampling for initial points, and a dynamic strategy for the number of initial points.

# Code
```python
from typing import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

class DynamicInitialGP_EI_BO:
    def __init__(self, dim:int, budget:int):
        # Initialize optimizer settings
        self.kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        self.n_restarts = 10
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.dim = dim
        self.budget = budget
        self.bounds = np.array([[-5] * dim, [5] * dim])
        
    def _sample_points(self, n_points, bounds) -> np.ndarray:
        # sample points using LHS
        sampler = qmc.LatinHypercube(d=bounds.shape[1])
        sample = sampler.random(n_points)
        return qmc.scale(sample, bounds[0], bounds[1])
    
    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # Scale data before training
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=self.n_restarts)
        model.fit(X_scaled, y_scaled)
        return  model

    def _acquisition_function(self, X, model, y_best) -> np.ndarray:
        # Implement Expected Improvement acquisition function 
        # calculate the acquisition function value for each point in X
        X_scaled = self.scaler_X.transform(X)
        y_best_scaled = self.scaler_y.transform(y_best.reshape(-1,1)).flatten()[0]

        mu, sigma = model.predict(X_scaled, return_std=True)
        imp = mu - y_best_scaled
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0
        return ei.reshape(-1, 1)

    def _select_next_points(self, model, batch_size, bounds, all_y) -> np.ndarray:
        # Implement the strategy to select the next points to evaluate
        # return array of shape (batch_size, n_dims)
        def obj_func(x):
           return -self._acquisition_function(x.reshape(1, -1), model, np.min(all_y))[0]
        
        x0 = self._sample_points(batch_size*10, bounds) #generate more candidates
        best_x = []
        for i in range(batch_size):
            res = minimize(obj_func, x0[i], bounds=list(zip(bounds[0], bounds[1])), method='L-BFGS-B')
            best_x.append(res.x)
        return np.array(best_x)

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        n_dims = self.dim
        n_initial_points = 2 * n_dims
        
        X_init = self._sample_points(n_initial_points, self.bounds)
        y_init = func(X_init)

        all_x = X_init
        all_y = y_init

        model = self._fit_model(all_x, all_y)

        rest_of_budget = self.budget - n_initial_points
        batch_size = 1
        while rest_of_budget > 0:
            X_next = self._select_next_points(model, batch_size, self.bounds, all_y)
            y_next = func(X_next)

            all_x = np.concatenate((all_x, X_next), axis=0)
            all_y = np.concatenate((all_y, y_next), axis=0)
            
            model = self._fit_model(all_x, all_y)
           
            rest_of_budget -= X_next.shape[0]

        return np.min(all_y), all_x
    
from scipy.stats import norm

```