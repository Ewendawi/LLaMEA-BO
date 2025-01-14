from typing import Callable
from scipy.stats import qmc
import numpy as np
import GPy
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy.stats import norm

class RobustGPBO:
    """
    Robust Bayesian Optimization using Gaussian Process with Matern kernel, 
    Latin Hypercube Sampling for initial points, and Expected Improvement as the acquisition function.
    
    Techniques:
    - Gaussian Process Regression: Utilizes GPy library for GP modeling.
    - Matern kernel: Employs Matern52 kernel for GP covariance function.
    - Latin Hypercube Sampling: Uses scipy.stats.qmc for initial point sampling.
    - Expected Improvement: Implements Expected Improvement as acquisition function.
    - L-BFGS-B optimization: Uses scipy.optimize.minimize with L-BFGS-B method to find the next point.
    
    Parameters:
    - n_initial_points: Number of initial points sampled using Latin Hypercube Sampling.
    - kernel: The kernel used in Gaussian Process Regression.
    - acquisition_function: The acquisition function used to select next points.
    - loss_name: The name of the loss used to evaluate the model.
    """
    def __init__(self):
        # Initialize optimizer settings
        self.n_initial_points = 10
        self.kernel = GPy.kern.Matern52(input_dim=1, variance=1., lengthscale=1.)
        self.acquisition_function = self._expected_improvement
        self.loss_name = "RMSE"
    
    def _sample_points(self, n_points, bounds) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=bounds.shape[1])
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, bounds[0], bounds[1])
    
    def _fit_model(self, X, y):
        model = GPy.models.GPRegression(X, y, self.kernel)
        model.optimize_restarts(num_restarts=5, verbose=False)
        return model

    def _get_model_mean_loss(self, model, X, y) -> np.float64:
        y_pred, _ = model.predict(X)
        return np.sqrt(mean_squared_error(y, y_pred))
    
    def _expected_improvement(self, X, model, y_best) -> np.ndarray:
        y_mean, y_var = model.predict(X)
        y_std = np.sqrt(y_var)
        imp = y_best - y_mean
        z = imp / y_std
        ei = imp * norm.cdf(z) + y_std * norm.pdf(z)
        ei[y_std <= 0] = 0
        return ei

    def _select_next_points(self, model, y_best, bounds) -> np.ndarray:
        
        def min_obj(x):
            x = np.array(x).reshape(1,-1)
            return -self._expected_improvement(x, model, y_best)

        best_x = None
        best_ei = float('inf')

        for _ in range(10):
            x0 = self._sample_points(1, bounds)
            res = minimize(min_obj, x0, bounds=list(zip(bounds[0], bounds[1])), method='L-BFGS-B')
            if res.fun < best_ei:
                best_ei = res.fun
                best_x = res.x
        
        return best_x.reshape(1,-1)


    def optimize(self, objective_fn:Callable[[np.ndarray], np.ndarray], bounds:np.ndarray, budget:int) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, str], int]:
        
        n_initial_points = self.n_initial_points
        X_init = self._sample_points(n_initial_points, bounds)
        y_init = objective_fn(X_init)
        X_all, y_all = X_init, y_init
        model_losses = []
        
        model = self._fit_model(X_init, y_init)
        model_loss = self._get_model_mean_loss(model, X_init, y_init)
        model_losses.append(model_loss)

        rest_of_budget = budget - n_initial_points
        while rest_of_budget > 0:
            y_best = np.min(y_all)
            next_point = self._select_next_points(model, y_best, bounds)
            y_next = objective_fn(next_point)
            X_all = np.concatenate((X_all, next_point), axis=0)
            y_all = np.concatenate((y_all, y_next), axis=0)
            
            model = self._fit_model(X_all, y_all)
            model_loss = self._get_model_mean_loss(model, X_all, y_all)
            model_losses.append(model_loss)
            
            rest_of_budget -= next_point.shape[0]
        return y_all, X_all, (np.array(model_losses), self.loss_name), n_initial_points