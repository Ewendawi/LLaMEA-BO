### Description
The problem involves minimizing two functions from the BBOB test suite, F24-LunacekBiRastrigin and F19-GriewankRosenbrock, in a 5-dimensional space with bounds [-5.0, 5.0] for each dimension. 
To address this problem, we can design a Bayesian Optimization algorithm that utilizes a Gaussian Process (GP) as the surrogate model and the Expected Improvement (EI) acquisition function to select the next points to evaluate. 
The GP will be used to model the objective function, and the EI acquisition function will be used to balance exploration and exploitation. 
We will also use a QMC sampling method to initialize the GP model.

### /Description

### Pseudocode
1. Initialize the GP model using QMC sampling and the objective function evaluations.
2. Fit the GP model to the initial data.
3. Calculate the acquisition function values for a set of candidate points.
4. Select the next point to evaluate based on the acquisition function values.
5. Evaluate the objective function at the selected point.
6. Update the GP model with the new data.
7. Repeat steps 3-6 until the budget is exhausted.
8. Return the best point found and its corresponding objective function value.

### /Pseudocode


### Code
```python
from typing import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm


class AdaptiveLengthScale_EI_UCB_BO:
    """
    Bayesian Optimization algorithm using Gaussian Process as the surrogate model,
    switching between Expected Improvement and Upper Confidence Bound as the acquisition function,
    adaptive length scale for the RBF kernel, Latin Hypercube Sampling for initial points,
    and a dynamic strategy for the number of initial points.

    Parameters:
        n_restarts (int): Number of restarts for the GP optimizer.
        exploration_iterations (int): Number of iterations to use UCB as the acquisition function.
    """
    def __init__(self, n_restarts=10, exploration_iterations=10):
        # Initialize optimizer settings
        self.kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
        self.n_restarts = n_restarts
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.exploration_iterations = exploration_iterations
        
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

    def _get_model_loss(self, model, X, y) -> np.float64:
        # Calculate the loss of the model
        # Scale data before calculating loss
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.scaler_y.transform(y.reshape(-1, 1)).flatten()
        return -model.log_marginal_likelihood(model.kernel_.theta, eval_gradient=False)

    def _adaptive_length_scale(self, X):
         # Calculate the mean distance between the points in X
        if X.shape[0] > 1:
            distances = pdist(X)
            mean_distance = np.mean(distances)
            self.kernel.k2.length_scale = mean_distance
            
    def _acquisition_function(self, X, model, y_best, iteration) -> np.ndarray:
        # Implement Expected Improvement acquisition function 
        # calculate the acquisition function value for each point in X
        X_scaled = self.scaler_X.transform(X)
        y_best_scaled = self.scaler_y.transform(y_best.reshape(-1,1)).flatten()[0]

        mu, sigma = model.predict(X_scaled, return_std=True)
        if iteration < self.exploration_iterations:
            # UCB for exploration
            beta = 2
            return mu + beta * sigma
        else:
            # EI for exploitation
            imp = mu - y_best_scaled
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0
            return ei.reshape(-1, 1)

    def _select_next_points(self, model, batch_size, bounds, all_y, iteration) -> np.ndarray:
        # Implement the strategy to select the next points to evaluate
        # return array of shape (batch_size, n_dims)
        def obj_func(x):
           return -self._acquisition_function(x.reshape(1, -1), model, np.min(all_y), iteration)[0]
        
        x0 = self._sample_points(batch_size*10, bounds) #generate more candidates
        best_x = []
        for i in range(batch_size):
            res = minimize(obj_func, x0[i], bounds=list(zip(bounds[0], bounds[1])), method='L-BFGS-B')
            best_x.append(res.x)
        return np.array(best_x)

    def optimize(self, objective_fn:Callable[[np.ndarray], np.ndarray], bounds:np.ndarray, budget:int) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, str], int]:
        # Main minimize optimization loop
        # objective_fn: Callable[[np.ndarray], np.ndarray], takes array of shape (n_points, n_dims) and returns array of shape (n_points, 1).
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        # Do not change the function signature
        # Evaluate the model using the metric you choose and record the value as model_loss after each training. the size of the model_loss should be equal to the number of iterations plus one for the fit on initial points.
        # Return a tuple (all_y, all_x, (model_losses, loss_name), n_initial_points)
        
        n_dims = bounds.shape[1]
        n_initial_points = 2 * n_dims
        
        X_init = self._sample_points(n_initial_points, bounds)
        y_init = objective_fn(X_init)

        all_x = X_init
        all_y = y_init
        model_losses = []
        loss_name = "Negative Log Likelihood"
        iteration = 0

        model = self._fit_model(all_x, all_y)
        model_loss = self._get_model_loss(model, all_x, all_y)
        model_losses.append(model_loss)

        rest_of_budget = budget - n_initial_points
        batch_size = 1
        while rest_of_budget > 0:
            self._adaptive_length_scale(all_x)
            X_next = self._select_next_points(model, batch_size, bounds, all_y, iteration)
            y_next = objective_fn(X_next)

            all_x = np.concatenate((all_x, X_next), axis=0)
            all_y = np.concatenate((all_y, y_next), axis=0)
            
            model = self._fit_model(all_x, all_y)
            model_loss = self._get_model_loss(model, all_x, all_y)
            model_losses.append(model_loss)
           
            rest_of_budget -= X_next.shape[0]
            iteration += 1

        return all_y, all_x, (np.array(model_losses), loss_name), n_initial_points

```

### /Code