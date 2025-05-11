# Description
**ReBO: Regularized Bayesian Optimization:** This algorithm introduces a regularization term to the acquisition function to prevent overfitting and promote exploration in Bayesian Optimization. It uses a Gaussian Process Regression (GPR) model with an RBF kernel for surrogate modeling. The acquisition function is a combination of Expected Improvement (EI) and a regularization term based on the L2 norm of the GPR's predicted mean. This regularization encourages the algorithm to explore regions where the model's predictions are less extreme, thus mitigating the risk of prematurely converging to a local optimum. Additionally, an adaptive scaling factor is introduced for the regularization term, which adjusts based on the iteration number, gradually decreasing the regularization as the algorithm progresses. The initial exploration is performed using Latin Hypercube Sampling (LHS).

# Justification
The key components and changes are justified as follows:

1.  **Regularized Acquisition Function:** The addition of an L2 regularization term to the EI acquisition function is designed to combat overfitting. Overfitting in BO can lead to premature convergence, especially when the number of function evaluations is limited. By penalizing extreme predictions, the regularization term encourages exploration of regions where the model is less certain.

2.  **Adaptive Scaling Factor:** The adaptive scaling factor for the regularization term provides a mechanism to balance exploration and exploitation over time. Initially, a higher scaling factor promotes broader exploration. As the algorithm gathers more data, the scaling factor decreases, allowing the algorithm to focus more on exploitation. The scaling factor decreases linearly with the number of iterations.

3.  **LHS Initial Sampling:** Latin Hypercube Sampling (LHS) is used for initial exploration to ensure a good initial coverage of the search space. This helps to build a more robust initial GPR model.

4.  **Addressing Previous Errors:** The CGHBO algorithm had a broadcasting error in the covariance sampling. In ReBO, covariance sampling is avoided to prevent this error. The DABO algorithm had an indexing error in the `_update_eval_points` function. ReBO does not use the same indexing mechanism, thus avoiding this error.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ReBO:
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
        self.reg_weight = 0.1 # Initial weight for the regularization term
        self.best_x = None
        self.best_y = float('inf')

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
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X, model, iteration):
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
            
        # Adaptive Regularization
        reg_weight = self.reg_weight * (1 - (iteration / self.budget))
        regularization_term = -reg_weight * np.linalg.norm(mu, axis=1, keepdims=True)**2
        ei = ei + regularization_term

        return ei

    def _select_next_points(self, batch_size, model, iteration):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # return array of shape (batch_size, n_dims)
        
        # Sample candidate points
        candidate_points = self._sample_points(100 * batch_size)

        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_points, model, iteration)

        # Select top batch_size points based on acquisition function values
        indices = np.argsort(-acquisition_values.flatten())[:batch_size]
        selected_points = candidate_points[indices]

        return selected_points

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
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        
        batch_size = 5
        iteration = self.n_init
        while self.n_evals < self.budget:
            # Optimization
            model = self._fit_model(self.X, self.y)
            
            # select points by acquisition function
            next_X = self._select_next_points(batch_size, model, iteration)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            iteration += batch_size

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ReBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1632 with standard deviation 0.0970.

took 51.69 seconds to run.