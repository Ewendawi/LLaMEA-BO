# Description
**EHBBO-AERLS: Efficient Hybrid Bayesian Optimization with Adaptive Exploration-Refinement and Local Search.** This algorithm builds upon EHBBO by introducing an adaptive strategy to balance exploration and refinement during point selection. It dynamically adjusts the number of candidates used for Thompson Sampling based on the uncertainty of the Gaussian Process model. Additionally, it integrates a more robust local search method (SLSQP) and adaptively scales the bounds of the local search based on the iteration number to control the search space.

# Justification
1.  **Adaptive Exploration-Refinement:** The original EHBBO uses a fixed number of candidates for Thompson Sampling. This modification dynamically adjusts the number of candidates based on the average standard deviation predicted by the GP model. When the uncertainty is high (early iterations), more candidates are sampled to explore the space. As the algorithm converges and the uncertainty decreases, fewer candidates are sampled, allowing for more focused refinement. This adaptive strategy improves the balance between exploration and exploitation.
2.  **Improved Local Search:** The original EHBBO uses L-BFGS-B for local search. This is replaced with SLSQP, which can handle constraints more effectively and is generally more robust.
3.  **Adaptive Local Search Bounds:** The bounds of the local search are adaptively scaled based on the iteration number. In the initial iterations, the local search is performed within a smaller region around the selected point to encourage exploration. As the algorithm progresses, the bounds are gradually increased to allow for more aggressive exploitation. This adaptive scaling helps to prevent premature convergence and improves the overall performance of the algorithm.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class EHBBORLS:
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
        self.iteration = 0

        # Do not add any other arguments without a default value
        self.best_x = None
        self.best_y = float('inf')

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

    def _acquisition_function(self, X, model):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if self.best_y is None:
            return np.zeros_like(mu)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # handle zero sigma

        return ei

    def _select_next_points(self, batch_size, model):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Adaptive Thompson Sampling for batch selection
        mu, sigma = model.predict(self.X, return_std=True)
        avg_sigma = np.mean(sigma)
        num_candidates = int(100 * batch_size * (1 + avg_sigma)) # Scale candidates by uncertainty
        num_candidates = max(batch_size, min(num_candidates, 5000)) # Clamp candidate number

        candidates = self._sample_points(num_candidates)
        
        # Sample from the posterior
        mu, sigma = model.predict(candidates, return_std=True)
        sampled_values = np.random.normal(mu, sigma)
        
        # Select top batch_size candidates
        indices = np.argsort(sampled_values)[:batch_size]
        selected_points = candidates[indices]

        # Local search refinement with SLSQP and adaptive bounds
        refined_points = []
        ls_frac = min(1.0, self.iteration / (self.budget / (2 * batch_size))) # scale local search range
        ls_bounds = ls_frac * (self.bounds[1] - self.bounds[0]) / 2 # adaptive bounds
        for point in selected_points:
            local_bounds = [(max(self.bounds[0][i], point[i] - ls_bounds[i]), min(self.bounds[1][i], point[i] + ls_bounds[i])) for i in range(self.dim)]
            res = minimize(lambda x: model.predict(x.reshape(1, -1))[0], point, bounds=local_bounds, method='SLSQP')
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
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        
        batch_size = 5
        while self.n_evals < self.budget:
            self.iteration += 1
            # Optimization
            model = self._fit_model(self.X, self.y)
            
            # select points by acquisition function
            next_X = self._select_next_points(batch_size, model)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm EHBBORLS got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1646 with standard deviation 0.1101.

took 122.69 seconds to run.