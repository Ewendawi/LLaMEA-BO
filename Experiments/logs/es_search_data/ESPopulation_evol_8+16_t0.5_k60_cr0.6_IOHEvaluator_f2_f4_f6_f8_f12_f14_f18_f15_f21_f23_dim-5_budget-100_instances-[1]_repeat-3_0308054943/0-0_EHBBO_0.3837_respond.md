# Description
**Efficient Hybrid Bayesian Optimization (EHBBO):** This algorithm combines Gaussian Process Regression (GPR) with Expected Improvement (EI) acquisition, Latin Hypercube Sampling (LHS) for initial exploration, and a local search refinement step using a simple gradient-based method to enhance exploitation. A batch selection strategy based on Thompson Sampling is used to select multiple points for evaluation in each iteration, balancing exploration and exploitation.

# Justification
1.  **Gaussian Process Regression (GPR):** GPR is chosen as the surrogate model due to its ability to provide uncertainty estimates along with predictions. This is crucial for effective acquisition function design.
2.  **Expected Improvement (EI):** EI is used as the acquisition function to balance exploration and exploitation. It quantifies the expected improvement over the current best solution.
3.  **Latin Hypercube Sampling (LHS):** LHS is employed for initial sampling to ensure a good coverage of the search space. This helps in building a better initial model.
4.  **Local Search Refinement:** A gradient-based local search is incorporated to refine promising solutions found by the BO process. This enhances exploitation and helps to converge to local optima more quickly.
5.  **Thompson Sampling for Batch Selection:** Thompson Sampling is used for batch selection because it naturally balances exploration and exploitation by sampling from the posterior distribution over the acquisition function.
6.  **Computational Efficiency:** The GPR model is implemented using scikit-learn, which provides an efficient implementation. The local search is kept simple to minimize computational overhead.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize

class EHBBO:
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
        
        # Thompson Sampling for batch selection
        num_candidates = 100 * batch_size
        candidates = self._sample_points(num_candidates)
        
        # Sample from the posterior
        mu, sigma = model.predict(candidates, return_std=True)
        sampled_values = np.random.normal(mu, sigma)
        
        # Select top batch_size candidates
        indices = np.argsort(sampled_values)[:batch_size]
        selected_points = candidates[indices]

        # Local search refinement
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
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        
        batch_size = 5
        while self.n_evals < self.budget:
            # Optimization
            model = self._fit_model(self.X, self.y)
            
            # select points by acquisition function
            next_X = self._select_next_points(batch_size, model)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm EHBBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1592 with standard deviation 0.0990.

took 151.03 seconds to run.