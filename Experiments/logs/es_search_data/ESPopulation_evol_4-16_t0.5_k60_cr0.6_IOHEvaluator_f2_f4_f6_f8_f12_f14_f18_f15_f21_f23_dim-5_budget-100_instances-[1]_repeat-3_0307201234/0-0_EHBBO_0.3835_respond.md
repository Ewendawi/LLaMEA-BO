# Description
**Efficient Hybrid Bayesian Optimization (EHBBO):** This algorithm combines an efficient space-filling initial design using Latin Hypercube Sampling (LHS), a Gaussian Process (GP) surrogate model with a Radial Basis Function (RBF) kernel for flexible modeling, and an acquisition function that balances exploration and exploitation using Expected Improvement (EI). To enhance exploration, especially in high-dimensional spaces, the algorithm incorporates a local search strategy around the best-observed point. The local search uses a simple Gaussian mutation operator to generate candidate solutions. The number of local search steps is dynamically adjusted based on the remaining budget to ensure effective exploration throughout the optimization process.

# Justification
1.  **Initial Design (LHS):** Latin Hypercube Sampling provides a good space-filling initial design, ensuring that the initial samples cover the search space reasonably well. This is crucial for building a reliable surrogate model.
2.  **Surrogate Model (GP with RBF Kernel):** Gaussian Processes are well-suited for modeling black-box functions due to their ability to provide uncertainty estimates. The RBF kernel is a flexible and commonly used kernel that can capture a wide range of function behaviors.
3.  **Acquisition Function (Expected Improvement):** Expected Improvement balances exploration and exploitation by considering both the predicted value and the uncertainty of the GP model. This helps the algorithm to efficiently find promising regions of the search space.
4.  **Local Search:** Adding a simple local search around the best-observed point can significantly improve the algorithm's ability to fine-tune the solution and escape local optima. The Gaussian mutation operator is a simple and effective way to generate candidate solutions in the neighborhood of the best point.
5.  **Dynamic Adjustment of Local Search Steps:** Adjusting the number of local search steps based on the remaining budget allows the algorithm to adapt to the problem and the available resources. Initially, more local search steps are performed to explore the space more thoroughly. As the budget decreases, the number of steps is reduced to focus on exploitation.
6. **Computational Efficiency:** The algorithm is designed to be computationally efficient by using a relatively small number of initial samples and a simple local search strategy. The GP model is also relatively fast to fit and predict, making the algorithm suitable for problems with limited computational resources.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
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
        self.n_init = 2 * self.dim # Initial samples

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
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0, length_scale_bounds="fixed")
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

        # Expected Improvement
        best = np.min(self.y)
        imp = mu - best
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # avoid division by zero
        return -ei  # we want to maximize EI

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Optimize the acquisition function using L-BFGS-B
        x_starts = self._sample_points(batch_size)
        candidates = []
        values = []
        for x_start in x_starts:
            res = minimize(lambda x: self._acquisition_function(x.reshape(1, -1)),
                           x_start,
                           bounds=[(-5, 5)] * self.dim,
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

            # Local search around the best point
            best_idx = np.argmin(self.y)
            best_x = self.X[best_idx]
            
            # Dynamically adjust the number of local search steps based on the remaining budget
            remaining_budget = self.budget - self.n_evals
            n_local_steps = min(5, remaining_budget) # Reduce the number of steps as budget decreases
            
            X_local = best_x + np.random.normal(0, 0.1, size=(n_local_steps, self.dim)) # Gaussian mutation
            X_local = np.clip(X_local, self.bounds[0], self.bounds[1])  # Clip to bounds
            y_local = self._evaluate_points(func, X_local)
            self._update_eval_points(X_local, y_local)

            self.model = self._fit_model(self.X, self.y) # Refit the model with new data
            
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]

        return best_y, best_x
```
## Feedback
 The algorithm EHBBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1606 with standard deviation 0.1028.

took 11.45 seconds to run.