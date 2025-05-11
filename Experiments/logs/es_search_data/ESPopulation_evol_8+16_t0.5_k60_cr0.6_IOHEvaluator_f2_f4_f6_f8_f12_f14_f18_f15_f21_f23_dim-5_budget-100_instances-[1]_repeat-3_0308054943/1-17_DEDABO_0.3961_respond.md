# Description
**DEDABO: Diversity Enhanced Distribution-Aware Bayesian Optimization:** This algorithm combines the strengths of DEBO (Diversity Enhanced Bayesian Optimization) and DABO (Distribution-Aware Bayesian Optimization). It uses a Gaussian Process Regression (GPR) model with an RBF kernel for surrogate modeling. The acquisition function is a hybrid of Expected Improvement (EI), a distance-based diversity term (from DEBO), and a distribution matching term using Kernel Density Estimation (KDE) (from DABO). The diversity term encourages exploration of regions far from previously evaluated points, while the distribution matching term focuses on sampling from regions identified as promising by the KDE. The algorithm uses Sobol sampling for initial exploration and KDE-guided sampling for subsequent iterations, enhanced by a diversity-promoting mechanism. The KDE is updated dynamically based on the top-performing samples. The error in DABO was due to a dimension mismatch when using boolean indexing. This is addressed by ensuring that the boolean index and the indexed array have compatible dimensions.

# Justification
The combination of DEBO and DABO aims to leverage the benefits of both diversity enhancement and distribution awareness. The diversity term helps prevent premature convergence by encouraging exploration, while the distribution matching term focuses the search on promising regions. The use of KDE allows the algorithm to adapt to the underlying structure of the objective function.

The error in DABO's `_update_eval_points` function arose because `self.y` has shape (n_points, 1), while the comparison `self.y <= threshold` results in a boolean array of shape (n_points, 1). When this boolean array is used to index `self.X`, which has shape (n_points, dim), a dimension mismatch occurs. To fix this, we need to flatten the boolean array before using it for indexing.

The initial exploration uses Sobol sampling to ensure good coverage of the search space. Subsequent sampling is guided by the KDE, which is updated dynamically based on the top-performing samples. The diversity term is added to the acquisition function to further encourage exploration.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KernelDensity
from sklearn.metrics import pairwise_distances

class DEDABO:
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
        self.kde = None
        self.best_x = None
        self.best_y = float('inf')
        self.distribution_weight = 0.1
        self.diversity_weight = 0.1

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, method='sobol'):
        # sample points
        # return array of shape (n_points, n_dims)
        if method == 'sobol':
            sampler = qmc.Sobol(d=self.dim, seed=42)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        elif method == 'kde':
            if self.kde is None:
                return self._sample_points(n_points, method='sobol')
            else:
                # Sample from KDE
                samples = self.kde.sample(n_points)
                samples = np.clip(samples, self.bounds[0], self.bounds[1])
                return samples
        else:
            raise ValueError("Invalid sampling method.")

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
            ei = np.zeros_like(mu)
        else:
            imp = self.best_y - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0  # handle zero sigma

        # Add distribution matching term
        if self.kde is not None:
            log_likelihood = self.kde.score_samples(X).reshape(-1, 1)
            ei = ei + self.distribution_weight * np.exp(log_likelihood) # Use exp to avoid negative values

        # Add diversity term
        if self.X is not None:
            distances = pairwise_distances(X, self.X)
            min_distances = np.min(distances, axis=1).reshape(-1, 1)
            ei = ei + self.diversity_weight * min_distances

        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # return array of shape (batch_size, n_dims)
        
        # Sample candidate points using KDE
        candidate_points = self._sample_points(100 * batch_size, method='kde')

        # Calculate acquisition function values
        model = self._fit_model(self.X, self.y)
        acquisition_values = self._acquisition_function(candidate_points, model)

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

        # Update KDE
        if self.X is not None:
            # Identify promising regions (e.g., top 20% of evaluated points)
            threshold = np.percentile(self.y, 20)
            promising_points = self.X[(self.y <= threshold).flatten()]

            if len(promising_points) > self.dim + 1:  # Ensure enough points for KDE
                self.kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(promising_points)
            else:
                self.kde = None
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial sampling
        initial_X = self._sample_points(self.n_init, method='sobol')
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        
        batch_size = 5
        while self.n_evals < self.budget:
            # Optimization
            model = self._fit_model(self.X, self.y)
            
            # select points by acquisition function
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm DEDABO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1679 with standard deviation 0.1015.

took 92.85 seconds to run.