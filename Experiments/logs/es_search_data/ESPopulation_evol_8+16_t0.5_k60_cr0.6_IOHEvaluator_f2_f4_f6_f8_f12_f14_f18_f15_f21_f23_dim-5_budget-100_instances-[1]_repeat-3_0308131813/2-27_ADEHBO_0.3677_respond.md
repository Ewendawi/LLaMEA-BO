# Description
**Adaptive Density-Enhanced Hybrid Bayesian Optimization (ADEHBO):** This algorithm combines the strengths of AHBBO and HyDEBO by integrating adaptive exploration, density estimation, and a hybrid acquisition function. It adaptively adjusts the exploration weight based on the optimization progress, uses Kernel Density Estimation (KDE) to focus the search on high-density regions of promising solutions, and employs a hybrid acquisition function (Expected Improvement + Distance-based Exploration) to balance exploration and exploitation. The KDE bandwidth is also dynamically adjusted based on the data distribution. Furthermore, a mechanism for dynamically adjusting the batch size based on the uncertainty of the GPR model is introduced to improve sample efficiency.

# Justification
The ADEHBO algorithm builds upon the strengths of AHBBO and HyDEBO to achieve a more robust and efficient optimization process.

1.  **Adaptive Exploration:** AHBBO's adaptive exploration strategy is incorporated to dynamically adjust the exploration weight based on the optimization progress. This allows the algorithm to shift its focus from exploration to exploitation as the number of evaluations increases, preventing premature convergence.

2.  **Density Estimation:** HyDEBO's Kernel Density Estimation (KDE) is used to focus the search on high-density regions of promising solutions. This helps the algorithm to efficiently explore the search space and identify promising regions.

3.  **Hybrid Acquisition Function:** The hybrid acquisition function (Expected Improvement + Distance-based Exploration) from HyDEBO is used to balance exploration and exploitation. This ensures that the algorithm explores the search space effectively while also exploiting the best-known solutions.

4. **Adaptive Bandwidth Selection:** The KDE bandwidth is dynamically adjusted based on the data distribution, similar to AdaptiveBandwidthDensiTreeBO, allowing the KDE to better capture the underlying density structure of the search space.

5. **Adaptive Batch Size:** The batch size is dynamically adjusted based on the uncertainty of the GPR model. When the uncertainty is high, a larger batch size is used to explore the search space more effectively. When the uncertainty is low, a smaller batch size is used to exploit the best-known solutions. This adaptive batch size strategy improves sample efficiency and allows the algorithm to converge faster.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KernelDensity

class ADEHBO:
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

        self.best_y = np.inf
        self.best_x = None
        self.kde_bandwidth = 0.5  # Bandwidth for KDE
        self.batch_size = min(10, dim)
        self.exploration_weight = 0.1 # Initial exploration weight
        self.exploration_decay = 0.995 # Decay factor for exploration weight
        self.min_exploration = 0.01 # Minimum exploration weight


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

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        # Distance-based exploration term
        if self.X is not None and len(self.X) > 0:
            min_dist = np.min(np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2), axis=1, keepdims=True)
            exploration = min_dist / np.max(min_dist)
        else:
            exploration = np.ones(X.shape[0])[:, None]

        # Hybrid acquisition function
        acquisition = ei + self.exploration_weight * exploration
        return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        if self.X is None or len(self.X) < self.dim + 1:
            # Not enough data for KDE, return random samples
            return self._sample_points(batch_size)
        
        #Adaptive Bandwidth Selection
        self.kde_bandwidth = 0.5 * np.std(self.X)

        # Fit KDE to the evaluated points
        kde = KernelDensity(bandwidth=self.kde_bandwidth).fit(self.X)

        # Generate candidate points
        candidate_points = self._sample_points(100 * batch_size)

        # Calculate KDE scores for candidate points
        kde_scores = kde.score_samples(candidate_points)

        # Select top candidate points based on KDE scores
        top_indices = np.argsort(kde_scores)[-batch_size:]
        next_points = candidate_points[top_indices]

        # Refine selection using acquisition function
        acquisition_values = self._acquisition_function(next_points)
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = next_points[indices]
        
        return next_points

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)
    
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

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            # Optimization
            # select points by acquisition function
            remaining_evals = self.budget - self.n_evals

            #Adaptive Batch Size
            mu, sigma = self.model.predict(self.X, return_std=True)
            uncertainty = np.mean(sigma)
            batch_size = min(int(self.batch_size * (1 + uncertainty)), remaining_evals)
            batch_size = max(1, batch_size) #Ensure batch_size is at least 1

            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self.model = self._fit_model(self.X, self.y)
            # Update exploration weight
            self.exploration_weight = max(self.exploration_weight * self.exploration_decay, self.min_exploration)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ADEHBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1369 with standard deviation 0.1051.

took 41.71 seconds to run.