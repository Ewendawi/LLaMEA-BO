# Description
**Adaptive Density-Guided Hybrid Bayesian Optimization (ADGHBO):** This algorithm combines the strengths of AHBBO and DensiTreeBO with adaptive components for enhanced exploration and exploitation. It uses a Gaussian Process Regression (GPR) model with a hybrid acquisition function (Expected Improvement + Distance-based Exploration) from AHBBO, incorporating an adaptive exploration weight. It integrates Kernel Density Estimation (KDE) from DensiTreeBO to focus the search on high-density regions of promising solutions and adaptively adjusts the KDE bandwidth based on the data distribution. The key novelty lies in adaptively balancing the exploration-exploitation trade-off through both the AHBBO's exploration weight and the KDE bandwidth adjustment, guided by the optimization progress and the observed function landscape.

# Justification
The algorithm builds upon the strengths of AHBBO and DensiTreeBO while addressing their limitations. AHBBO's adaptive exploration weight helps to avoid premature convergence by maintaining a minimum level of exploration. DensiTreeBO's KDE focuses the search on promising high-density regions. By combining these techniques and making the KDE bandwidth adaptive, the algorithm can more effectively navigate complex and multimodal search spaces.

The adaptive KDE bandwidth is crucial. A fixed bandwidth might be too broad or too narrow, leading to either over-exploration or premature convergence. By dynamically adjusting the bandwidth based on the data distribution, the KDE can better capture the underlying structure of the search space.

The hybrid acquisition function, combining Expected Improvement and a distance-based exploration term, further enhances the exploration-exploitation balance. The distance-based exploration term encourages the algorithm to explore regions that are far from previously evaluated points, while Expected Improvement focuses on regions with high potential for improvement.

The initial sampling using Latin Hypercube Sampling (LHS) ensures good space coverage, providing a solid foundation for the subsequent optimization process.

The batch selection strategy selects multiple points for evaluation in each iteration, improving the efficiency of the optimization process.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

class ADGHBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * dim # Initial number of points

        self.best_y = np.inf
        self.best_x = None

        self.batch_size = min(10, dim) # Batch size for selecting points
        self.exploration_weight = 0.1 # Initial exploration weight
        self.exploration_decay = 0.995 # Decay factor for exploration weight
        self.min_exploration = 0.01 # Minimum exploration weight
        self.kde_bandwidth = 0.5 # Initial bandwidth for KDE
        self.kde_bandwidth_decay = 0.995 # Decay factor for KDE bandwidth
        self.min_kde_bandwidth = 0.1 # Minimum KDE bandwidth

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
            exploration = np.ones((X.shape[0], 1))

        # Hybrid acquisition function
        acquisition = ei + self.exploration_weight * exploration
        return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Generate candidate points
        candidate_points = self._sample_points(100 * batch_size)
        
        if self.X is None or len(self.X) < self.dim + 1:
            # Not enough data for KDE, return random samples
            return candidate_points[:batch_size]

        # Fit KDE to the evaluated points
        kde = KernelDensity(bandwidth=self.kde_bandwidth).fit(self.X)

        # Calculate KDE scores for candidate points
        kde_scores = kde.score_samples(candidate_points)

        # Select top candidate points based on KDE scores
        top_indices = np.argsort(kde_scores)[-batch_size:]
        next_points = candidate_points[top_indices]

        # Refine selection using acquisition function
        acquisition_values = self._acquisition_function(next_points)
        best_index = np.argmax(acquisition_values)
        
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
            batch_size = min(self.batch_size, remaining_evals) # Adjust batch size to budget
            
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self.model = self._fit_model(self.X, self.y)
            
            # Update exploration weight
            self.exploration_weight = max(self.exploration_weight * self.exploration_decay, self.min_exploration)
            # Update KDE bandwidth
            self.kde_bandwidth = max(self.kde_bandwidth * self.kde_bandwidth_decay, self.min_kde_bandwidth)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ADGHBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1326 with standard deviation 0.1078.

took 39.86 seconds to run.