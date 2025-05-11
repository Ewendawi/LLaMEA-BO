# Description
**Adaptive Exploration with Dynamic Distance Scaling and Local Refinement Bayesian Optimization (AEDDSLBO):** This algorithm refines AEDDSBO by incorporating a local refinement step after selecting the next points using the acquisition function. This local refinement aims to improve the exploitation of promising regions. The refinement involves perturbing the selected points using a Gaussian distribution centered at the selected points, with the standard deviation dynamically adjusted based on the GPR's variance predictions. This allows for a more focused search around promising regions while still maintaining exploration capabilities through the dynamic distance scaling and exploration weight decay mechanisms. Additionally, the initial exploration weight is set higher and decays faster, and the minimum exploration is set to a lower value to encourage exploration in the early stages and exploitation in later stages.

# Justification
The key improvements are:
1.  **Local Refinement:** Adding a local refinement step helps to exploit promising regions more effectively. By perturbing the selected points based on the GPR's variance, the algorithm can fine-tune the search and potentially find better solutions in the vicinity of the selected points. The variance is used to control the step size, allowing for larger steps in uncertain regions and smaller steps in confident regions.
2.  **Exploration-Exploitation Balance:** The initial exploration weight is increased, and the decay is also increased to allow for more exploration at the beginning and more exploitation at the end. The minimum exploration is also lowered to allow for more exploitation at the end.
3.  **Computational Efficiency:** The local refinement is implemented efficiently using NumPy operations, minimizing the computational overhead. The Gaussian perturbations are generated using `np.random.normal`, which is highly optimized.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class AEDDSLBO:
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
        self.exploration_weight = 0.2 # Initial exploration weight
        self.exploration_decay = 0.99 # Decay factor for exploration weight
        self.min_exploration = 0.005 # Minimum exploration weight

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
        distances = np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2)
        min_dist = np.min(distances, axis=1, keepdims=True)

        # Dynamic distance scaling
        median_distance = np.median(distances)
        scaled_exploration = min_dist / (median_distance + 1e-6)  # Avoid division by zero

        # Hybrid acquisition function
        acquisition = ei + self.exploration_weight * scaled_exploration
        return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        
        # Generate candidate points
        candidate_points = self._sample_points(100 * batch_size)  # Generate more candidates
        
        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(candidate_points)
        
        # Select the top batch_size points with the highest acquisition values
        indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        next_points = candidate_points[indices]

        # Local Refinement
        refined_points = []
        _, var = self.model.predict(next_points, return_std=True)
        var = np.clip(var, 1e-6, None)  # Ensure variance is not zero
        for i in range(batch_size):
            # Perturb each point with Gaussian noise scaled by the variance
            perturbation = np.random.normal(0, np.sqrt(var[i]), size=self.dim)
            refined_point = next_points[i] + perturbation
            refined_point = np.clip(refined_point, self.bounds[0], self.bounds[1])  # Clip to bounds
            refined_points.append(refined_point)

        refined_points = np.array(refined_points)
        
        return refined_points

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

            # Update exploration weight - Variance Aware
            _, var = self.model.predict(self.X, return_std=True)
            mean_var = np.mean(var)
            self.exploration_decay = 0.99 + 0.01 * np.clip(mean_var, 0, 1) #Slower decay if variance is high
            self.exploration_weight = max(self.exploration_weight * self.exploration_decay, self.min_exploration)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AEDDSLBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1509 with standard deviation 0.0969.

took 54.43 seconds to run.