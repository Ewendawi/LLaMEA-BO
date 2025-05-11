# Description
**EHBBO_AFLS_v4: Enhanced Hybrid Bayesian Optimization with Adaptive Acquisition, Focused Local Search, Dynamic Temperature Scaling with Memory, and Kernel Adaptation (EHBBO-AFLS-DTS-M-KA).** This algorithm builds upon EHBBO_AFLS_v3 by incorporating a dynamic kernel adaptation strategy for the Gaussian Process model. The kernel adaptation involves periodically re-estimating the kernel hyperparameters (length_scale) based on the observed data and optimization progress. This allows the model to better capture the underlying function landscape, especially when dealing with non-stationary or multi-modal functions. The algorithm also includes a refined memory management strategy, prioritizing the storage of diverse and promising points.

# Justification
The key improvements in this version are:

1.  **Kernel Adaptation:** The original algorithm uses a fixed Matern kernel. Adapting the kernel hyperparameters, particularly the length scale, allows the GP to better model functions with varying degrees of smoothness. A shorter length scale allows the GP to fit more complex functions, while a longer length scale enforces smoothness. The length scale is optimized periodically using the data observed so far.
2.  **Refined Memory Management:** Instead of simply keeping the best `memory_size` points, the memory is now managed to maintain a balance between good function values and diversity in the search space. This is achieved by penalizing points that are too close to each other in the input space, encouraging the exploration of different regions.
3. **Acquisition Function Balance:** Instead of a simple linear combination of Thompson Sampling and Expected Improvement, the weighting `alpha` is now dynamically adjusted based on the uncertainty of the GP model. Higher uncertainty leads to more exploration via Thompson Sampling, while lower uncertainty favors exploitation via Expected Improvement.

These changes aim to improve the algorithm's ability to adapt to different function landscapes, balance exploration and exploitation, and make better use of the available function evaluations.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances

class EHBBO_AFLS_v4:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * (dim + 1) # Number of initial samples, increased for higher dimensions
        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.alpha = 0.5 # Initial weight for Thompson Sampling
        self.memory_X = None
        self.memory_y = None
        self.memory_size = 50  # Maximum size of the memory
        self.kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, nu=2.5)
        self.kernel_optim_interval = 10 # Adapt kernel every n iterations

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

        # Augment data with memory
        if self.memory_X is not None:
            X = np.vstack((X, self.memory_X))
            y = np.vstack((y, self.memory_y))

        # Kernel Adaptation
        if self.n_evals % self.kernel_optim_interval == 0 and self.n_evals > self.n_init:
            try:
                # Optimize kernel hyperparameters
                kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-2, 1e2))
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)
                gp.fit(X, y)
                self.kernel = gp.kernel_
            except Exception as e:
                print(f"Kernel optimization failed: {e}")

        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=0, alpha=1e-6)
        self.gp.fit(X, y)
        return self.gp

    def _expected_improvement(self, X):
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)
        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei.reshape(-1, 1)

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)

        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, np.inf)

        # Thompson Sampling
        fraction_evaluated = self.n_evals / self.budget
        temperature = (1.0 / (1.0 + np.exp(- (self.best_y + 1e-9)))) * (1 - fraction_evaluated) + 0.01 # Dynamic temperature
        thompson_samples = np.random.normal(mu, temperature * sigma).reshape(-1, 1)

        # Expected Improvement
        ei = self._expected_improvement(X)

        # Adaptive weighting based on GP uncertainty
        mean_sigma = np.mean(sigma)
        self.alpha = 0.5 * (1 + np.tanh(mean_sigma)) # Scale alpha based on uncertainty.
        acquisition = self.alpha * thompson_samples + (1 - self.alpha) * ei
        return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Focused Local Search around best point
        x_next = []
        remaining_budget_fraction = 1 - (self.n_evals / self.budget)
        local_search_radius = 0.15 * remaining_budget_fraction # Adaptive radius, slower decay

        # Memory-informed initialization
        if self.memory_X is not None:
            distances = np.linalg.norm(self.memory_X - self.best_x, axis=1)
            within_radius = distances < local_search_radius
            if np.any(within_radius):
                best_memory_index = np.argmin(self.memory_y[within_radius])
                x_start = self.memory_X[within_radius][best_memory_index]
            else:
                x_start = self.best_x + np.random.normal(0, local_search_radius, self.dim)
                x_start = np.clip(x_start, self.bounds[0], self.bounds[1])
        else:
            x_start = self.best_x + np.random.normal(0, local_search_radius, self.dim)  # Perturb best point
            x_start = np.clip(x_start, self.bounds[0], self.bounds[1]) # Clip to bounds

        for _ in range(batch_size):
            res = minimize(lambda x: -self._acquisition_function(x.reshape(1, -1))[0,0],
                           x_start,
                           bounds=[(self.bounds[0][i], self.bounds[1][i]) for i in range(self.dim)],
                           method='L-BFGS-B',
                           options={'maxiter': 20})  # Limit iterations
            x_next.append(res.x)
            x_start = self.best_x + np.random.normal(0, local_search_radius, self.dim)
            x_start = np.clip(x_start, self.bounds[0], self.bounds[1])
        return np.array(x_next)

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

        # Update memory with diversity
        if self.memory_X is None:
            self.memory_X = new_X
            self.memory_y = new_y
        else:
            # Calculate distances to existing memory points
            distances = euclidean_distances(new_X, self.memory_X)
            min_distances = np.min(distances, axis=1)

            # Add new points to memory, penalizing points that are too close
            for i in range(len(new_X)):
                if min_distances[i] > 0.01:  # Minimum distance threshold
                    self.memory_X = np.vstack((self.memory_X, new_X[i]))
                    self.memory_y = np.vstack((self.memory_y, new_y[i]))

        # Memory management: keep only the best 'memory_size' points, considering diversity
        if len(self.memory_X) > self.memory_size:
            # Calculate distances between all memory points
            distances = euclidean_distances(self.memory_X, self.memory_X)
            np.fill_diagonal(distances, np.inf)  # Avoid self-comparison
            min_distances = np.min(distances, axis=1)

            # Calculate a score that combines function value and diversity
            scores = self.memory_y.flatten() - 0.01 * min_distances # Diversity bonus

            indices = np.argsort(scores)[:self.memory_size]
            self.memory_X = self.memory_X[indices]
            self.memory_y = self.memory_y[indices]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Optimization loop
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Select next points by acquisition function
            batch_size = min(self.budget - self.n_evals, max(1, self.dim // 2)) # Adaptive batch size
            X_next = self._select_next_points(batch_size)

            # Evaluate the points
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm EHBBO_AFLS_v4 got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1481 with standard deviation 0.0990.

took 506.36 seconds to run.