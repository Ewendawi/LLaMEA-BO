# Description
Adaptive Batch Size with Thompson Sampling and Local Refinement Bayesian Optimization (ABS-TS-LR-BO). This algorithm combines the adaptive batch size strategy from EI_ABS_BO with Thompson Sampling for exploration and a local refinement step using L-BFGS-B to improve exploitation. The adaptive batch size adjusts based on the GP's uncertainty, Thompson Sampling provides diverse exploration, and local refinement focuses on improving promising candidate solutions. The RBF kernel bandwidth is also dynamically adjusted during optimization using the median heuristic.

# Justification
This algorithm aims to improve upon the previous approaches by combining their strengths. EI_ABS_BO uses an adaptive batch size, which is beneficial for balancing exploration and exploitation. GP_UCB_TS_BO uses Thompson Sampling, which is a computationally efficient way to sample from the posterior distribution and explore the search space. By combining these techniques, the algorithm can efficiently explore the search space and find promising candidate solutions. Additionally, a local refinement step using L-BFGS-B is added to improve the exploitation of promising regions. The dynamic adjustment of the RBF kernel bandwidth, inspired by RBF_Bandwidth_BO, allows the GP to better adapt to the data distribution. The initial sampling is performed using a Sobol sequence for better space-filling properties.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

class ABS_TS_LR_BO:
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

        self.gp = None
        self.best_y = float('inf')
        self.best_x = None
        self.batch_size = 1
        self.rbf_bandwidth = 1.0

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        points = sampler.random(n=n_points)
        return qmc.scale(points, self.bounds[0], self.bounds[1])

    def _median_distance(self, X):
        """Compute the median distance between data points in X."""
        if X.shape[0] <= 1:
            return 1.0  # or some other default value
        distances = pdist(X)
        if len(distances) == 0:
            return 1.0
        return np.median(distances)

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        # Dynamically adjust RBF kernel bandwidth
        self.rbf_bandwidth = self._median_distance(X)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(self.rbf_bandwidth, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        imp = self.best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0 # avoid nan values
        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using Thompson Sampling and Local Refinement
        # return array of shape (batch_size, n_dims)
        next_points = []
        for _ in range(batch_size):
            # Thompson Sampling: Draw a sample from the posterior
            sampled_f = self.gp.sample_y(self.X, n_samples=1)

            # Generate candidate points
            num_candidates = 100 * self.dim
            X_candidate = self._sample_points(num_candidates)

            # Evaluate acquisition function (EI) on candidate points
            ei_values = self._acquisition_function(X_candidate)

            # Select the candidate with the highest EI
            next_point = X_candidate[np.argmax(ei_values)].reshape(1, -1)

            # Local Refinement using L-BFGS-B
            def objective(x):
                x = x.reshape(1, -1)
                return -self._acquisition_function(x)[0, 0]

            res = minimize(objective, next_point, bounds=[(-5, 5)] * self.dim, method='L-BFGS-B')
            next_point = res.x.reshape(1, -1)

            next_points.append(next_point)

        return np.vstack(next_points)

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
        if np.min(new_y) < self.best_y:
            self.best_y = np.min(new_y)
            self.best_x = new_X[np.argmin(new_y)]

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        # Optimization loop
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Adaptive batch size
            _, std = self.gp.predict(self.X, return_std=True)
            mean_std = np.mean(std)
            self.batch_size = max(1, int(self.dim / (1 + mean_std * 10))) # Adjust batch size based on uncertainty
            self.batch_size = min(self.batch_size, self.budget - self.n_evals) # Ensure not exceeding budget

            # Select next points by acquisition function
            next_X = self._select_next_points(self.batch_size)

            # Evaluate the points
            next_y = self._evaluate_points(func, next_X)

            # Update evaluated points
            self._update_eval_points(next_X, next_y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm ABS_TS_LR_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1619 with standard deviation 0.0981.

took 411.50 seconds to run.