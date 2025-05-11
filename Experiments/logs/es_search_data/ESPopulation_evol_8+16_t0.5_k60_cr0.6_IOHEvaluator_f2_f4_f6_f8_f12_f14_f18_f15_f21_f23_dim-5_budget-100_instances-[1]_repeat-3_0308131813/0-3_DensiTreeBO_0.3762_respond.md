# Description
**DensiTreeBO (DTBO):** This algorithm introduces a novel approach to Bayesian Optimization by employing a density-based clustering technique to identify promising regions in the search space. Instead of relying solely on the Gaussian Process Regression (GPR) model and acquisition function to select the next points, DTBO uses a density estimation method (Kernel Density Estimation - KDE) on the evaluated points to locate high-density clusters. These clusters are then used to guide the selection of new points, promoting exploration within promising areas and exploitation of the best-performing clusters. The acquisition function is used to refine the point selection within the clusters. This approach aims to improve the balance between exploration and exploitation, especially in multimodal or complex search spaces.

# Justification
1.  **Density-Based Clustering:** The core idea is to use KDE to identify regions with a high concentration of good solutions. This helps to focus the search on areas that are likely to contain the global optimum, especially in cases where the function landscape has multiple local optima.
2.  **Integration with Acquisition Function:** The acquisition function (EI in this case) is used to select the best point *within* the identified clusters. This allows the algorithm to exploit the information provided by the GPR model while still being guided by the overall density of good solutions.
3.  **Diversity from Previous Algorithms:** This algorithm differs significantly from the previous ones in its use of density estimation as a primary driver of exploration and exploitation. EHBBO uses a distance-based exploration term, ATRBO uses a trust region, and BONGIBO uses gradient-based improvement. DTBO's density-based approach offers a different way to balance exploration and exploitation, which could be beneficial for certain types of optimization problems.
4.  **Addressing Errors of Previous Algorithms:** EHBBO can sometimes get stuck in local optima due to its reliance on the EI acquisition function. ATRBO can be slow due to the trust region management and local optimization within the trust region. BONGIBO's gradient-based improvement can be sensitive to noise (even though it tries to account for it). DTBO aims to address these issues by using the density estimation to guide the search towards promising regions, which can help to escape local optima and reduce the sensitivity to noise.
5.  **Computational Efficiency:** KDE is computationally efficient, especially when using optimized libraries like scikit-learn. The algorithm also avoids expensive gradient calculations, making it faster than BONGIBO.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KernelDensity

class DensiTreeBO:
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
        self.kde_bandwidth = 0.5  # Bandwidth for KDE, can be tuned

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
        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        if self.X is None or len(self.X) < self.dim + 1:
            # Not enough data for KDE, return random samples
            return self._sample_points(batch_size)

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
            batch_size = min(self.n_init, remaining_evals)
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            self.model = self._fit_model(self.X, self.y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm DensiTreeBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1502 with standard deviation 0.1066.

took 21.60 seconds to run.