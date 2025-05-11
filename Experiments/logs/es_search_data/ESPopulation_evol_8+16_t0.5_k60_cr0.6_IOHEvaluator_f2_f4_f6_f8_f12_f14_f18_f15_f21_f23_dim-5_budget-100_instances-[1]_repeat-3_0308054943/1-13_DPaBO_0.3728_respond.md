# Description
**DPaBO: Diversity-enhanced Pareto Bayesian Optimization:** This algorithm combines the strengths of DEBO and PaSBO to achieve a better balance between exploration and exploitation. It uses a Gaussian Process Regression (GPR) model with an RBF kernel for surrogate modeling. The acquisition function leverages a Pareto-based approach with Expected Improvement (EI), Probability of Improvement (PI), and Upper Confidence Bound (UCB) to generate a Pareto front of candidate points. A diversity term, similar to DEBO, is then incorporated into the Pareto front selection process. This term encourages the selection of points that are distant from previously evaluated points, promoting exploration of diverse regions in the search space. The initial exploration is performed using a Sobol sequence. An adaptive mechanism adjusts the diversity weight during the optimization process.

# Justification
The algorithm combines the Pareto-based multi-objective acquisition from PaSBO with the diversity enhancement from DEBO.
*   **Pareto-based Acquisition:** PaSBO's use of multiple acquisition functions (EI, PI, UCB) and Pareto selection provides a more robust exploration strategy than relying on a single acquisition function. This helps to avoid premature convergence to local optima.
*   **Diversity Enhancement:** DEBO's diversity term encourages exploration of less-visited regions. By incorporating this term into the Pareto front selection, we can guide the algorithm towards a more diverse set of candidate points.
*   **Adaptive Diversity Weight:** The diversity weight is adapted during the optimization process. This allows the algorithm to dynamically adjust the balance between exploration and exploitation, depending on the characteristics of the objective function and the current state of the search.
*   **Sobol Initial Sampling:** Sobol sequence provides better space-filling properties compared to LHS, which is used in PaSBO.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import pairwise_distances

class DPaBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim
        self.acquisition_functions = ['ei', 'pi', 'ucb']
        self.ucb_kappa = 2.0
        self.diversity_weight = 0.1
        self.best_x = None
        self.best_y = float('inf')

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(d=self.dim, seed=42)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X, model, acq_type='ei'):
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        if self.best_y is None:
            return np.zeros_like(mu)

        imp = self.best_y - mu
        Z = imp / sigma

        if acq_type == 'ei':
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma <= 1e-6] = 0.0
            return ei
        elif acq_type == 'pi':
            pi = norm.cdf(Z)
            return pi
        elif acq_type == 'ucb':
            ucb = mu + self.ucb_kappa * sigma
            return ucb
        else:
            raise ValueError("Invalid acquisition function type.")

    def _is_pareto_efficient(self, points):
        is_efficient = np.ones(points.shape[0], dtype=bool)
        for i, c in enumerate(points):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(points[is_efficient] > c, axis=1) | (points[is_efficient] == c).all(axis=1)
                is_efficient[i] = True
        return is_efficient

    def _select_next_points(self, batch_size, model):
        candidate_points = self._sample_points(100 * batch_size)
        acquisition_values = np.zeros((candidate_points.shape[0], len(self.acquisition_functions)))

        for i, acq_type in enumerate(self.acquisition_functions):
            acquisition_values[:, i] = self._acquisition_function(candidate_points, model, acq_type).flatten()

        is_efficient = self._is_pareto_efficient(acquisition_values)
        pareto_points = candidate_points[is_efficient]

        if self.X is not None:
            distances = pairwise_distances(pareto_points, self.X)
            min_distances = np.min(distances, axis=1)
            # Normalize distances to [0, 1]
            normalized_distances = (min_distances - np.min(min_distances)) / (np.max(min_distances) - np.min(min_distances) + 1e-9)
            # Add diversity score to Pareto points
            diversity_scores = self.diversity_weight * normalized_distances
            # Combine acquisition values and diversity scores
            combined_scores = np.sum(acquisition_values[is_efficient], axis=1) + diversity_scores

            # Select top batch_size points based on combined scores
            indices = np.argsort(-combined_scores)[:batch_size]
            selected_points = pareto_points[indices]
        else:
            if len(pareto_points) > batch_size:
                indices = np.random.choice(len(pareto_points), batch_size, replace=False)
                selected_points = pareto_points[indices]
            else:
                selected_points = pareto_points
                if len(selected_points) < batch_size:
                    remaining = batch_size - len(selected_points)
                    random_points = self._sample_points(remaining)
                    selected_points = np.vstack((selected_points, random_points))

        return selected_points

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X]).reshape(-1, 1)
        self.n_evals += len(X)
        return y
    
    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))
        
        idx = np.argmin(self.y)
        if self.y[idx][0] < self.best_y:
            self.best_y = self.y[idx][0]
            self.best_x = self.X[idx]
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        
        batch_size = 5
        while self.n_evals < self.budget:
            model = self._fit_model(self.X, self.y)
            next_X = self._select_next_points(batch_size, model)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Adaptive diversity weight
            self.diversity_weight *= 0.95  # Reduce diversity weight over time

        return self.best_y, self.best_x
```
## Feedback
 The algorithm DPaBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1483 with standard deviation 0.0941.

took 90.08 seconds to run.