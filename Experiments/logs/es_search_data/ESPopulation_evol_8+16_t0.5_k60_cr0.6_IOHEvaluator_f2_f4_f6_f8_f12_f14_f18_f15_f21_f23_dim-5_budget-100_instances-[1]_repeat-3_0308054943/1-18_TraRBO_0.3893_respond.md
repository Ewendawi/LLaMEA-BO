# Description
**TraRBO: Trust Region and Regularized Bayesian Optimization:** This algorithm combines the strengths of Adaptive Trust Region Bayesian Optimization (ATRBO) and Regularized Bayesian Optimization (ReBO). It uses a Gaussian Process Regression (GPR) model with a Matérn kernel for surrogate modeling, similar to ATRBO, but incorporates the adaptive regularization term from ReBO into the Expected Improvement (EI) acquisition function. The trust region mechanism adaptively adjusts the search space based on model agreement, while the regularization term prevents overfitting and encourages exploration. This combination aims to balance exploration and exploitation more effectively, leading to improved performance. An adaptive mechanism for the trust region radius and regularization weight is also included.

# Justification
The combination of trust regions and regularization offers a robust approach to Bayesian Optimization. The trust region helps to focus the search on areas where the model is likely to be accurate, while regularization prevents the model from becoming overly confident and encourages exploration in uncertain regions.

*   **Matérn Kernel:** The Matérn kernel (from ATRBO) is chosen for its flexibility in modeling functions with varying degrees of smoothness.
*   **Adaptive Trust Region:** The trust region radius is adjusted based on the agreement between the predicted and actual function values, allowing the algorithm to adapt to the local landscape of the objective function.
*   **Regularized Acquisition Function:** The Expected Improvement (EI) acquisition function is augmented with a regularization term that penalizes solutions with high predicted values, encouraging exploration in regions with high uncertainty. The regularization weight is adaptively adjusted over time, gradually decreasing as the algorithm converges.
*   **Sobol Sampling within Trust Region:** Sobol sequences are used to sample points within the trust region, ensuring good coverage of the search space.
*   **Computational Efficiency:** The algorithm is designed to be computationally efficient by using a relatively small batch size and avoiding expensive computations such as gradient calculations.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

class TraRBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0  # the number of function evaluations
        self.n_init = 2 * dim
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 2.0
        self.model_agreement_threshold = 0.75
        self.reg_weight = 0.1  # Initial weight for the regularization term
        self.best_x = None
        self.best_y = float('inf')

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
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-5)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X, model, iteration):
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

        # Adaptive Regularization
        reg_weight = self.reg_weight * (1 - (iteration / self.budget))
        regularization_term = -reg_weight * np.linalg.norm(mu, axis=1, keepdims=True)**2
        ei = ei + regularization_term

        return ei

    def _select_next_points(self, batch_size, model, trust_region_center, iteration):
        # Select the next points to evaluate
        # Sample within the trust region using Sobol sequence
        sobol_engine = qmc.Sobol(d=self.dim, seed=42)
        samples = sobol_engine.random(n=100 * batch_size)

        # Scale and shift samples to be within the trust region
        scaled_samples = trust_region_center + self.trust_region_radius * (2 * samples - 1)

        # Clip samples to stay within the problem bounds
        scaled_samples = np.clip(scaled_samples, self.bounds[0], self.bounds[1])

        # Calculate acquisition function values
        acquisition_values = self._acquisition_function(scaled_samples, model, iteration)

        # Select top batch_size points based on acquisition function values
        indices = np.argsort(-acquisition_values.flatten())[:batch_size]
        selected_points = scaled_samples[indices]

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

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        trust_region_center = self.X[np.argmin(self.y)]  # Initialize trust region center

        batch_size = 5
        iteration = self.n_init
        while self.n_evals < self.budget:
            # Optimization
            model = self._fit_model(self.X, self.y)

            # select points by acquisition function
            next_X = self._select_next_points(batch_size, model, trust_region_center, iteration)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Model agreement check (simplified)
            predicted_y = model.predict(next_X)
            agreement = np.corrcoef(next_y.flatten(), predicted_y.flatten())[0, 1]

            # Adjust trust region size
            if np.isnan(agreement) or agreement < self.model_agreement_threshold:
                self.trust_region_radius *= self.trust_region_shrink_factor
            else:
                self.trust_region_radius *= self.trust_region_expand_factor
                self.trust_region_radius = min(self.trust_region_radius, 5.0)  # Limit expansion

            # Update trust region center
            trust_region_center = self.X[np.argmin(self.y)]
            iteration += batch_size

        return self.best_y, self.best_x
```
## Feedback
 The algorithm TraRBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1634 with standard deviation 0.0996.

took 97.87 seconds to run.