# Description
**GARBO: Gradient-Aware Adaptive Regularized Bayesian Optimization with Trust Region:** This algorithm combines gradient information, adaptive regularization, and a trust region approach to enhance Bayesian optimization. It uses a Gaussian Process Regression (GPR) model with a Matérn kernel. The acquisition function combines Expected Improvement (EI), a gradient-based exploration term, and an adaptive regularization term within a trust region. The trust region size is adaptively adjusted based on model agreement. A dynamic batch size strategy is also incorporated.

# Justification
This algorithm aims to leverage the strengths of AGATBO and ARBO while addressing their limitations. AGATBO incorporates gradient information, which can be valuable for navigating the search space more efficiently, but it's computationally expensive and can get stuck in local optima. ARBO uses adaptive regularization and dynamic batch sizing to balance exploration and exploitation. GARBO combines these elements to create a more robust and efficient optimization process.

Specifically:

*   **Gradient-Awareness:** The gradient-based exploration term from AGATBO is included to guide the search towards promising regions. However, to reduce the computational cost, the gradient is estimated using a finite difference method only when the model uncertainty is high.
*   **Adaptive Regularization:** The adaptive regularization term from ARBO is used to prevent overfitting and encourage exploration in uncertain regions. This helps to balance exploration and exploitation.
*   **Trust Region:** A trust region approach, similar to AGATBO, is used to constrain the search space and improve the reliability of the GPR model. The trust region size is adaptively adjusted based on the agreement between the GPR model and the true objective function.
*   **Dynamic Batch Size:** The batch size is dynamically adjusted based on the iteration number, starting with a larger batch size for exploration and decreasing it as the algorithm progresses to focus on exploitation, similar to ARBO.
*   **Matern Kernel:** A Matern kernel is used for the GPR model, as in AGATBO and ATRBO, which is known to be effective for a wide range of optimization problems.
*   **Halton Sequence:** Halton sequence is used for initial exploration, and Sobol sequence is used for sampling within the trust region.

By combining these elements, GARBO aims to achieve a better balance between exploration and exploitation, leading to improved performance on the BBOB test suite.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
import warnings

class GARBO:
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
        self.gradient_weight = 0.01
        self.reg_weight = 0.1 # Initial weight for the regularization term
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 2.0
        self.model_agreement_threshold = 0.75
        self.kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        self.model = None
        self.exploration_factor = 0.01

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.Halton(d=self.dim, seed=42)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        try:
            model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5, alpha=1e-5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)
            self.kernel = model.kernel_  # Update kernel with optimized parameters
            return model
        except Exception as e:
            print(f"GP fitting failed: {e}. Returning None.")
            return None

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

        # Add gradient-based exploration term
        if self.X is not None:
            # Only estimate gradient when uncertainty is high
            if np.mean(sigma) > 0.1:
                dmu_dx = self._predict_gradient(X, model)
                gradient_norm = np.linalg.norm(dmu_dx, axis=1, keepdims=True)
                ei = ei + self.gradient_weight * gradient_norm

        # Adaptive Regularization
        reg_weight = self.reg_weight * (1 - (iteration / self.budget))
        regularization_term = -reg_weight * np.linalg.norm(mu / (sigma + 1e-6), axis=1, keepdims=True)**2 # Uncertainty aware regularization
        ei = ei + regularization_term + self.exploration_factor * sigma # Add exploration factor

        return ei

    def _predict_gradient(self, X, model):
        # Predict the gradient of the GPR mean function
        # return array of shape (n_points, n_dims)
        X = np.atleast_2d(X)
        dmu_dx = np.zeros((X.shape[0], self.dim))
        for i in range(self.dim):
            def obj(x):
                x_prime = x.copy()
                x_prime[i] += 1e-6
                return model.predict(x_prime.reshape(1, -1))[0]
            def obj0(x):
                return model.predict(x.reshape(1, -1))[0]
            dmu_dx[:, i] = (np.array([obj(x) for x in X]) - np.array([obj0(x) for x in X]))/1e-6
        return dmu_dx

    def _select_next_points(self, batch_size, trust_region_center, model, iteration):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # return array of shape (batch_size, n_dims)

        # Sample within the trust region using Sobol sequence
        sobol_engine = qmc.Sobol(d=self.dim, seed=42)
        samples = sobol_engine.random(n=100 * batch_size)

        # Scale and shift samples to be within the trust region
        scaled_samples = trust_region_center + self.trust_region_radius * (2 * samples - 1)

        # Clip samples to stay within the problem bounds
        scaled_samples = np.clip(scaled_samples, self.bounds[0], self.bounds[1])

        # Calculate acquisition function values
        if self.model is None:
            return scaled_samples[:batch_size]
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
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        trust_region_center = self.X[np.argmin(self.y)] # Initialize trust region center
        iteration = self.n_init
        while self.n_evals < self.budget:
            # Adaptive batch size
            batch_size = max(1, int(5 * (1 - self.n_evals / self.budget))) # Linearly decreasing batch size

            # Optimization
            self.model = self._fit_model(self.X, self.y)
            if self.model is None:
                print("GP model fitting failed. Returning current best.")
                return self.best_y, self.best_x

            # select points by acquisition function
            next_X = self._select_next_points(batch_size, trust_region_center, self.model, iteration)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Model agreement check (simplified)
            predicted_y = self.model.predict(next_X)
            agreement = np.corrcoef(next_y.flatten(), predicted_y.flatten())[0, 1]

            # Adjust trust region size
            if np.isnan(agreement) or agreement < self.model_agreement_threshold:
                self.trust_region_radius *= self.trust_region_shrink_factor
            else:
                self.trust_region_radius *= self.trust_region_expand_factor
                self.trust_region_radius = min(self.trust_region_radius, 5.0) # Limit expansion

            # Update trust region center
            trust_region_center = self.X[np.argmin(self.y)]
            iteration += batch_size

        return self.best_y, self.best_x
```
## Feedback
 The algorithm GARBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1707 with standard deviation 0.1091.

took 567.17 seconds to run.