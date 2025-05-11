# Description
**AGDEDABO: Adaptive Gradient, Diversity Enhanced, Distribution-Aware Bayesian Optimization:** This algorithm integrates gradient information, diversity enhancement, and distribution awareness within a trust region framework. It combines the strengths of AGATBO and DEDABO. It adaptively adjusts the trust region based on model agreement, and uses a hybrid acquisition function that incorporates Expected Improvement (EI), gradient-based exploration, a distance-based diversity term, and a distribution matching term using Kernel Density Estimation (KDE). The algorithm uses Sobol sampling for initial exploration and KDE-guided sampling within the trust region, enhanced by a diversity-promoting mechanism. The gradient is estimated using finite differences.

# Justification
The algorithm is designed to leverage the strengths of both AGATBO and DEDABO.

*   **Gradient Information:** Incorporating gradient information, as in AGATBO, can accelerate convergence by guiding the search towards promising regions.
*   **Diversity Enhancement:** The diversity term, borrowed from DEDABO, helps to prevent premature convergence and encourages exploration of different regions of the search space.
*   **Distribution Awareness:** The KDE-based distribution matching term, also from DEDABO, focuses on sampling from regions identified as promising based on the distribution of previously evaluated points.
*   **Trust Region:** The trust region approach, from AGATBO, helps to ensure the reliability of the surrogate model and to control the step size.
*   **Adaptive Trust Region:** Adaptively adjusting the trust region size based on model agreement allows the algorithm to balance exploration and exploitation.
*   **Sobol and KDE Sampling:** Using Sobol sampling for initial exploration and KDE-guided sampling within the trust region combines the benefits of good space-filling properties with focused sampling in promising regions.
*   **Computational Efficiency:** The gradient estimation is done using finite differences, which is computationally efficient.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C
from sklearn.neighbors import KernelDensity
from sklearn.metrics import pairwise_distances
import warnings

class AGDEDABO:
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
        self.diversity_weight = 0.01
        self.distribution_weight = 0.01
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0  # Initial trust region radius
        self.trust_region_shrink_factor = 0.5
        self.trust_region_expand_factor = 2.0
        self.model_agreement_threshold = 0.75
        self.kde = None
        self.kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        self.model = None

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points, method='sobol', trust_region_center=None):
        # sample points
        # return array of shape (n_points, n_dims)
        if method == 'sobol':
            sampler = qmc.Sobol(d=self.dim, seed=42)
            sample = sampler.random(n=n_points)
            scaled_samples = qmc.scale(sample, self.bounds[0], self.bounds[1])
            return scaled_samples
        elif method == 'kde':
            if self.kde is None:
                return self._sample_points(n_points, method='sobol')
            else:
                # Sample from KDE within the trust region
                samples = self.kde.sample(n_points)
                if trust_region_center is not None:
                    samples = np.clip(samples, trust_region_center - self.trust_region_radius, trust_region_center + self.trust_region_radius)
                samples = np.clip(samples, self.bounds[0], self.bounds[1])
                return samples
        else:
            raise ValueError("Invalid sampling method.")

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

        # Add gradient-based exploration term
        if self.X is not None:
            dmu_dx = self._predict_gradient(X, model)
            gradient_norm = np.linalg.norm(dmu_dx, axis=1, keepdims=True)
            ei = ei + self.gradient_weight * gradient_norm

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

    def _select_next_points(self, batch_size, trust_region_center):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # return array of shape (batch_size, n_dims)

        # Sample candidate points using KDE within the trust region
        candidate_points = self._sample_points(100 * batch_size, method='kde', trust_region_center=trust_region_center)

        # Clip samples to stay within the problem bounds and trust region
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])
        candidate_points = np.clip(candidate_points, trust_region_center - self.trust_region_radius, trust_region_center + self.trust_region_radius)

        # Calculate acquisition function values
        if self.model is None:
            return candidate_points[:batch_size]
        acquisition_values = self._acquisition_function(candidate_points, self.model)

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
                try:
                    self.kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(promising_points)
                except Exception as e:
                    print(f"KDE fitting failed: {e}. Setting KDE to None.")
                    self.kde = None
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

        trust_region_center = self.X[np.argmin(self.y)] # Initialize trust region center
        batch_size = 5
        while self.n_evals < self.budget:
            # Optimization
            self.model = self._fit_model(self.X, self.y)
            if self.model is None:
                print("GP model fitting failed. Returning current best.")
                return self.best_y, self.best_x

            # select points by acquisition function
            next_X = self._select_next_points(batch_size, trust_region_center)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Model agreement check (simplified)
            if self.model is not None:
                predicted_y = self.model.predict(next_X)
                agreement = np.corrcoef(next_y.flatten(), predicted_y.flatten())[0, 1]
            else:
                agreement = 0.0

            # Adjust trust region size
            if np.isnan(agreement) or agreement < self.model_agreement_threshold:
                self.trust_region_radius *= self.trust_region_shrink_factor
            else:
                self.trust_region_radius *= self.trust_region_expand_factor
                self.trust_region_radius = min(self.trust_region_radius, 5.0) # Limit expansion

            # Update trust region center
            trust_region_center = self.X[np.argmin(self.y)]

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AGDEDABO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1641 with standard deviation 0.1083.

took 500.71 seconds to run.