# Description
**AdaptiveEvolutionaryTrustRegionBO with Dynamic Kappa, Adaptive DE Iterations, Enhanced Trust Region Adaptation, Kernel Bandwidth Adjustment, and Gradient-Based Exploration (AETRBO-DKAEBG):** This algorithm combines the strengths of AETRBO-DKAE and AETRBO-DKAEB, incorporating dynamic adjustment of both the LCB kappa parameter and the GP kernel's bandwidth. It further enhances exploration by integrating a gradient-based mechanism that encourages sampling in regions with high gradient magnitudes, thereby promoting movement towards potentially promising areas. The trust region adaptation strategy is retained, and the initial trust region radius is scaled based on dimensionality. The algorithm dynamically adjusts the DE population size based on the remaining budget.

# Justification
The new algorithm, AETRBO-DKAEBG, builds upon the strengths of AETRBO-DKAE and AETRBO-DKAEB by combining their key features and adding a gradient-based exploration component.

*   **Dynamic Kappa and Kernel Bandwidth Adjustment:** These features, inherited from AETRBO-DKAE and AETRBO-DKAEB, allow the algorithm to adapt to varying landscape complexities. Dynamic kappa helps to balance exploration and exploitation by adjusting the confidence bound, while dynamic kernel bandwidth allows the GP model to better capture the underlying function's characteristics.
*   **Gradient-Based Exploration:** This new feature encourages exploration in regions with high gradient magnitudes. The rationale behind this is that high gradient magnitudes often indicate proximity to promising areas, and sampling in these regions can accelerate the optimization process. This is implemented by adding a term to the acquisition function that is proportional to the estimated gradient magnitude.
*   **Trust Region Adaptation:** The trust region mechanism helps to control the step size and ensure that the algorithm does not stray too far from promising regions. The adaptive radius adjustment strategy, based on success ratio and GP model error, allows the algorithm to dynamically adjust the trust region size based on the optimization progress.
*   **Adaptive DE Population Size:** Adjusting the DE population size based on the remaining budget and problem dimensionality allows for efficient exploration within the trust region.

By combining these features, AETRBO-DKAEBG aims to achieve a better balance between exploration and exploitation, leading to improved performance on a wide range of optimization problems. The gradient-based exploration is specifically designed to address limitations of previous approaches by actively seeking out promising regions indicated by high gradient magnitudes, rather than relying solely on uncertainty or diversity metrics.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution
from numpy.linalg import norm as linalg_norm

class AdaptiveEvolutionaryTrustRegionBO_DKAEBG:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = min(5*dim, self.budget//10)

        # Do not add any other arguments without a default value
        self.gp = None
        self.best_x = None
        self.best_y = float('inf')
        self.trust_region_radius = 2.0 * np.sqrt(dim)  # Initial trust region radius, scaled by dimension
        self.radius_decay = 0.95
        self.radius_increase = 1.1
        self.min_radius = 0.1
        self.success_ratio = 0.0 #ratio to track the success of trust region
        self.random_restart_prob = 0.05
        self.de_pop_size = 10  # Population size for differential evolution
        self.lcb_kappa = 2.0  # Kappa parameter for Lower Confidence Bound
        self.kappa_decay = 0.99 # Decay rate for kappa
        self.min_kappa = 0.1 # Minimum value for kappa
        self.gp_error_threshold = 0.1 # Threshold for GP error
        self.noise_estimate = 1e-4 #initial noise estimate
        self.initial_length_scale = 1.0
        self.length_scale = self.initial_length_scale
        self.length_scale_decay = 0.95
        self.length_scale_increase = 1.05
        self.min_length_scale = 0.01
        self.max_length_scale = 100.0
        self.gradient_weight = 0.1

    def _sample_points(self, n_points):
        # sample points within the trust region or randomly
        # return array of shape (n_points, n_dims)
        if self.best_x is None or np.random.rand() < self.random_restart_prob:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=n_points)
            return qmc.scale(sample, self.bounds[0], self.bounds[1])
        else:
            points = []
            while len(points) < n_points:
                sample = np.random.uniform(-1, 1, size=(self.dim,))
                sample = self.best_x + self.trust_region_radius * sample
                # Clip to bounds
                sample = np.clip(sample, self.bounds[0], self.bounds[1])
                points.append(sample)
            return np.array(points)

    def _fit_model(self, X, y):
        # Fit and tune surrogate model
        # return the model
        # Do not change the function signature
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.length_scale, length_scale_bounds=(1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
        self.gp.fit(X, y)
        return self.gp

    def _acquisition_function(self, X):
        # Implement acquisition function: Lower Confidence Bound
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None:
            return np.random.normal(size=(len(X), 1))
        else:
            mu, sigma = self.gp.predict(X, return_std=True)
            LCB = mu - self.lcb_kappa * sigma

            # Gradient-based exploration
            gradient_norm = self._estimate_gradient_norm(X)
            acquisition = LCB - self.gradient_weight * gradient_norm

            return acquisition.reshape(-1, 1)

    def _estimate_gradient_norm(self, X):
        # Estimate the gradient norm of the GP mean function at X
        delta = 1e-4  # Small perturbation for gradient estimation
        gradient_norm = np.zeros(len(X))
        for i in range(len(X)):
            gradient = np.zeros(self.dim)
            for j in range(self.dim):
                x_plus = X[i].copy()
                x_minus = X[i].copy()
                x_plus[j] += delta
                x_minus[j] -= delta
                mu_plus, _ = self.gp.predict(x_plus.reshape(1, -1), return_std=True)
                mu_minus, _ = self.gp.predict(x_minus.reshape(1, -1), return_std=True)
                gradient[j] = (mu_plus - mu_minus) / (2 * delta)
            gradient_norm[i] = linalg_norm(gradient)
        return gradient_norm

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate using differential evolution within the trust region
        # return array of shape (batch_size, n_dims)

        if self.best_x is None:
            return self._sample_points(batch_size)

        def de_objective(x):
            return self._acquisition_function(x.reshape(1, -1))[0, 0]

        de_bounds = [(max(self.bounds[0][i], self.best_x[i] - self.trust_region_radius),
                      min(self.bounds[1][i], self.best_x[i] + self.trust_region_radius)) for i in range(self.dim)]

        # Adjust maxiter based on remaining budget and optimization progress
        remaining_evals = self.budget - self.n_evals
        maxiter = max(1, int(remaining_evals / (self.de_pop_size * self.dim * 2)))
        maxiter = min(maxiter, 100) #limit maxiter to prevent excessive computation

        # Adjust DE population size
        self.de_pop_size = min(max(5, int(remaining_evals / (self.dim * 5))), 20)

        result = differential_evolution(de_objective, de_bounds, popsize=self.de_pop_size, maxiter=maxiter, tol=0.01, disp=False)

        return result.x.reshape(1, -1)

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
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]
            self.success_ratio = 1.0 #reset success ratio
        else:
            self.success_ratio *= 0.75 #reduce success ratio if not improving

        #Update noise estimate
        self.noise_estimate = np.var(self.y)

    def _adjust_trust_region(self):
        # Adjust the trust region size based on the success
        if self.best_x is not None:
            #Calculate GP error within the trust region
            distances = np.linalg.norm(self.X - self.best_x, axis=1)
            within_tr = distances < self.trust_region_radius
            if np.any(within_tr):
                X_tr = self.X[within_tr]
                y_tr = self.y[within_tr]
                mu, _ = self.gp.predict(X_tr, return_std=True)
                gp_error = np.mean(np.abs(mu.reshape(-1,1) - y_tr))
            else:
                gp_error = 0.0

            if self.success_ratio > 0.5 and gp_error < self.gp_error_threshold:
                self.trust_region_radius = min(self.trust_region_radius * self.radius_increase, 5.0)
            else:
                self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius)
        else:
            self.trust_region_radius = max(self.trust_region_radius * self.radius_decay, self.min_radius) #shrink if no best_x

    def _adjust_kernel_bandwidth(self):
        # Adjust kernel bandwidth based on optimization progress
        if self.gp is not None and self.best_x is not None:
            # Estimate landscape complexity using the gradient norm of the GP mean function
            delta = 1e-4  # Small perturbation for gradient estimation
            gradient = np.zeros_like(self.best_x)
            for i in range(self.dim):
                x_plus = self.best_x.copy()
                x_minus = self.best_x.copy()
                x_plus[i] += delta
                x_minus[i] -= delta
                mu_plus, _ = self.gp.predict(x_plus.reshape(1, -1), return_std=True)
                mu_minus, _ = self.gp.predict(x_minus.reshape(1, -1), return_std=True)
                gradient[i] = (mu_plus - mu_minus) / (2 * delta)
            gradient_norm = linalg_norm(gradient)

            if gradient_norm > 1.0:  # High complexity, reduce bandwidth
                self.length_scale = max(self.length_scale * self.length_scale_decay, self.min_length_scale)
            else:  # Low complexity, increase bandwidth
                self.length_scale = min(self.length_scale * self.length_scale_increase, self.max_length_scale)
        else:
            self.length_scale = self.initial_length_scale

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
        batch_size = min(4, self.dim) #increase batch size to 4 or dim, whichever is smaller
        while self.n_evals < self.budget:
            # Fit the GP model
            self._fit_model(self.X, self.y)

            # Adjust kernel bandwidth
            self._adjust_kernel_bandwidth()

            # Select points by acquisition function
            next_X = self._select_next_points(batch_size)

            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)

            # Update the evaluated points
            self._update_eval_points(next_X, next_y)

            # Adjust trust region radius
            self._adjust_trust_region()

            # Decay kappa, adjust min_kappa based on noise estimate
            self.lcb_kappa = max(self.lcb_kappa * self.kappa_decay, self.min_kappa)
            self.min_kappa = 0.1 + np.sqrt(self.noise_estimate) #dynamic min kappa

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AdaptiveEvolutionaryTrustRegionBO_DKAEBG got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1804 with standard deviation 0.1026.

took 1197.10 seconds to run.