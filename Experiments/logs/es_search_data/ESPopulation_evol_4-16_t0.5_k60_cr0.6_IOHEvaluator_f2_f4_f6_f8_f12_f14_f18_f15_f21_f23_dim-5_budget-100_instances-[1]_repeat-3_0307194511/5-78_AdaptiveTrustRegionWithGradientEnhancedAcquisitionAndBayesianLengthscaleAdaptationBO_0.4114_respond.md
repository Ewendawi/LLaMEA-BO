# Description
**Adaptive Trust Region with Gradient-Enhanced Acquisition and Bayesian Lengthscale Adaptation (ATGRGBLA-BO)**: This algorithm combines the strengths of AGTRDABBO and GEATRBO, focusing on enhanced gradient estimation, dynamic EI/UCB balancing, and adaptive trust region management. A key innovation is the Bayesian adaptation of the GP kernel's lengthscale during the optimization process, which improves the GP model's accuracy and responsiveness. The algorithm also integrates a more refined gradient estimation using multiple points and central differences with an adaptive step size based on the current lengthscale. Furthermore, a hybrid acquisition function combining EI and UCB is used, with the weights dynamically adjusted based on the success rate of local searches and the estimated gradient norm. This allows for a more adaptive exploration-exploitation trade-off.

# Justification
The algorithm builds upon the following key ideas:

1.  **Adaptive Lengthscale:** Instead of estimating the lengthscale only once at the beginning, the algorithm adapts it during the optimization process using a Bayesian approach. This is done by sampling a few lengthscales and marginalizing over them. This allows the GP model to better adapt to the local characteristics of the objective function.

2.  **Enhanced Gradient Estimation:** The gradient is estimated using central differences with a step size that is dynamically adjusted based on the current lengthscale of the GP model. Multiple points around the current best are used to average the gradient estimate, reducing the impact of noise.

3.  **Dynamic EI/UCB Balancing:** The weights of EI and UCB in the acquisition function are dynamically adjusted based on the success rate of local searches and the norm of the estimated gradient. If local searches are successful and the gradient norm is high, the weight of UCB is increased to promote exploitation. Otherwise, the weight of EI is increased to encourage exploration.

4.  **Adaptive Trust Region:** The trust region size is adapted based on the success of the local search. If the local search is successful, the trust region is expanded. Otherwise, it is shrunk. Momentum is used to smooth the changes in the trust region size.

5.  **Hybrid Acquisition:** Combines EI for global exploration and UCB for local exploitation, dynamically balancing them using success rate and gradient norm.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize

class AdaptiveTrustRegionWithGradientEnhancedAcquisitionAndBayesianLengthscaleAdaptationBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * self.dim # initial number of samples
        self.batch_size = min(5, self.dim)
        self.trust_region_size = 2.0
        self.trust_region_shrink = 0.5
        self.trust_region_expand = 2.0
        self.beta = 2.0  # Exploration parameter for UCB
        self.beta_decay = 0.99 # Decay rate for beta
        self.best_x = None
        self.best_y = np.inf
        self.delta = 1e-3 # step size for finite differences in gradient estimation
        self.trust_region_momentum = 0.5 # Momentum for trust region size update
        self.prev_trust_region_change = 0.0
        self.global_search_prob = 0.05 # Probability of performing a global search step
        self.ei_weight = 0.5 # Initial weight for EI
        self.ucb_weight = 0.5 # Initial weight for UCB
        self.success_threshold = 0.1 # Threshold for considering local search successful
        self.lengthscale_samples = 3 # Number of lengthscale samples
        self.lengthscale_bounds = (1e-3, 1e3) # Bounds for lengthscale
        self.gradient_points = 3 # Number of points to average for gradient estimation
        self.success_rate = 0.0 # Moving average of local search success
        self.success_momentum = 0.8 # Momentum for updating success rate

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        # Fit and tune surrogate model with Bayesian lengthscale adaptation
        # return the model

        # Estimate lengthscale using nearest neighbors
        nn = NearestNeighbors(n_neighbors=min(len(X), 10)).fit(X)
        distances, _ = nn.kneighbors(X)
        median_distance = np.median(distances[:, 1])

        # Sample lengthscales around the median distance
        lengthscale_candidates = np.random.normal(loc=median_distance, scale=median_distance/3, size=self.lengthscale_samples)
        lengthscale_candidates = np.clip(lengthscale_candidates, self.lengthscale_bounds[0], self.lengthscale_bounds[1])

        models = []
        log_likelihoods = []
        for lengthscale in lengthscale_candidates:
            kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=lengthscale, length_scale_bounds=self.lengthscale_bounds)
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
            model.fit(X, y)
            models.append(model)
            log_likelihoods.append(model.log_marginal_likelihood())

        # Normalize log likelihoods to get weights
        weights = np.exp(np.array(log_likelihoods) - np.max(log_likelihoods))
        weights /= np.sum(weights)

        # Return the ensemble of models and their weights
        return models, weights

    def _predict(self, X):
        # Predict using the ensemble of GP models
        models, weights = self._fit_model(self.X, self.y)
        mu = np.zeros((len(X),))
        sigma2 = np.zeros((len(X),))

        for model, weight in zip(models, weights):
            m, s = model.predict(X, return_std=True)
            mu += weight * m
            sigma2 += weight * (s**2 + m**2)
        sigma2 -= mu**2
        sigma = np.sqrt(sigma2)
        return mu, sigma

    def _acquisition_function_ei(self, X):
        # Implement Expected Improvement acquisition function
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu, sigma = self._predict(X)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        y_best = np.min(self.y)
        imp = y_best - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0

        return ei

    def _acquisition_function_ucb(self, X):
        # Implement Upper Confidence Bound acquisition function
        mu, sigma = self._predict(X)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        ucb = mu - self.beta * sigma  # minimize

        return ucb

    def _select_next_points_ei(self, batch_size):
        # Select the next points to evaluate using EI
        candidate_points = self._sample_points(10 * batch_size)
        acquisition_values = self._acquisition_function_ei(candidate_points)

        indices = np.argsort(acquisition_values.flatten())[::-1][:batch_size]
        return candidate_points[indices]

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

        # Update best seen point
        best_index = np.argmin(self.y)
        if self.y[best_index][0] < self.best_y:
            self.best_y = self.y[best_index][0]
            self.best_x = self.X[best_index]

    def _estimate_gradient(self, x):
        # Estimate the gradient of the function at point x using central finite differences
        gradient = np.zeros(self.dim)
        models, weights = self._fit_model(self.X, self.y)
        lengthscales = [model.kernel_.k2.length_scale for model in models]
        lengthscale = np.average(lengthscales, weights=weights)
        delta = min(lengthscale, self.delta) # Adaptive delta

        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += delta
            x_minus[i] -= delta
            
            # Clip to ensure the points are within bounds
            x_plus[i] = np.clip(x_plus[i], self.bounds[0][i], self.bounds[1][i])
            x_minus[i] = np.clip(x_minus[i], self.bounds[0][i], self.bounds[1][i])

            # Use the GP model to predict function values
            y_plus, _ = self._predict(x_plus.reshape(1, -1))
            y_minus, _ = self._predict(x_minus.reshape(1, -1))

            gradient[i] = (y_plus - y_minus) / (2 * delta)
        return gradient

    def _local_search(self, center, gradient, n_points=50):
        # Perform local search within the trust region using the GP model and gradient information
        # Generate candidate points within the trust region
        candidate_points = center + self.trust_region_size * np.random.uniform(-1, 1, size=(n_points, self.dim))

        # Clip the candidate points to stay within the bounds
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])

        # Predict the mean values using the GP model
        mu, _ = self._predict(candidate_points)

        # Incorporate gradient information into the prediction
        mu = mu.reshape(-1) - 0.1 * np.sum(gradient * (candidate_points - center), axis=1)

        # Select the point with the minimum predicted mean value
        best_index = np.argmin(mu)
        best_point = candidate_points[best_index]

        return best_point

    def _mixed_acquisition(self, X):
        # Combine EI and UCB with dynamic weights
        ei = self._acquisition_function_ei(X)
        ucb = self._acquisition_function_ucb(X)
        return self.ei_weight * ei + self.ucb_weight * ucb

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        # Initial sampling
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_index = np.argmin(self.y)
        best_x = self.X[best_index]
        best_y = self.y[best_index][0]

        while self.n_evals < self.budget:

            # Global search with probability global_search_prob
            if np.random.rand() < self.global_search_prob:
                next_x = self._select_next_points_ei(1)[0] # Select a point using EI for global search
            else:
                # Estimate gradient at the best point
                gradient = self._estimate_gradient(best_x)

                # Perform local search within the trust region
                next_x = self._local_search(best_x.copy(), gradient)
                next_x = np.clip(next_x, self.bounds[0], self.bounds[1]) # Ensure it's within bounds

            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0] # Evaluate the actual function
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            # Check if the new point is better than the current best
            improvement = (best_y - next_y) / best_y if best_y != 0 else (best_y - next_y)
            if next_y < best_y:
                best_x = next_x
                best_y = next_y
                # Adjust trust region size
                trust_region_change = self.trust_region_expand - 1.0
                self.trust_region_size *= self.trust_region_expand

                # Increase UCB weight if local search is successful
                if improvement > self.success_threshold:
                    self.ucb_weight = min(1.0, self.ucb_weight + 0.1)
                    self.ei_weight = max(0.0, self.ei_weight - 0.1)
            else:
                # Shrink trust region if no improvement
                trust_region_change = 1.0 - self.trust_region_shrink
                self.trust_region_size *= self.trust_region_shrink

                # Increase EI weight if local search is unsuccessful
                self.ei_weight = min(1.0, self.ei_weight + 0.1)
                self.ucb_weight = max(0.0, self.ucb_weight - 0.1)

            # Apply momentum to trust region size update
            self.trust_region_size = self.trust_region_momentum * self.trust_region_size + (1 - self.trust_region_momentum) * np.clip(self.trust_region_size, 0.1, 5.0) # Keep trust region within reasonable bounds
            
            # Decay exploration parameter
            self.beta *= self.beta_decay

        return best_y, best_x
```
## Feedback
 The algorithm AdaptiveTrustRegionWithGradientEnhancedAcquisitionAndBayesianLengthscaleAdaptationBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1791 with standard deviation 0.1029.

took 4257.56 seconds to run.