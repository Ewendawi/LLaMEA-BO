# Description
**Gradient-Enhanced Adaptive Trust Region Bayesian Optimization (GEATRBO)**: This algorithm enhances the AdaptiveTrustRegionBO by incorporating gradient information into both the local search and the trust region adaptation. It uses an efficient gradient estimation technique, similar to TREGEBO, to improve the accuracy of the Gaussian Process (GP) model and guide the local search. The trust region size is adaptively adjusted based on the agreement between predicted and actual improvements, as well as the magnitude of the estimated gradient. A global search step is included to escape local optima.

# Justification
This algorithm combines the strengths of AdaptiveTrustRegionBO and TREGEBO to achieve a better balance between exploration and exploitation.

1.  **Gradient-Enhanced Local Search:** The local search is enhanced by incorporating gradient information, similar to TREGEBO. This allows for a more informed search within the trust region, leading to faster convergence.
2.  **Adaptive Trust Region with Gradient Information:** The trust region size is adapted based on the ratio of actual to predicted improvement, as in AdaptiveTrustRegionBO. Additionally, the magnitude of the estimated gradient is used to influence the trust region size. A large gradient suggests a steep slope, which may warrant a larger trust region to explore further.
3.  **Global Search:** A global search step is included to escape local optima, as in AdaptiveTrustRegionBO.
4.  **Efficient Gradient Estimation:** The gradient is estimated using a finite difference method with a small number of function evaluations, as in TREGEBO. To avoid exceeding the budget during gradient estimation, the GP model is used to approximate function values for points used in finite difference calculations. This makes the algorithm more computationally efficient.
5. **Addressing previous errors**: The previous algorithms may get stuck in local optima. By incorporating a global search step and gradient information, the algorithm can better explore the search space and escape local optima. The adaptive trust region size allows the algorithm to adjust the search space based on the local landscape.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

class GEATRBO:
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
        self.local_search_exploitation = 0.8  # Weight for exploitation in local search
        self.global_search_prob = 0.05 # Probability of performing a global search step
        self.delta = 1e-3 # step size for finite differences in gradient estimation
        self.gradient_weight = 0.1 # Weight for gradient in local search

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
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        model = self._fit_model(self.X, self.y)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Expected Improvement
        y_best = np.min(self.y)
        imp = y_best - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # avoid division by zero
        return ei

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        candidate_points = self._sample_points(10 * batch_size)
        acquisition_values = self._acquisition_function(candidate_points)
        indices = np.argsort(acquisition_values.flatten())[::-1][:batch_size]
        return candidate_points[indices]

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
    
    def _estimate_gradient(self, model, x):
        # Estimate the gradient of the function at point x using finite differences
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.delta
            x_minus[i] -= self.delta
            
            # Clip to ensure the points are within bounds
            x_plus[i] = np.clip(x_plus[i], self.bounds[0][i], self.bounds[1][i])
            x_minus[i] = np.clip(x_minus[i], self.bounds[0][i], self.bounds[1][i])

            # Use the GP model to predict function values
            y_plus, _ = model.predict(x_plus.reshape(1, -1), return_std=True)
            y_minus, _ = model.predict(x_minus.reshape(1, -1), return_std=True)

            gradient[i] = (y_plus - y_minus) / (2 * self.delta)
        return gradient

    def _local_search(self, model, center, best_y, gradient, n_points=50):
        # Perform local search within the trust region using the GP model and acquisition function
        # Generate candidate points within the trust region
        candidate_points = center + self.trust_region_size * np.random.uniform(-1, 1, size=(n_points, self.dim))
        
        # Clip the candidate points to stay within the bounds
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])
        
        # Predict the mean values using the GP model
        mu, _ = model.predict(candidate_points, return_std=True)
        mu = mu.reshape(-1, 1)

        # Calculate acquisition function values
        ei = self._acquisition_function(candidate_points)

        # Incorporate gradient information into the prediction
        gradient_component = self.gradient_weight * np.sum(gradient * (candidate_points - center), axis=1, keepdims=True)

        # Combine predicted mean, acquisition function values, and gradient information
        weighted_values = self.local_search_exploitation * mu + (1 - self.local_search_exploitation) * (-ei) + gradient_component # Minimize mu, maximize EI, minimize gradient component

        # Select the point with the minimum weighted value
        best_index = np.argmin(weighted_values)
        best_point = candidate_points[best_index]
        
        return best_point

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
            # Fit the GP model
            model = self._fit_model(self.X, self.y)

            # Estimate gradient at the best point
            gradient = self._estimate_gradient(model, best_x)
            gradient_norm = np.linalg.norm(gradient)

            # Perform global search with a small probability
            if np.random.rand() < self.global_search_prob:
                next_x = self._select_next_points(1)[0] # Select a point using the acquisition function
            else:
                # Perform local search within the trust region
                next_x = self._local_search(model, best_x.copy(), best_y, gradient)

            next_x = np.clip(next_x, self.bounds[0], self.bounds[1]) # Ensure it's within bounds
            
            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0] # Evaluate the actual function
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            # Calculate the improvement
            improvement = best_y - next_y

            # Predict the improvement using the GP model
            predicted_y, _ = model.predict(next_x.reshape(1, -1), return_std=True)
            predicted_improvement = best_y - predicted_y[0]

            # Adjust trust region size based on the ratio of actual to predicted improvement and gradient magnitude
            if predicted_improvement != 0:
                ratio = improvement / predicted_improvement
                if ratio > 0.5:
                    self.trust_region_size *= self.trust_region_expand
                else:
                    self.trust_region_size *= self.trust_region_shrink
            else:
                # If predicted improvement is zero, shrink the trust region
                self.trust_region_size *= self.trust_region_shrink

            # Adjust trust region size based on gradient magnitude
            self.trust_region_size *= (1 + 0.1 * gradient_norm) # Increase trust region if gradient is large
            
            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0) # Keep trust region within reasonable bounds

            # Check if the new point is better than the current best
            if next_y < best_y:
                best_x = next_x
                best_y = next_y

        return best_y, best_x
```
## Feedback
 The algorithm GEATRBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1705 with standard deviation 0.1027.

took 258.02 seconds to run.