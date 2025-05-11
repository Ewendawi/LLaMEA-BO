You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- GradientEnhancedAdaptiveTrustRegionOptimisticHybridMBO: 0.1986, 81.67 seconds, **GradientEnhancedAdaptiveTrustRegionOptimisticHybridMBO (GEATROHMBO)**: This algorithm builds upon the strengths of GEATRBO and GEATROHBO by incorporating a more sophisticated mechanism for balancing exploration and exploitation within the trust region. It uses a hybrid acquisition function combining EI and UCB, gradient-enhanced local search, and adaptive trust region management. A key innovation is the dynamic adjustment of the weights assigned to EI and UCB based on the success of recent local searches. Specifically, if local searches consistently yield improvements, the weight of UCB is increased to intensify local refinement. Conversely, if local searches stagnate, the weight of EI is increased to encourage global exploration. The algorithm also incorporates momentum in the trust region size adaptation and a probabilistic global search step. The lengthscale is estimated using nearest neighbors for better GP model fitting.


- AdaptiveGradientEnhancedTrustRegionWithDynamicAcquisitionBalancingBO: 0.1918, 168.97 seconds, **Adaptive Gradient-enhanced Trust Region with Dynamic Acquisition Balancing BO (AGTRDABBO)**: This algorithm builds upon the strengths of GEATRBO and GEATR-OHBO by introducing a dynamic balancing strategy between Expected Improvement (EI) and Upper Confidence Bound (UCB) within the trust region. It retains gradient estimation for local search enhancement and adaptive trust region management with momentum. The key innovation is the dynamic adjustment of the EI/UCB balance based on the success rate of local searches. Furthermore, the gradient estimation is improved by averaging gradients over multiple points near the current best, and a more robust trust region update is implemented.


- GradientEnhancedAdaptiveTrustRegionBO: 0.1917, 159.41 seconds, **GradientEnhancedAdaptiveTrustRegionBO (GEATRBO) with Improved Gradient Estimation and Acquisition Balancing:** This algorithm builds upon the existing GEATRBO framework by refining the gradient estimation and introducing a more dynamic balance between exploitation and exploration. The gradient estimation is enhanced by using a central difference scheme with a dynamically adjusted step size based on the lengthscale of the GP model. The acquisition function is modified to incorporate a weighted combination of Expected Improvement (EI) and Upper Confidence Bound (UCB), with the weights dynamically adjusted based on the success of the local search. This allows for a more adaptive exploration-exploitation trade-off.


- AdaptiveGradientTrustRegionBayesBO: 0.1866, 81.77 seconds, **AdaptiveGradientTrustRegionBayesBO (AGTRBayesBO)**: This algorithm builds upon the GradientEnhancedAdaptiveTrustRegionOptimisticHybridBO (GEATROHBO) by introducing a dynamic weighting mechanism for the EI and UCB components of the acquisition function. It also refines the gradient estimation process using a central difference scheme and incorporates a more robust trust region adaptation strategy based on the actual improvement observed. Furthermore, it includes a mechanism to adapt the lengthscale of the GP kernel during the optimization process, instead of only estimating it initially.




The selected solutions to update are:
## AdaptiveGradientEnhancedTrustRegionWithDynamicAcquisitionBalancingBO
**Adaptive Gradient-enhanced Trust Region with Dynamic Acquisition Balancing BO (AGTRDABBO)**: This algorithm builds upon the strengths of GEATRBO and GEATR-OHBO by introducing a dynamic balancing strategy between Expected Improvement (EI) and Upper Confidence Bound (UCB) within the trust region. It retains gradient estimation for local search enhancement and adaptive trust region management with momentum. The key innovation is the dynamic adjustment of the EI/UCB balance based on the success rate of local searches. Furthermore, the gradient estimation is improved by averaging gradients over multiple points near the current best, and a more robust trust region update is implemented.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors

class AdaptiveGradientEnhancedTrustRegionWithDynamicAcquisitionBalancingBO:
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
        self.global_search_prob = 0.05 # Probability of performing a global search step
        self.delta = 1e-3 # step size for finite differences in gradient estimation
        self.trust_region_momentum = 0.5 # Momentum for trust region size update
        self.ei_weight = 0.5 # Initial weight for EI
        self.ucb_weight = 0.5 # Initial weight for UCB
        self.success_threshold = 0.7 # Threshold for adjusting EI/UCB weights
        self.success_rate = 0.0 # Moving average of local search success
        self.success_momentum = 0.8 # Momentum for updating success rate
        self.gradient_points = 3 # Number of points to average for gradient estimation

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

        # Efficient lengthscale estimation using nearest neighbors
        nn = NearestNeighbors(n_neighbors=min(len(X), 10)).fit(X)
        distances, _ = nn.kneighbors(X)
        median_distance = np.median(distances[:, 1])  # Exclude the point itself

        # Define the kernel with the estimated lengthscale
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=median_distance, length_scale_bounds=(1e-3, 1e3))

        # Gaussian Process Regressor
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
        model.fit(X, y)
        return model

    def _acquisition_function_ei(self, X):
        # Implement Expected Improvement acquisition function
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        model = self._fit_model(self.X, self.y)
        mu, sigma = model.predict(X, return_std=True)
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
        model = self._fit_model(self.X, self.y)
        mu, sigma = model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        ucb = mu - self.beta * sigma  # minimize

        return ucb

    def _acquisition_function(self, X):
        # Combine EI and UCB with dynamic weights
        ei = self._acquisition_function_ei(X)
        ucb = self._acquisition_function_ucb(X)
        return self.ei_weight * ei + self.ucb_weight * ucb

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        candidate_points = self._sample_points(10 * batch_size)
        acquisition_values = self._acquisition_function(candidate_points)

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

    def _estimate_gradient(self, model, x):
        # Estimate the gradient of the function at point x using finite differences
        gradient = np.zeros(self.dim)
        
        # Average gradient over multiple points near x
        for _ in range(self.gradient_points):
            x_sample = x + np.random.normal(0, self.trust_region_size / 5, self.dim) # Add some noise
            x_sample = np.clip(x_sample, self.bounds[0], self.bounds[1]) # Clip to bounds
            
            gradient_sample = np.zeros(self.dim)
            for i in range(self.dim):
                x_plus = x_sample.copy()
                x_minus = x_sample.copy()
                x_plus[i] += self.delta
                x_minus[i] -= self.delta
                
                # Clip to ensure the points are within bounds
                x_plus[i] = np.clip(x_plus[i], self.bounds[0][i], self.bounds[1][i])
                x_minus[i] = np.clip(x_minus[i], self.bounds[0][i], self.bounds[1][i])

                # Use the GP model to predict function values
                y_plus, _ = model.predict(x_plus.reshape(1, -1), return_std=True)
                y_minus, _ = model.predict(x_minus.reshape(1, -1), return_std=True)

                gradient_sample[i] = (y_plus - y_minus) / (2 * self.delta)
            
            gradient += gradient_sample
        
        return gradient / self.gradient_points

    def _local_search(self, model, center, gradient, n_points=50):
        # Perform local search within the trust region using the GP model and gradient information
        # Generate candidate points within the trust region
        candidate_points = center + self.trust_region_size * np.random.uniform(-1, 1, size=(n_points, self.dim))

        # Clip the candidate points to stay within the bounds
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])

        # Predict the mean values using the GP model
        mu, _ = model.predict(candidate_points, return_std=True)

        # Incorporate gradient information into the prediction
        mu = mu.reshape(-1) - 0.1 * np.sum(gradient * (candidate_points - center), axis=1)

        # Select the point with the minimum predicted mean value
        best_index = np.argmin(mu)
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
        
        num_success = 0

        while self.n_evals < self.budget:
            # Fit the GP model
            model = self._fit_model(self.X, self.y)

            # Global search with probability global_search_prob
            if np.random.rand() < self.global_search_prob:
                next_x = self._select_next_points(1)[0] # Select a point using EI for global search
            else:
                # Estimate gradient at the best point
                gradient = self._estimate_gradient(model, best_x)

                # Perform local search within the trust region
                next_x = self._local_search(model, best_x.copy(), gradient)
                next_x = np.clip(next_x, self.bounds[0], self.bounds[1]) # Ensure it's within bounds

            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0] # Evaluate the actual function
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            # Check if the new point is better than the current best
            if next_y < best_y:
                best_x = next_x
                best_y = next_y
                
                # Update success rate
                num_success += 1
                
                # Adjust trust region size
                self.trust_region_size *= self.trust_region_expand
            else:
                # Shrink trust region if no improvement
                self.trust_region_size *= self.trust_region_shrink

            # Apply momentum to trust region size update
            self.trust_region_size = self.trust_region_momentum * self.trust_region_size + (1 - self.trust_region_momentum) * np.clip(self.trust_region_size, 0.1, 5.0) # Keep trust region within reasonable bounds
            
            # Decay exploration parameter
            self.beta *= self.beta_decay
            
            # Update success rate and adjust EI/UCB weights
            self.success_rate = self.success_momentum * self.success_rate + (1 - self.success_momentum) * (num_success > 0)
            num_success = 0 #reset
            
            if self.success_rate > self.success_threshold:
                # Increase UCB weight, decrease EI weight
                self.ucb_weight = min(1.0, self.ucb_weight + 0.05)
                self.ei_weight = max(0.0, self.ei_weight - 0.05)
            else:
                # Increase EI weight, decrease UCB weight
                self.ei_weight = min(1.0, self.ei_weight + 0.05)
                self.ucb_weight = max(0.0, self.ucb_weight - 0.05)

        return best_y, best_x

```
The algorithm AdaptiveGradientEnhancedTrustRegionWithDynamicAcquisitionBalancingBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1918 with standard deviation 0.1019.

took 168.97 seconds to run.

## AdaptiveGradientTrustRegionBayesBO
**AdaptiveGradientTrustRegionBayesBO (AGTRBayesBO)**: This algorithm builds upon the GradientEnhancedAdaptiveTrustRegionOptimisticHybridBO (GEATROHBO) by introducing a dynamic weighting mechanism for the EI and UCB components of the acquisition function. It also refines the gradient estimation process using a central difference scheme and incorporates a more robust trust region adaptation strategy based on the actual improvement observed. Furthermore, it includes a mechanism to adapt the lengthscale of the GP kernel during the optimization process, instead of only estimating it initially.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors

class AdaptiveGradientTrustRegionBayesBO:
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
        self.success_threshold = 0.7
        self.beta = 2.0  # Exploration parameter for UCB
        self.beta_decay = 0.99 # Decay rate for beta
        self.global_search_prob = 0.05 # Probability of performing a global search step
        self.delta = 1e-3 # step size for finite differences in gradient estimation
        self.ei_weight = 0.5
        self.ucb_weight = 0.5
        self.ei_success = 0
        self.ucb_success = 0
        self.weight_update_rate = 0.1
        self.lengthscale = 1.0 # Initial lengthscale

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

        # Efficient lengthscale estimation using nearest neighbors
        # Adaptive lengthscale: re-estimate every few iterations
        if len(X) > self.n_init and self.n_evals % (2 * self.dim) == 0:
            nn = NearestNeighbors(n_neighbors=min(len(X), 10)).fit(X)
            distances, _ = nn.kneighbors(X)
            self.lengthscale = np.median(distances[:, 1])  # Exclude the point itself
            self.lengthscale = np.clip(self.lengthscale, 1e-3, 1e3)

        # Define the kernel with the estimated lengthscale
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=self.lengthscale, length_scale_bounds=(1e-3, 1e3))

        # Gaussian Process Regressor
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))  # Return zeros if the model hasn't been fit yet

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

        # Upper Confidence Bound
        ucb = mu - self.beta * sigma # minimize

        # Combine EI and UCB (weighted average)
        acquisition = self.ei_weight * ei + self.ucb_weight * ucb
        return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)
        candidate_points = self._sample_points(10 * batch_size)  # Generate more candidates
        acquisition_values = self._acquisition_function(candidate_points)

        # Sort by acquisition function value and select top batch_size points
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
        # Estimate the gradient of the function at point x using central finite differences
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

    def _local_search(self, model, center, gradient, n_points=50):
        # Perform local search within the trust region using the GP model and gradient information
        # Generate candidate points within the trust region
        candidate_points = center + self.trust_region_size * np.random.uniform(-1, 1, size=(n_points, self.dim))

        # Clip the candidate points to stay within the bounds
        candidate_points = np.clip(candidate_points, self.bounds[0], self.bounds[1])

        # Predict the mean values using the GP model
        mu, _ = model.predict(candidate_points, return_std=True)

        # Incorporate gradient information into the prediction
        mu = mu.reshape(-1) - 0.1 * np.sum(gradient * (candidate_points - center), axis=1)

        # Select the point with the minimum predicted mean value
        best_index = np.argmin(mu)
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

            # Perform local search within the trust region
            if np.random.rand() < self.global_search_prob:
                # Global search step
                next_x = self._select_next_points(1)[0]
                acquisition_source = "EI" # Assume EI is used for global search
            else:
                next_x = self._local_search(model, best_x.copy(), gradient)
                acquisition_source = "UCB" # Assume UCB is used for local search

            next_x = np.clip(next_x, self.bounds[0], self.bounds[1]) # Ensure it's within bounds

            next_y = self._evaluate_points(func, next_x.reshape(1, -1))[0, 0] # Evaluate the actual function
            self._update_eval_points(next_x.reshape(1, -1), np.array([[next_y]]))

            # Check if the new point is better than the current best
            improvement = best_y - next_y
            predicted_improvement, _ = model.predict(next_x.reshape(1, -1), return_std=True)
            predicted_improvement = best_y - predicted_improvement[0]

            if next_y < best_y:
                best_x = next_x
                best_y = next_y

                # Adjust trust region size based on actual vs predicted improvement
                if predicted_improvement > 0:
                    improvement_ratio = improvement / predicted_improvement
                    if improvement_ratio > 0.5:
                        self.trust_region_size *= self.trust_region_expand
                    else:
                        self.trust_region_size *= (1 + improvement_ratio) # Slightly expand if there's some improvement
                else:
                    self.trust_region_size *= self.trust_region_expand # Expand if predicted improvement is zero

                # Update EI/UCB success counters
                if acquisition_source == "EI":
                    self.ei_success += 1
                else:
                    self.ucb_success += 1
            else:
                # Shrink trust region if no improvement
                self.trust_region_size *= self.trust_region_shrink

            self.trust_region_size = np.clip(self.trust_region_size, 0.1, 5.0) # Keep trust region within reasonable bounds

            # Decay exploration parameter
            self.beta *= self.beta_decay

            # Update EI/UCB weights
            total_success = self.ei_success + self.ucb_success
            if total_success > 10: # start updating weights after some evaluations
                ei_ratio = self.ei_success / total_success
                ucb_ratio = self.ucb_success / total_success

                self.ei_weight = (1 - self.weight_update_rate) * self.ei_weight + self.weight_update_rate * ei_ratio
                self.ucb_weight = (1 - self.weight_update_rate) * self.ucb_weight + self.weight_update_rate * ucb_ratio

                self.ei_weight = np.clip(self.ei_weight, 0.1, 0.9)
                self.ucb_weight = np.clip(self.ucb_weight, 0.1, 0.9)

                # Reset success counters
                self.ei_success = 0
                self.ucb_success = 0

        return best_y, best_x

```
The algorithm AdaptiveGradientTrustRegionBayesBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1866 with standard deviation 0.1084.

took 81.77 seconds to run.

Combine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it. If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.




Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

