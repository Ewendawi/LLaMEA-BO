# Description
**Adaptive Trust Region Evolutionary BO with Dynamic Kernel, Acquisition Blending, Adaptive DE, and Gradient-Enhanced Trust Region Adjustment (ATREBO-DKAB-aDE-GE):** This algorithm combines the strengths of ATREBO-DKAB and ATREBO-DKAB-aDE by incorporating a dynamic kernel, acquisition function blending, and adaptive Differential Evolution (DE). It further enhances trust region adaptation by incorporating gradient information to make the trust region adjustment more informed. Specifically, it estimates the gradient of the surrogate model and uses this information to adjust the trust region radius, promoting expansion along promising directions and contraction in less promising ones. The adaptive DE strategy dynamically adjusts both the mutation and crossover rates based on the observed improvement in the acquisition function values.

# Justification
This algorithm builds upon the ATREBO-DKAB-aDE framework.
1.  **Gradient-Enhanced Trust Region Adjustment:** The key addition is the use of gradient information to guide trust region adaptation. By estimating the gradient of the GP surrogate model at the current best point, the algorithm can determine the direction of steepest ascent (or descent, for minimization). This allows for more intelligent trust region adjustments, expanding the region along promising directions and shrinking it along unpromising ones. This is achieved by calculating the cosine similarity between the estimated gradient and the vector pointing from the trust region center to the best point. This cosine similarity is then used to modulate the trust region expansion/shrinkage factor.
2.  **Dynamic Kernel and Acquisition Blending:** Inherited from ATREBO-DKAB, this feature allows the GP to better capture the function's landscape and balances exploration and exploitation.
3.  **Adaptive DE:** Inherited from ATREBO-DKAB-aDE, this allows the DE to better adapt to the local landscape of the acquisition function.
4.  **Efficiency:** The gradient is approximated using finite differences within the trust region to keep the computational cost low.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution, minimize

class AdaptiveTrustRegionEvolutionaryBO_DKAB_aDE_GE:
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
        self.trust_region_radius = 2.0 # Initial trust region radius
        self.trust_region_shrink = 0.5 # Shrink factor for trust region
        self.trust_region_expand = 1.5 # Expansion factor for trust region
        self.success_threshold = 0.75 # Threshold for trust region expansion
        self.failure_threshold = 0.25 # Threshold for trust region contraction
        self.trust_region_center = np.zeros(dim) # Initial trust region center
        self.best_x = None
        self.best_y = np.inf
        self.de_popsize = 10 # Population size for differential evolution
        self.de_mutation = 0.5 # Initial mutation rate for differential evolution
        self.de_crossover = 0.7 # Crossover rate for differential evolution
        self.mutation_adaptation_rate = 0.1 # Rate to adapt mutation based on success
        self.crossover_adaptation_rate = 0.1 # Rate to adapt crossover based on success
        self.success_threshold_de = 0.2 # Threshold for considering a generation successful
        self.exploration_weight = 0.1  # Initial weight for exploration
        self.exploration_weight_decay = 0.99 # Decay rate for exploration weight
        self.length_scale = 1.0 #Initial length scale
        self.length_scale_bounds = (1e-2, 1e2) #Bounds for the length scale
        self.min_trust_region_radius = 0.1
        self.gradient_estimation_delta = 0.01 # Step size for gradient estimation
        self.gradient_trust_region_adaptation_rate = 0.2 # Rate to adapt trust region based on gradient

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points within the trust region
        # return array of shape (n_points, n_dims)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        scaled_sample = qmc.scale(sample, -self.trust_region_radius, self.trust_region_radius)
        return scaled_sample + self.trust_region_center

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature
        # Optimize the length_scale
        def neg_log_likelihood(length_scale):
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=length_scale, length_scale_bounds="fixed")
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
            gp.fit(X, y)
            return -gp.log_marginal_likelihood()

        res = minimize(neg_log_likelihood, x0=self.length_scale, bounds=[self.length_scale_bounds])
        self.length_scale = res.x[0]

        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.length_scale, length_scale_bounds="fixed")
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.gp is None:
            return np.zeros((len(X), 1)) # Return zeros if the model is not fitted yet

        mu, sigma = self.gp.predict(X, return_std=True)
        # Expected Improvement acquisition function
        improvement = self.best_y - mu
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma <= 1e-6] = 0.0  # Avoid division by zero

        # Weighted combination of EI and exploration
        acquisition = (1 - self.exploration_weight) * ei.reshape(-1, 1) + self.exploration_weight * sigma.reshape(-1, 1)
        return acquisition

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a differential evolution strategy to optimize the acquisition function within the trust region
        # return array of shape (batch_size, n_dims)

        def de_objective(x):
            # Objective function for differential evolution (negative acquisition function)
            return -self._acquisition_function(x.reshape(1, -1))[0, 0]

        # Define bounds for differential evolution within the trust region
        de_bounds = list(zip(np.maximum(self.bounds[0], self.trust_region_center - self.trust_region_radius),
                             np.minimum(self.bounds[1], self.trust_region_center + self.trust_region_radius)))

        # Perform differential evolution
        de_result = differential_evolution(
            func=de_objective,
            bounds=de_bounds,
            popsize=self.de_popsize,
            mutation=self.de_mutation,
            recombination=self.de_crossover,
            maxiter=5, # Reduce maxiter for computational efficiency
            tol=0.01,
            seed=None,
            strategy='rand1bin',
            init='latinhypercube'
        )
        
        # Select the best point from the differential evolution result
        next_point = de_result.x.reshape(1, -1)
        
        return next_point

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
        
        # Update best observed solution
        best_index = np.argmin(self.y)
        if self.y[best_index, 0] < self.best_y:
            self.best_x = self.X[best_index]
            self.best_y = self.y[best_index, 0]

    def _estimate_gradient(self, x):
        # Estimate the gradient of the GP model at point x using finite differences
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.gradient_estimation_delta
            x_minus[i] -= self.gradient_estimation_delta
            
            # Ensure the points are within the bounds
            x_plus = np.clip(x_plus, self.bounds[0], self.bounds[1])
            x_minus = np.clip(x_minus, self.bounds[0], self.bounds[1])
            
            mu_plus = self.gp.predict(x_plus.reshape(1, -1))[0]
            mu_minus = self.gp.predict(x_minus.reshape(1, -1))[0]
            gradient[i] = (mu_plus - mu_minus) / (2 * self.gradient_estimation_delta)
        return gradient

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_X = np.clip(initial_X, self.bounds[0], self.bounds[1])
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        while self.n_evals < self.budget:
            # Optimization

            # Fit GP model
            self.gp = self._fit_model(self.X, self.y)

            # Select points by acquisition function using differential evolution
            batch_size = min(1, self.budget - self.n_evals) # Batch size of 1 for evolutionary strategy
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Calculate the ratio of improvement
            predicted_improvement = self.best_y - self.gp.predict(self.best_x.reshape(1, -1))[0]
            actual_improvement = self.best_y - min(self.y)[0]

            if abs(predicted_improvement) < 1e-9:
                ratio = 0.0
            else:
                ratio = actual_improvement / predicted_improvement

            # Estimate gradient
            gradient = self._estimate_gradient(self.best_x)
            
            # Calculate cosine similarity between gradient and vector to best point
            direction_vector = self.best_x - self.trust_region_center
            if np.linalg.norm(gradient) < 1e-6 or np.linalg.norm(direction_vector) < 1e-6:
                cosine_similarity = 0.0
            else:
                cosine_similarity = np.dot(gradient, direction_vector) / (np.linalg.norm(gradient) * np.linalg.norm(direction_vector))

            # Adjust trust region size based on ratio and gradient
            if ratio > self.success_threshold and abs(predicted_improvement) > 1e-3:
                self.trust_region_radius *= (self.trust_region_expand + self.gradient_trust_region_adaptation_rate * cosine_similarity)
            elif ratio < self.failure_threshold:
                self.trust_region_radius *= self.trust_region_shrink
                self.trust_region_radius = max(self.trust_region_radius, self.min_trust_region_radius)

            # Update trust region center
            self.trust_region_center = self.best_x
            
            # Adapt mutation and crossover rate based on success
            if len(self.y) > self.de_popsize:
                recent_ys = self.y[-self.de_popsize:]
                improvement_ratios = (recent_ys[:-1] - recent_ys[1:]) / (recent_ys[:-1] + 1e-9)  # Adding a small value to avoid division by zero
                success_rate = np.sum(improvement_ratios > 0) / len(improvement_ratios)

                if success_rate > self.success_threshold_de:
                    self.de_mutation *= (1 - self.mutation_adaptation_rate)
                    self.de_crossover *= (1 + self.crossover_adaptation_rate)
                else:
                    self.de_mutation *= (1 + self.mutation_adaptation_rate)
                    self.de_crossover *= (1 - self.crossover_adaptation_rate)

                self.de_mutation = np.clip(self.de_mutation, 0.1, 1.99) # Clip mutation to avoid ValueError
                self.de_crossover = np.clip(self.de_crossover, 0.1, 0.99)

            # Decay exploration weight
            self.exploration_weight *= self.exploration_weight_decay
            self.exploration_weight = max(self.exploration_weight, 0.01)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AdaptiveTrustRegionEvolutionaryBO_DKAB_aDE_GE got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.2020 with standard deviation 0.1084.

took 590.94 seconds to run.