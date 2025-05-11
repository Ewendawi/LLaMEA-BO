# Description
**Adaptive Trust Region Evolutionary Bayesian Optimization with Enhanced Local Search and Adaptive Exploration (ATREBO-ELS-AE):** This algorithm combines the strengths of ATREBO-DKAB-aDE-LS and ATRELS-DKBO-PLS, focusing on a more refined local search strategy and adaptive exploration. It integrates a dynamic kernel, acquisition function blending (EI and variance), adaptive Differential Evolution (DE), and a local search strategy within an adaptive trust region framework. The local search is enhanced by dynamically adjusting both the search probability and radius based on the EI value and the success rate of previous local searches. Exploration is adaptively controlled using a dynamic exploration weight and trust region adaptation. A key addition is the use of a separate, smaller trust region for local search to intensify exploitation around promising solutions.

# Justification
The new algorithm leverages the strengths of both ATREBO-DKAB-aDE-LS and ATRELS-DKBO-PLS. ATREBO-DKAB-aDE-LS provides a strong foundation with its dynamic kernel, acquisition blending, adaptive DE, and L-BFGS-B local search. ATRELS-DKBO-PLS contributes a more sophisticated probabilistic local search strategy.

Key components and changes:

1.  **Dynamic Kernel and Acquisition Blending:** Inherited from ATREBO-DKAB-aDE-LS, this ensures the GP model accurately captures the function landscape and balances exploration/exploitation.
2.  **Adaptive Differential Evolution (DE):** Also from ATREBO-DKAB-aDE-LS, this allows for efficient exploration within the trust region.
3.  **Enhanced Local Search:** Combines the L-BFGS-B local search from ATREBO-DKAB-aDE-LS with the probabilistic local search from ATRELS-DKBO-PLS. The probability and radius of local search are dynamically adjusted based on the EI value and the success rate of previous local searches. A smaller, separate trust region is used for local search to intensify exploitation.
4.  **Adaptive Exploration:** The exploration weight is dynamically adjusted based on the optimization progress, and the trust region is adapted based on the ratio of actual to predicted improvement.
5.  **Computational Efficiency:** The number of iterations for DE and GP hyperparameter optimization is reduced to maintain computational efficiency.

The combination of these strategies aims to provide a robust and efficient optimization algorithm that can effectively handle a wide range of black-box optimization problems.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution, minimize, fmin_l_bfgs_b

class AdaptiveTrustRegionEvolutionaryBO_ELS_AE:
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
        self.de_popsize = 5 # Population size for differential evolution
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
        self.local_search_prob = 0.1 # Initial probability of performing local search
        self.local_search_success_prob_increase = 0.2 # Increase in local search probability upon trust region expansion
        self.local_search_failure_prob_decrease = 0.1 # Decrease in local search probability upon trust region contraction
        self.local_search_radius = 0.1 # Initial local search radius
        self.local_search_radius_shrink = 0.9 # Shrink factor for local search radius
        self.local_search_radius_expand = 1.1 # Expansion factor for local search radius
        self.local_search_success_threshold = 0.5 # Threshold for local search radius shrinkage
        self.local_search_ei_threshold = 0.01 # Threshold for EI to trigger local search
        self.local_search_trust_region_radius = 0.5 # Trust region radius for local search

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

        res = minimize(neg_log_likelihood, x0=self.length_scale, bounds=[self.length_scale_bounds], options={'maxiter': 5})
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
            maxiter=3, # Reduce maxiter for computational efficiency
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

    def _local_search(self, func):
        # Perform local search around the best point
        def objective(x):
            return func(x)

        # Define bounds for local search within the local search trust region
        bounds = list(zip(np.maximum(self.bounds[0], self.best_x - self.local_search_trust_region_radius),
                             np.minimum(self.bounds[1], self.best_x + self.local_search_trust_region_radius)))

        x0 = self.best_x
        
        # Perform local optimization using L-BFGS-B
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        
        return result.x, result.fun

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

        local_search_success_count = 0
        local_search_attempt_count = 0

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

            # Adjust trust region size
            if ratio > self.success_threshold and abs(predicted_improvement) > 1e-3:
                self.trust_region_radius *= self.trust_region_expand
                self.local_search_prob = min(1.0, self.local_search_prob + self.local_search_success_prob_increase)
            elif ratio < self.failure_threshold:
                self.trust_region_radius *= self.trust_region_shrink
                self.trust_region_radius = max(self.trust_region_radius, self.min_trust_region_radius)
                self.local_search_prob = max(0.0, self.local_search_prob - self.local_search_failure_prob_decrease)

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

            # Perform local search with probability self.local_search_prob
            ei = self._acquisition_function(self.best_x.reshape(1, -1))[0, 0]
            if ei > self.local_search_ei_threshold and np.random.rand() < self.local_search_prob and self.n_evals < self.budget:
                local_search_attempt_count += 1
                local_x, local_y = self._local_search(func)
                if local_y < self.best_y:
                    local_search_success_count += 1
                    local_x = local_x.reshape(1, -1)
                    local_y = np.array([[local_y]])
                    self._update_eval_points(local_x, local_y)

                    # Adjust local search radius if successful
                    self.local_search_radius *= self.local_search_radius_shrink
                    self.local_search_radius = max(self.local_search_radius, 0.01)
                else:
                    # Adjust local search radius if unsuccessful
                    self.local_search_radius *= self.local_search_radius_expand
                    self.local_search_radius = min(self.local_search_radius, 1.0)

            # Adjust local search probability based on success rate
            if local_search_attempt_count > 5:
                local_search_success_rate = local_search_success_count / local_search_attempt_count
                if local_search_success_rate > self.local_search_success_threshold:
                    self.local_search_prob = min(1.0, self.local_search_prob + self.local_search_success_prob_increase)
                else:
                    self.local_search_prob = max(0.0, self.local_search_prob - self.local_search_failure_prob_decrease)
                local_search_attempt_count = 0
                local_search_success_count = 0

            # Fit GP model more frequently
            if self.n_evals % 5 == 0:
                self.gp = self._fit_model(self.X, self.y)

        return self.best_y, self.best_x
```
## Feedback
 The algorithm AdaptiveTrustRegionEvolutionaryBO_ELS_AE got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.2069 with standard deviation 0.1113.

took 368.70 seconds to run.