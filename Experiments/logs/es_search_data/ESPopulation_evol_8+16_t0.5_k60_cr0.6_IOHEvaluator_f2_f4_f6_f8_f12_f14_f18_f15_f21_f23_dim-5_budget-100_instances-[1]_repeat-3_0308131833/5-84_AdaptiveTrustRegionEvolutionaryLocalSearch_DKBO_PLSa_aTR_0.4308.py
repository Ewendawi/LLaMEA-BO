from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import differential_evolution, minimize

class AdaptiveTrustRegionEvolutionaryLocalSearch_DKBO_PLSa_aTR:
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
        self.exploration_weight = 0.1  # Initial weight for exploration
        self.exploration_weight_decay = 0.99 # Decay rate for exploration weight
        self.length_scale = 1.0 #Initial length scale
        self.length_scale_bounds = (1e-2, 1e2) #Bounds for the length scale
        self.local_search_radius = 0.1
        self.local_search_success_rate = 0.0
        self.local_search_success_memory = []
        self.local_search_success_window = 5
        self.ei_threshold = 0.01 # Threshold for performing local search
        self.local_search_probability = 0.1 # Initial probability of performing local search
        self.min_trust_region_radius = 0.1 # Minimum trust region radius
        self.predicted_improvement_threshold = 1e-6 # Threshold for predicted improvement
        self.kernel_tuning_frequency = 5 # Tune kernel every n iterations

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
            if self.n_evals % self.kernel_tuning_frequency == 0:
                self.gp = self._fit_model(self.X, self.y)
            else:
                # Fit GP model without tuning length scale
                kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=self.length_scale, length_scale_bounds="fixed")
                self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-6)
                self.gp.fit(self.X, self.y)

            # Select points by acquisition function using differential evolution
            batch_size = min(1, self.budget - self.n_evals) # Batch size of 1 for evolutionary strategy
            next_X = self._select_next_points(batch_size)
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Calculate the ratio of improvement
            predicted_improvement = self.best_y - self.gp.predict(self.best_x.reshape(1, -1))[0]
            actual_improvement = self.best_y - min(self.y)[0]

            if abs(predicted_improvement) < self.predicted_improvement_threshold:
                ratio = 0.0
            else:
                ratio = actual_improvement / predicted_improvement

            # Adjust trust region size
            if ratio > self.success_threshold and abs(predicted_improvement) > self.predicted_improvement_threshold:
                self.trust_region_radius *= self.trust_region_expand
            elif ratio < self.failure_threshold:
                self.trust_region_radius *= self.trust_region_shrink
            
            # Adjust trust region based on local search success
            if self.local_search_success_rate > 0.6:
                self.trust_region_radius *= 1.1
            elif self.local_search_success_rate < 0.3:
                self.trust_region_radius *= 0.9

            self.trust_region_radius = max(self.trust_region_radius, self.min_trust_region_radius)

            # Update trust region center
            self.trust_region_center = self.best_x
            
            # Local search around the best solution
            if self.best_x is not None and self.n_evals < self.budget:
                # Calculate EI value
                ei = self._acquisition_function(self.best_x.reshape(1, -1))[0, 0]

                # Calculate local search probability
                self.local_search_probability = 0.1 + 0.9 * norm.cdf(ei / self.ei_threshold) * self.local_search_success_rate
                self.local_search_probability = np.clip(self.local_search_probability, 0.01, 0.99)

                # Perform local search with probability
                if np.random.rand() < self.local_search_probability:
                    local_X = self._sample_points(1) * self.local_search_radius + self.best_x
                    local_X = np.clip(local_X, self.bounds[0], self.bounds[1])
                    local_y = self._evaluate_points(func, local_X)
                    self._update_eval_points(local_X, local_y)

                    # Update local search success rate
                    if local_y[0, 0] < self.best_y:
                        self.local_search_success_memory.append(1)
                    else:
                        self.local_search_success_memory.append(0)

                    if len(self.local_search_success_memory) > self.local_search_success_window:
                        self.local_search_success_memory.pop(0)

                    self.local_search_success_rate = np.mean(self.local_search_success_memory)

                    # Adjust local search radius
                    if self.local_search_success_rate > 0.5:
                        self.local_search_radius *= 0.9  # Reduce radius if successful
                    else:
                        self.local_search_radius *= 1.1  # Increase radius if unsuccessful
                    self.local_search_radius = np.clip(self.local_search_radius, 0.01, 1.0)

            # Decay exploration weight
            self.exploration_weight *= self.exploration_weight_decay
            self.exploration_weight = max(self.exploration_weight, 0.01)

        return self.best_y, self.best_x
