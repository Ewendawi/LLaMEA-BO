# Description
**BayesianEnsembleWithAdaptiveESBO (BEAESBO)**: This algorithm builds upon the BayesianEnsembleBO by introducing an adaptive evolutionary strategy (ES) for selecting the next points. The adaptation lies in dynamically adjusting the mutation rate of the ES based on the success rate of previous mutations. If mutations are consistently leading to improvements in the acquisition function, the mutation rate is decreased to promote exploitation. Conversely, if mutations are not yielding better solutions, the mutation rate is increased to encourage exploration. Additionally, a local search is performed around the best point found so far using a gradient-free optimization method (Nelder-Mead) to refine the solution. This aims to improve the exploitation capabilities of the algorithm.

# Justification
1.  **Adaptive Mutation Rate:** The core idea is to control the exploration-exploitation trade-off within the evolutionary strategy. By adapting the mutation rate based on the recent success of mutations, the algorithm can dynamically shift its focus between exploring new regions and refining existing promising solutions.

2.  **Local Search:** Adding a local search step using Nelder-Mead around the current best solution enhances the algorithm's ability to converge to a local optimum. This is particularly useful in the later stages of the optimization process when the algorithm has already identified a promising region.

3.  **Computational Efficiency:** The Nelder-Mead method is a gradient-free optimization technique, which makes it suitable for black-box optimization problems where gradient information is not available. Also, the adaptive mutation rate helps to maintain a balance between exploration and exploitation, which can lead to faster convergence and better performance.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel
from scipy.optimize import minimize, differential_evolution, shgo, dual_annealing, basinhopping, direct

class BayesianEnsembleWithAdaptiveESBO:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = 2 * (dim + 1)
        self.ensemble_size = 3
        self.kernels = [
            C(1.0, constant_value_bounds=(1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
            C(1.0, constant_value_bounds=(1e-2, 1e2)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5),
            C(1.0, constant_value_bounds=(1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
        ]
        self.gps = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-5) for kernel in self.kernels]
        self.update_interval = 5
        self.last_gp_update = 0
        self.mutation_rate = 0.1
        self.success_rate = 0.0
        self.success_history = []

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
        for gp in self.gps:
            gp.fit(X, y)
        return self.gps

    def _acquisition_function(self, X):
        # Implement acquisition function
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)
        if self.X is None or self.y is None:
            return np.zeros((len(X), 1))

        mu_list = []
        sigma_list = []
        for gp in self.gps:
            mu, sigma = gp.predict(X, return_std=True)
            mu_list.append(mu)
            sigma_list.append(sigma)

        mu_ensemble = np.mean(mu_list, axis=0)
        sigma_ensemble = np.sqrt(np.mean(np.square(sigma_list) + np.square(mu_list), axis=0) - np.square(mu_ensemble)) #ensemble variance

        sigma_ensemble = np.clip(sigma_ensemble, 1e-9, np.inf)
        y_best = np.min(self.y)
        gamma = (y_best - mu_ensemble) / sigma_ensemble
        ei = sigma_ensemble * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Adaptive evolutionary strategy
        population_size = 50
        n_generations = 10

        # Initialize population
        population = self._sample_points(population_size)
        fitness = self._acquisition_function(population).flatten()
        best_index = np.argmax(fitness)
        best_fitness = fitness[best_index]
        success_count = 0
        
        for _ in range(n_generations):
            # Select parents (tournament selection)
            selected_indices = np.random.choice(population_size, size=population_size, replace=True)
            parents = population[selected_indices]

            # Create offspring (mutation)
            offspring = parents.copy()
            for i in range(population_size):
                for j in range(self.dim):
                    if np.random.rand() < self.mutation_rate:
                        offspring[i, j] += np.random.normal(0, 0.5)  # Mutate with Gaussian noise
                        offspring[i, j] = np.clip(offspring[i, j], self.bounds[0][j], self.bounds[1][j])  # Clip to bounds

            # Evaluate offspring
            fitness_offspring = self._acquisition_function(offspring).flatten()
            
            # Replace parents with offspring if better
            for i in range(population_size):
                if fitness_offspring[i] > fitness[i]:
                    population[i] = offspring[i]
                    fitness[i] = fitness_offspring[i]
                    success_count +=1

            best_index = np.argmax(fitness)
            if fitness[best_index] > best_fitness:
                best_fitness = fitness[best_index]

        # Update mutation rate
        self.success_history.append(success_count / (population_size * n_generations))
        if len(self.success_history) > 5:
            self.success_history.pop(0)
        self.success_rate = np.mean(self.success_history)

        if self.success_rate > 0.3:
            self.mutation_rate *= 0.8  # Reduce mutation rate if successful
        else:
            self.mutation_rate *= 1.2  # Increase mutation rate if not successful
        self.mutation_rate = np.clip(self.mutation_rate, 0.01, 0.5)

        # Select best points from the final population
        selected_indices = np.argsort(fitness)[-batch_size:]
        selected_points = population[selected_indices]

        # Local search around the best point
        best_point = population[np.argmax(fitness)]
        res = minimize(lambda x: -self._acquisition_function(x.reshape(1, -1)), best_point, bounds=list(zip(self.bounds[0], self.bounds[1])), method='Nelder-Mead')
        if -self._acquisition_function(res.x.reshape(1, -1))[0][0] > best_fitness:
            selected_points[-1] = res.x #replace the last point with the locally optimized point

        return selected_points

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
    
    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64.
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initial exploration
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx][0]
        best_x = self.X[best_idx]
        
        while self.n_evals < self.budget:
            # Fit the Gaussian Process model
            if self.n_evals - self.last_gp_update >= self.update_interval:
                self._fit_model(self.X, self.y)
                self.last_gp_update = self.n_evals

            # Select the next points to evaluate
            batch_size = min(self.budget - self.n_evals, 5)
            next_X = self._select_next_points(batch_size)

            # Evaluate the points
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Update the best solution
            best_idx = np.argmin(self.y)
            best_y = self.y[best_idx][0]
            best_x = self.X[best_idx]

        return best_y, best_x
```
## Feedback
 The algorithm BayesianEnsembleWithAdaptiveESBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1542 with standard deviation 0.1023.

took 346.91 seconds to run.