# Description
**BayesianEnsembleBO with Adaptive Mutation (BEBO-AM)**: This algorithm refines the BayesianEnsembleBO by introducing an adaptive mutation rate within the evolutionary strategy used for selecting the next points. The mutation rate is dynamically adjusted based on the diversity of the population and the improvement in the acquisition function. If the population diversity is low or the improvement stagnates, the mutation rate is increased to encourage exploration. Otherwise, it's decreased to focus on exploitation. This adaptive mutation aims to improve the balance between exploration and exploitation, leading to better optimization performance. Additionally, a simple restart mechanism is added to escape local optima.

# Justification
The key improvements are:

1.  **Adaptive Mutation Rate:** The original evolutionary strategy used a fixed mutation rate. Adapting the mutation rate allows for a more nuanced exploration-exploitation trade-off. When the population converges (low diversity), a higher mutation rate is beneficial to explore new regions. When the population is diverse and improving, a lower mutation rate helps to refine the search around promising areas.
2.  **Diversity Metric:** The diversity of the population is measured by the average pairwise distance between individuals. This provides a quantitative measure of how spread out the population is.
3.  **Stagnation Detection:** Stagnation is detected by tracking the improvement in the acquisition function over a few generations. If the improvement is below a threshold, it indicates that the search is stuck in a local optimum.
4.  **Restart Mechanism:** If stagnation is detected, the algorithm restarts the evolutionary strategy with a new, randomly sampled population. This helps to escape local optima and explore new regions of the search space.
5.  **Computational Efficiency:** The diversity and stagnation checks are performed periodically to avoid excessive computational overhead.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


class BayesianEnsembleBOAdaptiveMutation:
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
        self.diversity_threshold = 0.1 # Threshold for population diversity
        self.stagnation_threshold = 1e-3 # Threshold for improvement stagnation
        self.stagnation_generations = 3 # Number of generations to check for stagnation
        self.last_improvement = np.inf
        self.restart_flag = False

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

    def _calculate_population_diversity(self, population):
        # Calculate the average pairwise distance between individuals in the population
        if len(population) <= 1:
            return 1.0  # Maximum diversity if only one individual
        distances = pdist(population)
        avg_distance = np.mean(distances)
        # Normalize the diversity by the range of the search space
        normalized_diversity = avg_distance / (np.linalg.norm(self.bounds[1] - self.bounds[0]))
        return normalized_diversity

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

        # Simple evolutionary strategy
        population_size = 50
        n_generations = 10

        # Initialize population
        if self.restart_flag:
            population = self._sample_points(population_size)
            self.restart_flag = False
        else:
            population = self._sample_points(population_size)

        best_fitness = -np.inf

        for gen in range(n_generations):
            # Evaluate acquisition function
            fitness = self._acquisition_function(population).flatten()

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

            current_best_fitness = np.max(fitness)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness

        # Adaptive mutation rate
        diversity = self._calculate_population_diversity(population)
        if diversity < self.diversity_threshold:
            self.mutation_rate = min(self.mutation_rate * 1.2, 0.5)  # Increase mutation rate
        else:
            self.mutation_rate = max(self.mutation_rate * 0.8, 0.01)  # Decrease mutation rate

        # Stagnation detection and restart
        if best_fitness - self.last_improvement < self.stagnation_threshold:
            self.stagnation_counter += 1
            if self.stagnation_counter >= self.stagnation_generations:
                self.restart_flag = True
                self.stagnation_counter = 0
        else:
            self.stagnation_counter = 0
            self.last_improvement = best_fitness

        # Select best points from the final population
        fitness = self._acquisition_function(population).flatten()
        selected_indices = np.argsort(fitness)[-batch_size:]
        selected_points = population[selected_indices]

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
        self.stagnation_counter = 0
        self.last_improvement = np.inf
        self.restart_flag = False

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
 The algorithm BayesianEnsembleBOAdaptiveMutation got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.1525 with standard deviation 0.1045.

took 146.52 seconds to run.