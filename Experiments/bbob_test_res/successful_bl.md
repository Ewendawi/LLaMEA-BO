# Description: A population-based algorithm that combines local search with a novel adaptive neighborhood exploration strategy, using a decaying radius and a probabilistic selection of neighbors based on fitness.
# Code:
```python
import numpy as np

class AdaptiveNeighborhoodSearch:
    def __init__(self, budget=10000, dim=10, population_size=20, initial_radius=1.0, min_radius=0.001, radius_decay=0.95, neighbor_prob=0.8, stagnation_threshold=50):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.initial_radius = initial_radius
        self.min_radius = min_radius
        self.radius_decay = radius_decay
        self.neighbor_prob = neighbor_prob
        self.stagnation_threshold = stagnation_threshold
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = None
        self.fitness = None
        self.radius = self.initial_radius
        self.eval_count = 0
        self.stagnation_count = 0

    def initialize_population(self, func):
        lb = -5.0
        ub = 5.0
        self.population = np.random.uniform(lb, ub, size=(self.population_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count += self.population_size
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.f_opt:
            self.f_opt = self.fitness[best_idx]
            self.x_opt = self.population[best_idx].copy()

    def create_neighbor(self, x, lb, ub):
        neighbor = x.copy()
        if np.random.rand() < self.neighbor_prob:
            random_dim = np.random.randint(self.dim)
            neighbor[random_dim] += np.random.uniform(-self.radius, self.radius)
        else:
            neighbor += np.random.uniform(-self.radius, self.radius, size=self.dim)
        return np.clip(neighbor, lb, ub)

    def update_population(self, func):
        lb = -5.0
        ub = 5.0
        new_population = np.zeros_like(self.population)
        new_fitness = np.zeros_like(self.fitness)

        for i in range(self.population_size):
            neighbor = self.create_neighbor(self.population[i], lb, ub)
            neighbor_fitness = func(neighbor)
            self.eval_count += 1

            if neighbor_fitness < self.fitness[i]:
                new_population[i] = neighbor
                new_fitness[i] = neighbor_fitness
            else:
                new_population[i] = self.population[i]
                new_fitness[i] = self.fitness[i]
        
        self.population = new_population
        self.fitness = new_fitness
        
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.f_opt:
            self.f_opt = self.fitness[best_idx]
            self.x_opt = self.population[best_idx].copy()
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

    def restart_population(self, func):
        self.initialize_population(func)
        self.radius = self.initial_radius
        self.stagnation_count = 0

    def __call__(self, func):
        self.initialize_population(func)
        while self.eval_count < self.budget:
            self.update_population(func)
            self.radius = max(self.radius * self.radius_decay, self.min_radius)
            if self.stagnation_count > self.stagnation_threshold:
                self.restart_population(func)
        return self.f_opt, self.x_opt
```