from .population import Population
from ..individual import Individual

class SequencePopulation(Population):
    """
    Represents a population of individuals in the evolutionary algorithm.
    """

    def __init__(self):
        super().__init__()
        self.individuals: list[Individual] = []

    def get_population_size(self):
        return len(self.individuals)

    def add_individual(self, individual):
        self.individuals.append(individual)

    def remove_individual(self, individual):
        self.individuals = [ind for ind in self.individuals if ind.id != individual.id]

    def get_offspring_queryitems(self, n_parent:int=None) -> list[list[Individual]]:
        if not self.individuals:
            return [[]]

        return [self.individuals[-1:]]

    def all_individuals(self):
        return self.individuals

    def get_best_individual(self, maximize=False):
        if not self.individuals:
            return None

        best = self.individuals[0]
        for ind in self.individuals:
            if maximize:
                if ind.fitness > best.fitness:
                    best = ind
            else:
                if ind.fitness < best.fitness:
                    best = ind
        return best