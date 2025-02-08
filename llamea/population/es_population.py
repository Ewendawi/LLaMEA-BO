import logging
import math
import itertools
import numpy as np
from .population import Population, PopulationQueryItem, desc_similarity
from ..individual import Individual


class ESPopulation(Population):
    def __init__(self,n_parent:int=2, n_parent_per_offspring: int = 1, n_offspring: int = 1, use_elitism: bool = True):
        super().__init__()

        self.preorder_aware_init = False
        self.cross_over_rate = 0.6
        
        self.n_parent = n_parent
        self.n_parent_per_offspring = n_parent_per_offspring
        self.n_offspring = n_offspring
        self.use_elitism = use_elitism

        self.individuals:dict[str, Individual] = {}
        # all individuals per generation
        self.generations:list[list[str]] = []
        # selected individuals per generation
        self.selected_generations:list[list[str]] = []

        if not self.use_elitism and self.n_parent > self.n_offspring:
            raise ValueError("n_parent should be less than or equal to n_offspring when not using elitism.")

        if math.comb(n_parent, n_parent_per_offspring) < self.n_offspring:
            raise ValueError(f"n_parent({n_parent}) choose n_parent_per_offspring({n_parent_per_offspring}) is {math.comb(n_parent, n_parent_per_offspring)}. It should be greater than or equal to n_offspring({n_offspring}).")

    def all_individuals(self):
        return self.individuals.values()

    def get_population_size(self):
        return len(self.individuals)
    
    def add_individual(self, individual: Individual, generation: int = 0):
        if generation >= len(self.generations):
            while len(self.generations) <= generation:
                self.generations.append([])
        individual.generation = generation
        self.individuals[individual.id] = individual
        self.generations[generation].append(individual.id)

        if self.debug_save_on_the_fly:
            self.save_on_the_fly(individual=individual, generation=generation) 
       

    def remove_individual(self, individual):
        if individual.id in self.individuals:
            del self.individuals[individual.id]
            if individual.generation in self.generations:
                gen = self.generations[individual.generation]
                if individual.id in gen:
                    gen.remove(individual.id)
            else:
                for gen in self.generations:
                    if individual.id in gen:
                        gen.remove(individual.id)

    def select_next_generation(self):
        if len(self.generations) == 0 or len(self.generations[-1]) == 0:
            return

        # If the population is not full in the first generation and the init is preorder-aware, do not select next generation
        if self.get_current_generation() == 0 and self.preorder_aware_init and len(self.generations[0]) < self.n_parent:
            return

        last_gen = self.generations[-1]
        last_pop = []
        if len(self.selected_generations) > 0:
            last_pop = self.selected_generations[-1]

        if self.selection_strategy is None:
            candidates = None
            if self.use_elitism:
                candidates = last_gen + last_pop
            else:
                candidates = last_gen
                if len(candidates) < self.n_parent:
                    logging.warning("Population size is less than n_parent. Using elitism.")
                    candidates = candidates + last_pop

            next_candidates = sorted(candidates, key=lambda x: self.individuals[x].fitness, reverse=True)
            next_pop = next_candidates[:self.n_parent]
            self.selected_generations.append(next_pop)
        else:
            ind_last_gen = [self.individuals[id] for id in last_gen]
            ind_last_pop = [self.individuals[id] for id in last_pop]
            ind_next_pop = self.selection_strategy(ind_last_gen, ind_last_pop, self.n_parent)
            next_pop = [ind.id for ind in ind_next_pop]
            self.selected_generations.append(next_pop)

        # Save population every n generations
        n_gen = len(self.selected_generations)
        if self.save_per_generation is not None and n_gen % self.save_per_generation == 0:
            _suffix = f'checkpoint_{n_gen}'
            self.save(suffix=_suffix)

    def get_current_generation(self):
        return len(self.selected_generations)

    def get_offspring_queryitems(self, n_parent:int=None, max_n_offspring:int=None) -> list[PopulationQueryItem]:
        if len(self.selected_generations) == 0:
            # Initial population
            if self.preorder_aware_init:
                parents = [self.individuals[id] for id in self.generations[0]] if len(self.generations) > 0 else []
                query_item = PopulationQueryItem(parent=parents, offspring=Individual())
                query_item.is_initialized = True
                return [query_item]
            else:
                query_items = []
                for i in range(self.n_parent):
                    query_item = PopulationQueryItem(qid=i, parent=[], offspring=Individual())
                    query_item.is_initialized = True
                    query_items.append(query_item)
                return query_items

        _n_parent_per_offspring = self.n_parent_per_offspring if n_parent is None else n_parent
        _n_offspring = self.n_offspring
        if max_n_offspring is not None and max_n_offspring < self.n_offspring:
            _n_offspring = max_n_offspring

        last_pop = self.selected_generations[-1]
        last_pop = [self.individuals[id] for id in last_pop if id in self.individuals]
        sorted_last_pop = sorted(last_pop, key=lambda x: x.fitness, reverse=True)

        # custom parent selection strategy
        if self.get_parent_strategy is not None:
            return self.get_parent_strategy(sorted_last_pop, _n_parent_per_offspring, _n_offspring)
            
        parents = []
        n_comb = list(itertools.combinations(range(len(sorted_last_pop)), _n_parent_per_offspring))
        one_comb = list(range(len(sorted_last_pop)))
        for _ in range(_n_offspring):
            _n_parent = _n_parent_per_offspring
            if _n_parent_per_offspring > 1 and np.random.rand() > self.cross_over_rate:
                _n_parent = 1

            if _n_parent == 1 and len(one_comb) > 0:
                parent_index = one_comb.pop(0)
                parents.append([sorted_last_pop[parent_index]])
            else:
                parent_index = n_comb.pop(0)
                parents.append([sorted_last_pop[i] for i in parent_index])
            
        query_items = []
        mutation_count = 0
        crossover_count = 0
        for parent in parents:
            if len(parent) == 1:
                mutation_count += 1
            else:
                crossover_count += 1
            query_item = PopulationQueryItem(qid=0, parent=parent, offspring=Individual())
            query_items.append(query_item)

        logging.info("Mutation: %d, Crossover: %d", mutation_count, crossover_count)
            
        return query_items

    def get_individuals(self, generation: int = None):
        gen = generation
        if gen is None:
            gen = len(self.selected_generations) - 1
        if gen < 0 or gen >= len(self.selected_generations):
            return []
        else:
            ind_ids = self.selected_generations[gen]
            inds = [self.individuals[id] for id in ind_ids]
            return inds

    def get_offsprings(self, generation: int = None):
        gen = generation
        if gen is None:
            gen = len(self.generations) - 1
        if gen < 0 or gen >= len(self.generations):
            return []
        else:
            ind_ids = self.generations[gen]
            inds = [self.individuals[id] for id in ind_ids]
            return inds

    def get_best_individual(self, maximize: bool = False):
        best = None
        if len(self.selected_generations) == 0:
            return best

        inds = [self.individuals[id] for id in self.selected_generations[-1]]
        best = inds[0]
        for ind in inds:
            if maximize:
                if ind.fitness > best.fitness:
                    best = ind
            else:
                if ind.fitness < best.fitness:
                    best = ind
        return best