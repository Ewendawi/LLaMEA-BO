from datetime import datetime
import logging
import os
import pickle
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
        self.save_per_generation = None
        self.save_per_generation_dir = None

        self.update_simliarity_per_generation = True

        self.individuals:dict[str, Individual] = {}
        # all individuals per generation
        self.generations:list[list[str]] = []
        # selected individuals per generation
        self.selected_generations:list[list[str]] = []

        if not self.use_elitism and self.n_parent > self.n_offspring:
            raise ValueError("n_parent should be less than or equal to n_offspring when not using elitism.")

        if self.n_parent_per_offspring * self.n_offspring > self.n_parent:
            raise ValueError("n_parent should be greater than or equal to n_parent_per_offspring * n_offspring.")


    def all_individuals(self):
        return self.individuals.values()

    def get_population_size(self):
        return len(self.individuals)
    
    def add_individual(self, individual: Individual, generation: int = 0):
        if generation >= len(self.generations):
            while len(self.generations) <= generation:
                self.generations.append([])
        self.individuals[individual.id] = individual
        self.generations[generation].append(individual.id)

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

        if self.update_simliarity_per_generation and self.n_parent > 1:
            # Update the similarity matrix
            mean_similarity, _ = desc_similarity([self.individuals[id] for id in next_pop])
            for i, ind_id in enumerate(next_pop):
                ind = self.individuals[ind_id]
                Population.get_handler_from_individual(ind).eval_result.similarity = mean_similarity[i]
        
        # Save population every n generations
        n_gen = len(self.selected_generations)
        if self.save_per_generation is not None and n_gen % self.save_per_generation == 0:
            dir_path = self.save_per_generation_dir
            if dir_path is None:
                time_stamp = datetime.now().strftime("%m%d%H%M%S")
                dir_path = f'Experiments/pop_temp/{self.__class__.__name__}_{self.name}_{time_stamp}'
                self.save_per_generation_dir = dir_path
            os.makedirs(dir_path, exist_ok=True)
            file_path = f'{dir_path}/pop_{n_gen}.pkl'
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)

    def get_current_generation(self):
        return len(self.selected_generations)

    def get_offspring_queryitems(self, n_parent:int=None) -> list[PopulationQueryItem]:
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

        if n_parent is not None:
            n_parent_per_offspring = n_parent
        else:
            if self.n_parent_per_offspring > 1:
                # decide mutation or crossover
                if np.random.rand() < self.cross_over_rate:
                    n_parent_per_offspring = self.n_parent_per_offspring
                else:
                    n_parent_per_offspring = 1
            else:
                n_parent_per_offspring = self.n_parent_per_offspring

        last_pop = self.selected_generations[-1]
        last_pop = [self.individuals[id] for id in last_pop if id in self.individuals]

        # custom parent selection strategy
        if self.get_parent_strategy is not None:
            return self.get_parent_strategy(last_pop, n_parent_per_offspring, self.n_offspring)
            
        # if donot have enough parents, repeat the last population
        n_last_pop_needed = n_parent_per_offspring * self.n_offspring
        if len(last_pop) < n_last_pop_needed:
            last_pop = last_pop * (n_last_pop_needed // len(last_pop) + 1)

        parents = []
        idx_last_pop = 0
        for _ in range(self.n_offspring):
            parent = last_pop[idx_last_pop: idx_last_pop+n_parent_per_offspring]
            parents.append(parent)
            idx_last_pop += n_parent_per_offspring

        query_items = []
        for parent in parents:
            query_item = PopulationQueryItem(qid=0, parent=parent, offspring=Individual())
            query_items.append(query_item)
            
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