
from enum import Enum
from typing import Callable
import logging
import numpy as np
from .population import Population, PopulationQueryItem, desc_similarity, score_to_probability_with_logarithmic_curve
from .es_population import ESPopulation
from ..individual import Individual


class IslandESPopulation(Population):

    class IslandStatus(Enum):
        INITIAL = 0
        GROWING = 1
        MATURE = 2
        RESETING = 3
        KILLED = 4

    class IslandAge(Enum):
        WARMUP = 0
        CAMBRIAN = 1
        NEOGENE = 2

    class IslandPopWrapper:
        def __init__(self, pop:ESPopulation, status):
            self.pop:ESPopulation = pop
            self.status = status
            self.island_id = None
            self.growing_gen = 1

        def update_growth(self):
            if self.status == IslandESPopulation.IslandStatus.GROWING:
                self.growing_gen += 1
            else:
                self.growing_gen = 1

    def __init__(self, n_islands: int = 1,
                 n_parent: int = 1,
                 n_offspring: int = 1,
                 n_parent_per_offspring: int = 1,
                 use_elitism: bool = True,
                 crossover_rate: float = 0.6,
                 update_strategy: Callable[[list[Individual], list[Individual], int], list[Individual]] = None,
                 selection_strategy: Callable[[list[Individual], list[Individual], int], list[Individual]] = None,
                 preoder_aware_init: bool = False, 
                 cyclic_migration: bool = False, 
                 migration_batch: int = 1,
                 n_warmup_generations: int = 3,
                 n_cambrian_generations: int = 10,
                 n_neogene_generations: int = 10,):
        super().__init__()

        self.generation = 0
        self.geo_age = self.IslandAge.WARMUP
        self.migration_batch = migration_batch
        self.cyclic_migration = cyclic_migration

        self.n_islands = n_islands
        self.n_parent = n_parent
        self.n_offspring = n_offspring
        self.n_parent_per_offspring = n_parent_per_offspring
        self.use_elitism = use_elitism

        self.n_warmup_generations = n_warmup_generations
        self.n_cambrian_generations = n_cambrian_generations
        self.n_neogene_generations = n_neogene_generations

        self.reset_rate = 0.5
        self.kill_rate = 0.5

        self.preorder_aware_init = preoder_aware_init

        self.populations = []
        for i in range(n_islands):
            wrapper = IslandESPopulation.IslandPopWrapper(ESPopulation(n_parent, n_parent_per_offspring, n_offspring, use_elitism), IslandESPopulation.IslandStatus.INITIAL)
            wrapper.island_id = i
            wrapper.pop.preorder_aware_init = self.preorder_aware_init
            wrapper.pop.selection_strategy = selection_strategy
            wrapper.pop.get_parent_strategy = update_strategy
            wrapper.cross_over_rate = crossover_rate
            self.populations.append(wrapper)

        self.__preorder_init_queue = []
        if self.preorder_aware_init:
            for wrapper in self.populations:
                self.__preorder_init_queue.append(wrapper.island_id)


    def __gen_to_age(self, generation: int):
        if generation < self.n_warmup_generations:
            return self.IslandAge.WARMUP
        else:
            evol_gen = generation - self.n_warmup_generations
            rest = evol_gen % (self.n_cambrian_generations + self.n_neogene_generations)
            if rest < self.n_cambrian_generations:
                return self.IslandAge.CAMBRIAN
            else:
                return self.IslandAge.NEOGENE

    def __is_the_end_of_age(self, generation: int, age: IslandAge):
        if age == self.IslandAge.WARMUP:
            return generation == self.n_warmup_generations - 1
        elif age == self.IslandAge.CAMBRIAN:
            return (generation - self.n_warmup_generations) % (self.n_cambrian_generations + self.n_neogene_generations) == self.n_cambrian_generations - 1
        elif age == self.IslandAge.NEOGENE:
            return (generation - self.n_warmup_generations) % (self.n_cambrian_generations + self.n_neogene_generations) == self.n_cambrian_generations + self.n_neogene_generations - 1
        return False

    def __is_the_start_of_age(self, generation: int, age: IslandAge):
        if age == self.IslandAge.WARMUP:
            return generation == 0
        elif age == self.IslandAge.CAMBRIAN:
            return (generation - self.n_warmup_generations) % (self.n_cambrian_generations + self.n_neogene_generations) == 0
        elif age == self.IslandAge.NEOGENE:
            return (generation - self.n_warmup_generations) % (self.n_cambrian_generations + self.n_neogene_generations) == self.n_cambrian_generations
        return False
        
    def __set_island_index(self, individual: Individual, island_index: int):
        setattr(individual, "island_index", island_index)

    def __get_island_index(self, individual: Individual):
        index = getattr(individual, "island_index", None)
        return index

    def get_population_size(self):
        return sum([wrapper.pop.get_population_size() for wrapper in self.populations])

    def add_individual(self, individual: Individual, generation: int):
        island_index = self.__get_island_index(individual)
        if island_index is not None:
            self.populations[island_index].pop.add_individual(individual, generation)
            self.__set_island_index(individual, island_index)
        else:
            # raise ValueError("Individual does not belong to any island.")
            pass
    
    def remove_individual(self, individual):
        island_index = self.__get_island_index(individual)
        if island_index is not None:
            self.populations[island_index].remove_individual(individual)

    def __get_all_mature_individuals(self, sort: bool = False, revserse: bool = False):
        all_mature_inds = []
        for wrapper in self.populations:
            if wrapper.status == self.IslandStatus.MATURE:
                ind = wrapper.pop.get_best_individual()
                if ind is not None:
                    all_mature_inds.append(ind)
        if sort:
            all_mature_inds = sorted(all_mature_inds, key=lambda x: x.fitness, reverse=revserse)
        return all_mature_inds

    def __get_all_best_individuals(self, sort: bool = False, revserse: bool = False):
        all_best_inds = []
        for wrapper in self.populations:
            if wrapper.status == self.IslandStatus.KILLED:
                continue
            ind = wrapper.pop.get_best_individual()
            if ind is not None:
                all_best_inds.append(ind)

        if sort:
            all_best_inds = sorted(all_best_inds, key=lambda x: x.fitness, reverse=revserse)
        return all_best_inds

    def select_next_generation(self):
        should_update = False
        if self.preorder_aware_init and len(self.__preorder_init_queue) > 0:
            island_index = self.__preorder_init_queue[0]
            wrapper = self.populations[island_index]
            wrapper.pop.select_next_generation()
            if wrapper.pop.get_current_generation() == 1:
                self.__preorder_init_queue.pop(0)

            if len(self.__preorder_init_queue) == 0:
                should_update = True
        else:
            should_update = True
            for wrapper in self.populations:
                status = wrapper.status
                if status != self.IslandStatus.KILLED:
                    wrapper.pop.select_next_generation()

                    if status == self.IslandStatus.INITIAL:
                        wrapper.status = self.IslandStatus.GROWING
                    elif status == self.IslandStatus.RESETING:
                        wrapper.status = self.IslandStatus.GROWING
                    elif status == self.IslandStatus.GROWING:
                        wrapper.update_growth()
                        if wrapper.growing_gen >= self.n_warmup_generations:
                            wrapper.status = self.IslandStatus.MATURE
                            wrapper.update_growth()

        if not should_update:
            return

        if self.geo_age == self.IslandAge.NEOGENE:
            if self.__is_the_end_of_age(self.generation, self.IslandAge.NEOGENE):
                all_best_inds = self.__get_all_mature_individuals(sort=True)
                # kill the worst kill_rate of the islands
                n_islands = len(all_best_inds)
                n_killed = int(n_islands * self.kill_rate)
                if n_islands - n_killed < 1:
                    n_killed = n_islands - 1
                n_killed = max(0, n_killed)
                for i in range(n_killed):
                    ind = all_best_inds[i]
                    island_index = self.__get_island_index(ind)
                    self.populations[island_index].status = self.IslandStatus.KILLED
                    logging.info("Island %s is killed.", island_index)

        self.generation += 1
        # update the age of the islands
        self.geo_age = self.__gen_to_age(self.generation)

        # update the status of the islands 
        if self.geo_age == self.IslandAge.CAMBRIAN:
            if self.__is_the_start_of_age(self.generation, self.IslandAge.CAMBRIAN):
                all_best_inds = self.__get_all_mature_individuals(sort=True)
                # reset the worst reset_rate of individuals
                n_islands = len(all_best_inds)
                n_reset = int(n_islands * self.reset_rate)
                if n_islands - n_reset < 1:
                    n_reset = n_islands - 1
                n_reset = max(0, n_reset)
                for i in range(n_reset):
                    ind = all_best_inds[i]
                    island_index = self.__get_island_index(ind)
                    self.populations[island_index].status = self.IslandStatus.RESETING
                    logging.info("Island %s is reseting.", island_index)


    def __get_prob_migration_queryitems(self, all_best_inds: list[Individual], migration_batch: int, wrapper: IslandPopWrapper, similarity_matrix: np.ndarray) -> list[PopulationQueryItem]:

        all_best_inds = sorted(all_best_inds, key=lambda x: x.fitness, reverse=True)

        max_fitness = all_best_inds[0].fitness
        mean_fitness = np.mean([ind.fitness for ind in all_best_inds])
        my_best_ind = wrapper.pop.get_best_individual()
        migration_prob = score_to_probability_with_logarithmic_curve(score=my_best_ind.fitness, max_score=min(1.0,max_fitness*1.5))

        migration_batch = min(self.migration_batch, len(all_best_inds))
        migration_batch = max(wrapper.pop.n_parent_per_offspring-1, migration_batch) 

        if np.random.random() > migration_prob:
            return wrapper.pop.get_offspring_queryitems()
        else:
            migrant_parent = []
            if my_best_ind.fitness >= mean_fitness:
                # diversity
                # find the index of the individual
                sim_index = -1
                for i, ind in enumerate(all_best_inds):
                    if ind.id == my_best_ind.id:
                        sim_index = i
                        break
                # get the least similar individuals
                if sim_index >= 0:
                    sim_inds = np.argsort(similarity_matrix[sim_index])
                    for i in range(migration_batch):
                        ind = all_best_inds[sim_inds[i]]
                        migrant_parent.append(ind)
            else:
                # fitness
                migrant_parent = [all_best_inds[i] for i in range(migration_batch)]

            # get the rest of the parents from the population
            items = []
            n_parent = wrapper.pop.n_parent_per_offspring - len(migrant_parent)
            p_items = wrapper.pop.get_offspring_queryitems(n_parent=n_parent)
            for p_item in p_items:
                parent = p_item.parent + migrant_parent
                query_item = PopulationQueryItem(qid=wrapper.island_id, parent=parent, offspring=Individual())
                items.append(query_item)
        return items

    def get_offspring_queryitems(self, n_parent:int=None) -> list[PopulationQueryItem]:

        if self.preorder_aware_init and len(self.__preorder_init_queue) > 0:
            logging.info("Generation %s, Age %s", self.generation, self.geo_age)
            island_index = self.__preorder_init_queue[0]
            wrapper = self.populations[island_index]
            items = wrapper.pop.get_offspring_queryitems()
            migrant = self.all_individuals()
            for item in items:
                item.parent = migrant
                self.__set_island_index(item.offspring, island_index)
            return items
            
        # migration: get individuals from other islands(diversity)
        # killed: ignore this island
        # reseting: return empty parents
        parents = []
        all_best_inds = self.__get_all_best_individuals()
        similarity_matrix = None
        if self.geo_age == self.IslandAge.CAMBRIAN and len(all_best_inds) > 1:
            _, similarity_matrix = desc_similarity(all_best_inds)

        logging.info("Generation %s, Age %s, %s Islands ", self.generation, self.geo_age, len(all_best_inds))

        for wrapper in self.populations:
            status = wrapper.status
            items = []
            if status == self.IslandStatus.INITIAL:
                items = wrapper.pop.get_offspring_queryitems()
            elif status == self.IslandStatus.GROWING:
                n_parent_per_offspring = wrapper.pop.n_parent_per_offspring
                items = wrapper.pop.get_offspring_queryitems(n_parent = n_parent_per_offspring)
            elif status == self.IslandStatus.MATURE:
                if self.geo_age == self.IslandAge.CAMBRIAN:
                    if len(all_best_inds) > 1:
                        if self.cyclic_migration:
                            # TODO: cyclic migration
                            items = wrapper.pop.get_offspring_queryitems()
                        else:
                            query_items = self.__get_prob_migration_queryitems(all_best_inds, self.migration_batch, wrapper, similarity_matrix)
                            items.extend(query_items)
                    else:
                        items = wrapper.pop.get_offspring_queryitems()

                elif self.geo_age == self.IslandAge.NEOGENE:
                    items = wrapper.pop.get_offspring_queryitems()
            elif status == self.IslandStatus.KILLED:
                continue
            elif status == self.IslandStatus.RESETING:
                for _ in range(wrapper.pop.n_offspring):
                    sample_size = min(wrapper.pop.n_parent_per_offspring, len(all_best_inds))
                    parent = np.random.choice(all_best_inds, size=sample_size, replace=False).tolist()
                    query_item = PopulationQueryItem(qid=wrapper.island_id, parent=parent, offspring=Individual())
                    query_item.is_initialized = True
                    items.append(query_item)
            # Set the island index for the offspring
            for query_item in items:
                self.__set_island_index(query_item.offspring, wrapper.island_id)
            parents.extend(items)

        return parents

    def get_individuals(self, generation = None):
        inds = []
        gen = generation if generation is not None else self.generation
        for wrapper in self.populations:
            inds.extend(wrapper.pop.get_individuals(gen))
        return inds
    
    def get_offsprings(self, generation = None):
        inds = []
        gen = generation if generation is not None else self.generation
        for wrapper in self.populations:
            inds.extend(wrapper.pop.get_offsprings(gen))
        return inds
    
    def get_current_generation(self):
        return self.generation

    def get_best_individual(self, maximize: bool = False):
        best = None
        for wrapper in self.populations:
            ind = wrapper.pop.get_best_individual(maximize)
            if best is None or (ind is not None and ind.fitness > best.fitness):
                best = ind
        return best

    def all_individuals(self):
        all_inds = []
        for wrapper in self.populations:
            all_inds.extend(wrapper.pop.all_individuals())
        return all_inds
