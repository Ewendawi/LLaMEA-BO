import logging
import math
import itertools
import numpy as np
from .population import Population, PopulationQueryItem, desc_similarity
from ..individual import Individual

class ESPopulationQueryItemType:
    INIT = "INIT"
    EXCLUSIVE_CROSSOVER = "EXCLUSIVE_CROSSOVER"
    EXCLUSIVE_MUTATION = "EXCLUSIVE_MUTATION"
    CROSSOVER = "CROSSOVER"
    MUTATION = "MUTATION"

class ESPopulation(Population):
    def __init__(self,n_parent:int=2, n_parent_per_offspring: int = 1, n_offspring: int = 1, use_elitism: bool = True):
        super().__init__()

        self.preorder_aware_init = False
        self.cross_over_rate = 0.6
        self.exclusive_operations = True
        self.random_parent_selection = False
        self.replaceable_parent_selection = True

        self.light_cross_over_evaluator = None
        self.light_cross_over_promptor = None
        
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

        if self.n_parent_per_offspring > self.n_parent:
            raise ValueError("n_parent_per_offspring should be less than or equal to n_parent.")

        # if math.comb(n_parent, n_parent_per_offspring) < self.n_offspring:
        #     raise ValueError(f"n_parent({n_parent}) choose n_parent_per_offspring({n_parent_per_offspring}) is {math.comb(n_parent, n_parent_per_offspring)}. It should be greater than or equal to n_offspring({n_offspring}).")

    def __str__(self):
        return self.desc_settings()

    def desc_settings(self):
        settings = f'''Population settings:
    n_parent: {self.n_parent}
    n_parent_per_offspring: {self.n_parent_per_offspring} 
    n_offspring: {self.n_offspring}  
    use_elitism: {self.use_elitism}
    preorder_aware_init: {self.preorder_aware_init}
    cross_over_rate: {self.cross_over_rate}
    exclusive_operations: {self.exclusive_operations}
    random_parent_selection: {self.random_parent_selection}
    replaceable_parent_selection: {self.replaceable_parent_selection}
    selection_strategy: {self.selection_strategy}
    get_parent_strategy: {self.get_parent_strategy}
'''
        return settings

    def __getstate__(self):
        state = self.__dict__.copy()
        state['light_cross_over_evaluator'] = None
        state['light_cross_over_promptor'] = None
        state['selection_strategy'] = None
        state['get_parent_strategy'] = None
        return state

    def get_promptor(self, query_item):
        if query_item.type == ESPopulationQueryItemType.CROSSOVER:
            return self.light_cross_over_promptor
        return None

    def get_evaluator(self, query_item):
        if query_item.type == ESPopulationQueryItemType.CROSSOVER:
            return self.light_cross_over_evaluator
        return None

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
                _candidates = last_gen + last_pop
                candidates = sorted(_candidates, key=lambda x: self.individuals[x].fitness, reverse=True)
            else:
                _candidates = last_gen
                if len(_candidates) < self.n_parent:
                    logging.warning("Population size is less than n_parent. Using elitism.")
                    _candidates = _candidates + last_pop
                candidates = sorted(_candidates, key=lambda x: self.individuals[x].fitness, reverse=True)

            next_pop = candidates[:self.n_parent]
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
            _suffix = f'gen_checkpoint_{n_gen-1}'
            self.save(suffix=_suffix)

    def get_current_generation(self):
        return len(self.selected_generations)


    def _get_query_item(self, comb_set:list, is_randam_selection, pop_list, parent_count) -> PopulationQueryItem:
        selected_index = 0
        if is_randam_selection:
            selected_index = np.random.randint(0, len(comb_set))
        parent_index = comb_set.pop(selected_index)
        parent = [pop_list[i] for i in parent_index]

        _parent_index_set = frozenset(parent_index)
        parent_count[_parent_index_set] = parent_count.get(_parent_index_set, 0) + 1

        query_item = PopulationQueryItem(parent=parent, offspring=Individual())
        return query_item

    def get_next_queryitems(self, query_items):
        _next_query_items = []
        for query_item in query_items:
            if query_item.type != ESPopulationQueryItemType.CROSSOVER:
                continue

            _parent = [query_item.offspring]
            _next_query_item = PopulationQueryItem(parent=_parent, offspring=Individual())
            _next_query_item.type = ESPopulationQueryItemType.MUTATION
            _next_query_items.append(_next_query_item)

        self.logging_mutation_crossover_count(_next_query_items)
        return _next_query_items
    
    def get_offspring_queryitems(self, n_parent:int=None, max_n_offspring:int=None) -> list[PopulationQueryItem]:

        if len(self.selected_generations) == 0:
            # Initial population
            if self.preorder_aware_init:
                parents = [self.individuals[id] for id in self.generations[0]] if len(self.generations) > 0 else []
                query_item = PopulationQueryItem(parent=parents, offspring=Individual())
                query_item.is_initialized = True
                query_item.type = ESPopulationQueryItemType.INIT
                return [query_item]
            else:
                query_items = []
                for i in range(self.n_parent):
                    query_item = PopulationQueryItem(parent=[], offspring=Individual())
                    query_item.is_initialized = True
                    query_item.type = ESPopulationQueryItemType.INIT
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
            
        query_items = []
        parent_count = {}

        _one_comb = list(itertools.combinations(range(len(sorted_last_pop)), 1))
        _n_comb = list(itertools.combinations(range(len(sorted_last_pop)), _n_parent_per_offspring))
        for i in range(_n_offspring):
            _n_parent = _n_parent_per_offspring

            if self.exclusive_operations:
                # exclusive operations
                if _n_parent_per_offspring > 1 and np.random.rand() > self.cross_over_rate and len(_one_comb) > 0:
                    _n_parent = 1

                _current_comb = _one_comb if _n_parent == 1 else _n_comb
                query_item = self._get_query_item(_current_comb, self.random_parent_selection, sorted_last_pop, parent_count)
                query_item.type = ESPopulationQueryItemType.EXCLUSIVE_MUTATION if _n_parent == 1 else ESPopulationQueryItemType.EXCLUSIVE_CROSSOVER
                query_items.append(query_item)
            else:
                _query_item = None
                if np.random.rand() < self.cross_over_rate:
                    _cross_item = self._get_query_item(_n_comb, self.random_parent_selection, sorted_last_pop, parent_count)
                    _cross_item.type = ESPopulationQueryItemType.CROSSOVER
                    _query_item = _cross_item
                if _query_item is None:
                    _mut_item = self._get_query_item(_one_comb, self.random_parent_selection, sorted_last_pop, parent_count)
                    _mut_item.type = ESPopulationQueryItemType.MUTATION
                    _query_item = _mut_item

                query_items.append(_query_item)

            # update _comb
            if len(_one_comb) == 0:
                _one_comb = list(itertools.combinations(range(len(sorted_last_pop)), 1))

            if len(_n_comb) == 0:
                if self.replaceable_parent_selection:
                    # continue to use the same parent
                    _n_comb = list(itertools.combinations(range(len(sorted_last_pop)), _n_parent_per_offspring))
                else:
                    # use bigger combiantions
                    _n_parent_per_offspring += 1
                    if _n_parent_per_offspring > len(sorted_last_pop):
                        # here means all combinations are used, reset
                        _n_parent_per_offspring = self.n_parent_per_offspring
                        _n_comb = list(itertools.combinations(range(len(sorted_last_pop)), _n_parent_per_offspring))
                    else:
                        _n_comb = list(itertools.combinations(range(len(sorted_last_pop)), _n_parent_per_offspring))

        logging.info("Parent count: %s", parent_count)
        self.logging_mutation_crossover_count(query_items)
            
        return query_items

    def logging_mutation_crossover_count(self, query_items):
        if len(query_items) == 0:
            return
        mutation_count = 0
        crossover_count = 0
        for _item in query_items:
            if len(_item.parent) == 1:
                mutation_count += 1
            else:
                crossover_count += 1
        logging.info("Mutation: %d, Crossover: %d", mutation_count, crossover_count)

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

    def get_best_of_all(self, maximize: bool = True):
        best = None
        for ind in self.all_individuals():
            if best is None:
                best = ind
            else:
                if maximize:
                    if ind.fitness > best.fitness:
                        best = ind
                else:
                    if ind.fitness < best.fitness:
                        best = ind
        return best

    def get_best_individual(self, maximize: bool = True):
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