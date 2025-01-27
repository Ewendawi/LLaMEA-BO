import os
import json
import pickle
import uuid
import logging
import math
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Optional
from collections.abc import Callable
from enum import Enum
from functools import cmp_to_key
import numpy as np


class Individual:
    """
    Represents a candidate solution (an individual) in the evolutionary algorithm.
    Each individual has properties such as solution code, fitness, feedback, and metadata for additional information.
    """

    def __init__(
        self,
        solution="",
        name="",
        description="",
        configspace=None,
        generation=0,
        parent_id=None,
    ):
        """
        Initializes an individual with optional attributes.

        Args:
            solution (str): The solution (code) of the individual.
            name (str): The name of the individual (typically the class name in the solution).
            description (str): A short description of the individual (e.g., algorithm's purpose or behavior).
            configspace (Optional[ConfigSpace]): Optional configuration space for HPO.
            generation (int): The generation this individual belongs to.
            parent_id (str): UUID of the parent individual.
        """
        self.id = str(uuid.uuid4())  # Unique ID for this individual
        self.solution = solution
        self.name = name
        self.description = description
        self.configspace = configspace
        self.generation = generation
        self.fitness = None
        self.feedback = ""
        self.error = ""
        self.parent_id = parent_id
        self.metadata = {}  # Dictionary to store additional metadata
        self.mutation_prompt = None

    def set_mutation_prompt(self, mutation_prompt):
        """
        Sets the mutation prompt of this individual.

        Args:
            mutation_prompt (str): The mutation instruction to apply to this individual.
        """
        self.mutation_prompt = mutation_prompt

    def add_metadata(self, key, value):
        """
        Adds key-value pairs to the metadata dictionary.

        Args:
            key (str): The key for the metadata.
            value (Any): The value associated with the key.
        """
        self.metadata[key] = value

    def get_metadata(self, key):
        """
        Get a metadata item from the dictionary.

        Args:
            key (str): The key for the metadata to obtain.
        """
        return self.metadata[key] if key in self.metadata.keys() else None

    def set_scores(self, fitness, feedback="", error=""):
        self.fitness = fitness
        self.feedback = feedback
        self.error = error

    def get_summary(self):
        """
        Returns a string summary of this individual's key attributes.

        Returns:
            str: A string representing the individual in a summary format.
        """
        return f"{self.name}: {self.description} (Score: {self.fitness})"

    def copy(self):
        """
        Returns a copy of this individual, with a new unique ID and a reference to the current individual as its parent.

        Returns:
            Individual: A new instance of Individual with the same attributes but a different ID.
        """
        new_individual = Individual(
            solution=self.solution,
            name=self.name,
            description=self.description,
            configspace=self.configspace,
            generation=self.generation + 1,
            parent_id=self.id,  # Link this individual as the parent
        )
        new_individual.metadata = self.metadata.copy()  # Copy the metadata as well
        return new_individual

    @classmethod
    def from_dict(cls, data):
        """
        Creates an individual from a dictionary.

        Args:
            data (dict): A dictionary containing the individual's attributes.

        Returns:
            Individual: An instance of Individual created from the dictionary.
        """
        individual = cls()
        individual.id = data["id"]
        individual.solution = data["solution"]
        individual.name = data["name"]
        individual.description = data["description"]
        individual.generation = data["generation"]
        individual.fitness = data["fitness"]
        individual.feedback = data["feedback"]
        individual.error = data["error"]
        individual.parent_id = data["parent_id"]
        individual.metadata = data["metadata"]
        individual.mutation_prompt = data["mutation_prompt"]
        return individual

    def to_dict(self):
        """
        Converts the individual to a dictionary.

        Returns:
            dict: A dictionary representation of the individual.
        """
        try:
            cs = self.configspace
            cs = cs.to_serialized_dict()
        except Exception:
            cs = ""
        return {
            "id": self.id,
            "solution": self.solution,
            "name": self.name,
            "description": self.description,
            "configspace": cs,
            "generation": self.generation,
            "fitness": self.fitness,
            "feedback": self.feedback,
            "error": self.error,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
            "mutation_prompt": self.mutation_prompt,
        }

    # JSON serialization methods. Used by a custom JSON encoder in utils.py.
    def __to_json__(self):
        return self.to_dict()

    def to_json(self):
        """
        Converts the individual to a JSON string.

        Returns:
            str: A JSON string representation of the individual.
        """
        return json.dumps(self.to_dict(), default=str, indent=4)
        

class PopulationQueryItem:
    def __init__(self, qid=None, parent=None, offspring=None):
        self.qid = qid
        self.is_initialized = False
        self.parent: list[Individual] = parent
        self.offspring: Individual = offspring

class Population(ABC):
    """
    Represents a population of individuals in the evolutionary algorithm.
    """
    def __init__(self):
        self.name = None
        self.get_parent_strategy:Callable[[list[Individual], int, int], list[PopulationQueryItem]] = None 
        self.selection_strategy:Callable[[list[Individual], list[Individual], int], list[Individual]] = None

    @abstractmethod
    def get_population_size(self):
        pass

    @abstractmethod
    def add_individual(self, individual: Individual, generation: int):
        pass

    @abstractmethod
    def remove_individual(self, individual):
        pass

    def select_next_generation(self):
        pass

    @abstractmethod
    def get_offspring_queryitems(self, n_parent:int=None) -> list[PopulationQueryItem]:
        return None

    @abstractmethod
    def get_current_generation(self):
        return 0

    def get_individuals(self, generation: int = None):
        return []

    @abstractmethod
    def get_best_individual(self, maximize: bool = False):
        pass

    @abstractmethod
    def all_individuals(self):
        pass

    @classmethod
    def set_handler_to_individual(cls, individual: Individual, handler):
        if individual is not None:
            individual.metadata["res_handler"] = handler

    @classmethod
    def get_handler_from_individual(cls, individual: Individual):
        if individual is not None:
            return individual.metadata["res_handler"] if "res_handler" in individual.metadata else None
        return None

    def save(self, filename=None, dirname=None):
        if dirname is None:
            dirname = "population_logs"
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        if filename is None:
            filename = self.name
        filename = filename.replace(" ", "")
        filename = filename.replace(":", "_")
        filename = filename.replace("/", "_")
        time_stamp = datetime.now().strftime("%m%d%H%M%S")
        filename = os.path.join(dirname, f"{self.__class__.__name__}_{filename}_{time_stamp}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath=None):
        if filepath is None and not os.path.exists(filepath):
            return None
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                pop = pickle.load(f)
            return pop
        else:
            raise FileNotFoundError(f"File {filepath} not found")
        return None

class ESPopulation(Population):
    def __init__(self,n_parent:int=2, n_parent_per_offspring: int = 1, n_offspring: int = 1, use_elitism: bool = True):
        super().__init__()

        self.preorder_aware_init = False
        self.cross_over_rate = 0.6
        
        self.n_parent = n_parent
        self.n_parent_per_offspring = n_parent_per_offspring
        self.n_offspring = n_offspring
        self.use_elitism = use_elitism
        self.save_per_generation = 10
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
                Population.get_handler_from_individual(ind).eval_result.simiarity = mean_similarity[i]
        
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
            gen = len(self.selected_generations) - 1
        
        ind_ids = self.selected_generations[gen]
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
        for wrapper in self.populations:
            if wrapper.status != self.IslandStatus.KILLED:
                inds.extend(wrapper.pop.get_individuals(generation))
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



# Utility functions
from sentence_transformers import SentenceTransformer, util

# disable sentence transformer logging
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

def desc_similarity(inds:list[Individual]) -> tuple[np.ndarray, np.ndarray]:
    if not inds:
        return np.array([]), np.array([])

    logging.info("Calculating desc diversity of %s individuals", len(inds))
    # Calculate the diversity of the candidates based on the description
    descs = [Population.get_handler_from_individual(ind).desc for ind in inds]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(descs, convert_to_tensor=False)
    similarity_matrix = util.cos_sim(embeddings, embeddings).numpy()
    # Calculate mean similarity excluding diagonal (self-similarity)
    mean_similarity = np.array([
        np.mean(np.concatenate([similarity_matrix[i,:i], similarity_matrix[i,i+1:]]))
        for i in range(len(inds))
    ])

    return mean_similarity, similarity_matrix

def diversity_awarness_selection_fn(next_inds: list[Individual], last_inds: list[Individual], n_parent: int, fitness_threshold: float = None) -> list[Individual]:
    candidates = []
    if last_inds:
        candidates += last_inds
    if next_inds:
        candidates += next_inds
    if len(candidates) <= n_parent:
        return candidates

    def get_score(ind):
        return Population.get_handler_from_individual(ind).eval_result.score
    
    # Calculate the similarity of the candidates based on the description
    mean_similarity, _ = desc_similarity(candidates)

    if fitness_threshold is None:
        mean_score = np.mean([get_score(ind) for ind in candidates])
        mae_score = np.mean([abs(get_score(ind) - mean_score) for ind in candidates])
        fitness_threshold = mae_score

    def cmp_inds(a, b):
        ind1, sim1 = a
        ind2, sim2 = b
        score1, score2 = get_score(ind1), get_score(ind2)
        if abs(score1 - score2) > fitness_threshold:
            return score1 - score2
        else:
            return -(sim1 - sim2)

    tuple_cands = sorted(list(zip(candidates, mean_similarity)), key=cmp_to_key(cmp_inds), reverse=True)
    sorted_candidates = [ind for ind, _ in tuple_cands]
    return sorted_candidates[:n_parent]

def max_divese_desc_get_parent_fn(individuals: list[Individual], n_parent: int, n_offspring: int) -> list[PopulationQueryItem]:
    mean_similarity, similarity_matrix = desc_similarity(individuals)
    # get n_offspring individuals with the lowest similarity
    candidate_idxs = np.argsort(mean_similarity)[:n_offspring].tolist()
    parent_ids = []
    for i in candidate_idxs:
        parent_id = [i]
        others = np.argsort(similarity_matrix[i])[:n_parent-1].tolist()
        for j in others:
            parent_id.append(j)
        parent_ids.append(parent_id)
    query_items = []
    for parent_id in parent_ids:
        parent = [individuals[i] for i in parent_id]
        query_item = PopulationQueryItem()
        query_item.parent = parent
        query_item.offspring = Individual()
        query_item.qid = 0
        query_items.append(query_item)
    return query_items

def score_to_probability_with_logarithmic_curve(
    score: float,
    min_score: float = 0.0,
    max_score: float = 1.0,
    curve_factor: float = 1.0
) -> float:
    """
    Convert score to probability using tunable logarithmic curve
    Args:
        score: Input score value
        min_score: Minimum possible score
        max_score: Maximum possible score
        curve_factor: Controls curve shape (>1 steeper, <1 flatter)
    Returns:
        probability: float between 0.0 and 1.0
    """
    if not all(isinstance(x, (int, float)) for x in [score, min_score, max_score, curve_factor]):
        logging.error("All inputs must be numbers")
        return 0.0
        
    if min_score >= max_score:
        logging.error("min_score must be less than max_score")
        return 0.0
        
    if score < min_score or score > max_score:
        logging.error("Score must be between %s and %s", min_score, max_score)
        return 0.0
        
    if curve_factor <= 0:
        logging.error("curve_factor must be positive")
        return 0.0

    # Normalize score to 0-1 range
    normalized_score = (score - min_score) / (max_score - min_score)
    
    # Apply tunable log transformation
    epsilon = 1e-10
    probability = math.log(normalized_score * curve_factor + epsilon + 1) / math.log(curve_factor + 1)
    
    return min(max(probability, 0.0), 1.0)  # Ensure output is between 0 and 1


def score_to_probability_with_sigmoid(
    score: float,
    min_score: float = 0.0,
    max_score: float = 1.0,
    steepness: float = 12.0,
    center: float = 6.0
) -> float:
    """
    Convert a score to probability using configurable sigmoid function
    Args:
        score: Input score value
        min_score: Minimum possible score
        max_score: Maximum possible score
        steepness: Controls how steep the S-curve is (default 12.0)
        center: Controls the midpoint shift (default 6.0)
    """
    if not all(isinstance(x, (int, float)) for x in [score, min_score, max_score]):
        logging.error("All inputs must be numbers")
        return 0.0
        
    if min_score >= max_score:
        logging.error("min_score must be less than max_score")
        return 0.0
        
    if score < min_score or score > max_score:
        logging.error("Score must be between %s and %s", min_score, max_score)
        return 0.0

    normalized_score = (score - min_score) / (max_score - min_score)
    x = steepness * normalized_score - center
    probability = 1 / (1 + math.exp(-x))
    
    return probability

def test_diversity_awarness_selection_fn():
    descs = [
        "A Bayesian Optimization algorithm that uses a Gaussian Process surrogate model with Expected Improvement acquisition function and a batch-sequential optimization strategy with quasi-Monte Carlo sampling for initial points and a local search around the best point.",
        "A Bayesian Optimization algorithm using a Tree-structured Parzen Estimator (TPE) surrogate model with a multi-point acquisition strategy, employing random sampling for initialization and a trust-region-like exploration around promising regions."
    ]
    scores = [0.6, 0.7]
    descs2 = [
        "A Bayesian Optimization algorithm that combines Thompson sampling and Expected Improvement with an ensemble of GPs, adaptive batch sizes, trust region based local search, and dynamic adjustment of exploration/exploitation balance, and incorporates a warm restart strategy.",
        "A Bayesian optimization algorithm that adaptively balances exploration and exploitation using a dynamic trust region with a mixed batch strategy, local search, adaptive kernel length scale and focuses on exploration when progress stagnates."
    ] 
    scores2 = [0.8, 0.6]

    class test_res:
        def __init__(self):
            self.score = 0.0
    class Res_handler:
        def __init__(self):
            self.desc = ""
            self.eval_result = test_res()

    last_inds = []
    next_inds = []
    for i in range(2):
        last = Individual()
        res = Res_handler()
        res.desc = descs[i]
        res.eval_result.score = scores[i]
        Population.set_handler_to_individual(last, res)
        last_inds.append(last)

        next = Individual()
        res = Res_handler()
        res.desc = descs2[i]
        res.eval_result.score = scores2[i]
        Population.set_handler_to_individual(next, res)
        next_inds.append(next)
        
    selected = diversity_awarness_selection_fn(last_inds=last_inds, next_inds=next_inds, n_parent=2)
    assert len(selected) == 2

def test_max_diverse_parent_fn():
    descs = [
        "A Bayesian Optimization algorithm that uses a Gaussian Process surrogate model with Expected Improvement acquisition function and a batch-sequential optimization strategy with quasi-Monte Carlo sampling for initial points and a local search around the best point.",
        "A Bayesian Optimization algorithm using a Gaussian Process surrogate model with Expected Improvement, employing a batch-sequential strategy with Sobol sequence for initial sampling and a gradient-based local search around the best point for exploitation.",
        "A Bayesian Optimization algorithm using a Gaussian Process surrogate model with Expected Improvement, employing a batch-sequential strategy with Latin Hypercube Sampling for initial sampling and a CMA-ES local search around the best point for exploitation.",
        "A Bayesian Optimization algorithm using a Tree-structured Parzen Estimator (TPE) surrogate model with a multi-point acquisition strategy, employing random sampling for initialization and a trust-region-like exploration around promising regions."
    ]
    inds = []
    for desc in descs:
        ind = Individual()

        class Res_handler:
            def __init__(self):
                self.desc = ""

        res = Res_handler()
        res.desc = desc
        Population.set_handler_to_individual(ind, res)
        inds.append(ind)
    parents = max_divese_desc_get_parent_fn(inds, 2, 1)
    assert len(parents) == 2
