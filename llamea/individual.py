import os
import json
import pickle
import uuid
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Optional
from collections.abc import Callable


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

class Population(ABC):
    """
    Represents a population of individuals in the evolutionary algorithm.
    """
    def __init__(self):
        self.name = None

    @abstractmethod
    def get_population_size(self):
        pass

    @abstractmethod
    def add_individual(self, individual: Individual):
        pass

    @abstractmethod
    def remove_individual(self, individual):
        pass

    def select_next_generation(self, update_strategy: Callable[[list[Individual], list[Individual], int], list[Individual]] = None):
        pass
        
    @abstractmethod
    def get_parents(self, selection_strategy: Callable[[list[Individual], int, int], list[Individual]] = None) -> list[list[Individual]]:
        return None

    def get_last_successful_parent(self, candidate: Individual) -> Individual:
        return None

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

class ESPopulation(Population):
    def __init__(self,n_parent:int=2, n_parent_per_offspring: int = 1, n_offspring: int = 1, use_elitism: bool = True):
        super().__init__()

        self.n_parent = n_parent
        self.n_parent_per_offspring = n_parent_per_offspring
        self.n_offspring = n_offspring
        self.use_elitism = use_elitism
        self.save_per_generation = 10
        self.save_per_generation_dir = None

        self.individuals:dict[str, Individual] = {}
        # all individuals per generation
        self.generations:list[list[str]] = []
        # selected individuals per generation
        self.selected_generations:list[list[str]] = []

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
        
    def select_next_generation(self, update_strategy: Callable[[list[Individual], list[Individual], int], list[Individual]] = None):
        if len(self.generations) == 0 or len(self.generations[-1]) == 0:
            return
        
        last_gen = self.generations[-1]
        last_pop = []
        if len(self.selected_generations) > 0:
            last_pop = self.selected_generations[-1]

        if update_strategy is None:
            candidates = None
            if self.use_elitism:
                candidates = last_gen + last_pop
            else:
                candidates = last_gen

            next_candidates = sorted(candidates, key=lambda x: self.individuals[x].fitness, reverse=True)
            next_pop = next_candidates[:self.n_parent]
            self.selected_generations.append(next_pop)
        else:
            ind_last_gen = [self.individuals[id] for id in last_gen]
            ind_last_pop = [self.individuals[id] for id in last_pop]
            ind_next_pop = update_strategy(ind_last_gen, ind_last_pop, self.n_parent)
            next_pop = [ind.id for ind in ind_next_pop]
            self.selected_generations.append(next_pop)
        
        n_gen = len(self.selected_generations)
        if n_gen % self.save_per_generation == 0:
            dir_path = self.save_per_generation_dir
            if dir_path is None:
                time_stamp = datetime.now().strftime("%m%d%H%M%S")
                dir_path = f'Experiments/pop_temp/{self.__class__.__name__}_{self.name}_{time_stamp}'
                self.save_per_generation_dir = dir_path
            os.makedirs(dir_path, exist_ok=True)
            file_path = f'{dir_path}/pop_{n_gen}.pkl'
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)

    def get_parents(self, selection_strategy: Callable[[list[Individual], int, int], list[Individual]] = None) -> list[list[Individual|None]]:
        if len(self.selected_generations) == 0:
            return [[]] * self.n_parent

        last_pop = self.selected_generations[-1]
        last_pop = [self.individuals[id] for id in last_pop if id in self.individuals]

        if selection_strategy is not None:
            return selection_strategy(last_pop, self.n_parent, self.n_parent_per_offspring)
            
        n_last_pop_needed = self.n_parent_per_offspring * self.n_offspring
        if len(last_pop) < n_last_pop_needed:
            last_pop = last_pop * (n_last_pop_needed // len(last_pop) + 1)

        parents = []
        idx_last_pop = 0
        for _ in range(self.n_offspring):
            parent = last_pop[idx_last_pop: idx_last_pop+self.n_parent_per_offspring]
            parents.append(parent)
            idx_last_pop += self.n_parent_per_offspring
            
        return parents

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

    def get_parents(self) -> list[list[Individual]]:
        if not self.individuals:
            return [[]]

        return [self.individuals[-1:]]

    def get_last_successful_parent(self, candidate: Individual) -> Individual:
        if candidate is None:
            return None

        # Find the last successful parent of the candidate
        is_before_candidate = True 
        for ind in reversed(self.individuals):
            if is_before_candidate: 
                if ind.id == candidate.id:
                    is_before_candidate = False
            else:
                if ind.error is None:
                    return ind
        return None

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
