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
    def __init__(self, max_size: int = None):
        self.max_size = max_size
        self.problem = None
        self.model = None
        self.name = None

    @abstractmethod
    def get_population_size(self):
        pass

    @abstractmethod
    def add_individual(self, individual: Individual):
        pass

    @abstractmethod
    def remove_individual(self, individual):
        """
        Removes an individual from the population.

        Args:
            individual (Individual): The individual to remove from the population.
        """

    def select_next_generation(self, selection_strategy: Callable[[Individual,Individual], int] = None, num_individuals: int = 1) -> Individual:
        pass

    @abstractmethod
    def all_individuals(self):
        pass

class SequencePopulation(Population):
    """
    Represents a population of individuals in the evolutionary algorithm.
    """

    def __init__(self, max_size: int = None):
        super().__init__(max_size)
        self.individuals: list[Individual] = []

    def get_population_size(self):
        """
        Returns the number of individuals in the population.

        Returns:
            int: The number of individuals in the population.
        """
        return len(self.individuals)

    def add_individual(self, individual):
        """
        Adds an individual to the population.

        Args:
            individual (Individual): The individual to add to the population.
        """
        super().add_individual(individual)
        if self.max_size and len(self.individuals) >= self.max_size:
            raise ValueError("Population size exceeds the maximum size.")
        self.individuals.append(individual)

    def remove_individual(self, individual):
        """
        Removes an individual from the population.

        Args:
            individual (Individual): The individual to remove from the population.
        """
        self.individuals = [ind for ind in self.individuals if ind.id != individual.id]

    def select_next_generation(self, selection_strategy: Callable[[Individual,Individual], int] = None, num_individuals: int = 1) -> Individual:
        if not self.individuals:
            return None
        sorted_individuals = []
        if selection_strategy is None:
            sorted_individuals = self.individuals
        else:
            sorted_individuals = sorted(self.individuals, cmp = selection_strategy, reverse=False)

        next_generation = []
        if num_individuals > len(sorted_individuals):
            next_generation = sorted_individuals
        else:
            # select last num_individuals from the sorted list
            next_generation = sorted_individuals[-num_individuals:]
        return next_generation[0]

    def all_individuals(self):
        """
        Returns all individuals in the population.

        Returns:
            List[Individual]: A list of all individuals in the population.
        """
        return self.individuals
