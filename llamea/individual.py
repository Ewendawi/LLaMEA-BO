import os
import json
import pickle
import uuid
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Optional
from collections.abc import Callable
from enum import Enum
from functools import cmp_to_key


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
        self.get_parent_strategy:Callable[[list[Individual], int, int], list[Individual]] = None 
        self.selection_strategy:Callable[[list[Individual], list[Individual], int], list[Individual]] = None

    @abstractmethod
    def get_population_size(self):
        pass

    @abstractmethod
    def add_individual(self, individual: Individual):
        pass

    @abstractmethod
    def remove_individual(self, individual):
        pass

    def select_next_generation(self):
        pass
        
    @abstractmethod
    def get_parents(self) -> list[list[Individual]]:
        return None

    @abstractmethod
    def get_current_generation(self):
        return 0

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

        self.preorder_aware_init = False
        self.get_parent_strategy:Callable[[list[Individual], int, int], list[Individual]] = None 
        self.selection_strategy:Callable[[list[Individual], list[Individual], int], list[Individual]] = None
        
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
        
        # Save population every n generations
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

    def get_current_generation(self):
        return len(self.selected_generations)

    def get_parents(self) -> list[list[Individual|None]]:
        if len(self.selected_generations) == 0:
            if self.preorder_aware_init:
                parents = [self.individuals[id] for id in self.generations[0]] if len(self.generations) > 0 else []
                return [parents]
            else:
                return [[]] * self.n_parent

        last_pop = self.selected_generations[-1]
        last_pop = [self.individuals[id] for id in last_pop if id in self.individuals]

        if self.get_parent_strategy is not None:
            return self.get_parent_strategy(last_pop, self.n_parent_per_offspring, self.n_offspring)
            
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

class IslandESPopulation(Population):

    class IslandStatus(Enum):
        NORMAL = 0
        KILLED = 1
        RESETING = 2
        REQUIRE_MIGRANT = 3
    
    class IslandOperation(Enum):
        MIGRATION = 0
        KILL = 1
        RESET = 2
    
    
    def __init__(self, n_islands: int = 1, n_parent: int = 1, n_offspring: int = 1, n_parent_per_offspring: int = 1, use_elitism: bool = True):
        super().__init__()

        self.n_islands = n_islands
        self.n_parent = n_parent
        self.n_offspring = n_offspring
        self.n_parent_per_offspring = n_parent_per_offspring
        self.use_elitism = use_elitism

        self.populations = [ESPopulation(n_parent, n_parent_per_offspring, n_offspring, use_elitism) for _ in range(n_islands)]

        self.pop_status = [self.IslandStatus.NORMAL] * n_islands

    def __set_island_index(self, individual: Individual, island_index: int):
        individual.add_metadata("island_index", island_index)

    def __get_island_index(self, individual: Individual):
        index = individual.get_metadata("island_index")
        if index is None and individual.parent_id is not None:
            parent_ids = individual.parent_id
            if isinstance(individual.parent_id, str):
                parent_ids = [individual.parent_id] 
            for parent_id in parent_ids:
                parent = self.__get_individual_with_id(parent_id)
                if parent is not None:
                    index = self.__get_island_index(parent)
                    break
        return index

    def __get_individual_with_id(self, individual_id: str):
        for pop in self.populations:
            if individual_id in pop.individuals:
                return pop.individuals[individual_id]
        return None

    def get_population_size(self):
        return sum([pop.get_population_size() for pop in self.populations])

    def add_individual(self, individual: Individual):
        island_index = self.__get_island_index(individual)
        if island_index is not None:
            self.populations[island_index].add_individual(individual)
            self.__set_island_index(individual, island_index)
    
    def remove_individual(self, individual):
        island_index = self.__get_island_index(individual)
        if island_index is not None:
            self.populations[island_index].remove_individual(individual)
            
    def select_next_generation(self):
        for pop in self.populations:
            pop.select_next_generation()
            
    def get_parents(self) -> list[list[Individual]]:
        parents = []
        for pop in self.populations:
            parents.extend(pop.get_parents())
        return parents
    
    def get_current_generation(self):
        return self.populations[0].get_current_generation()

    def get_best_individual(self, maximize: bool = False):
        best = None
        for pop in self.populations:
            ind = pop.get_best_individual(maximize)
            if best is None or (ind is not None and ind.fitness > best.fitness):
                best = ind
        return best

    def all_individuals(self):
        all_inds = []
        for pop in self.populations:
            all_inds.extend(pop.all_individuals())
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



# Utility functions
from sentence_transformers import SentenceTransformer, util
import numpy as np

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
    
    def get_desc(ind):
        return Population.get_handler_from_individual(ind).desc

    # Calculate the diversity of the candidates based on the description
    descs = [get_desc(ind) for ind in candidates]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(descs, convert_to_tensor=False)
    similarity_matrix = util.cos_sim(embeddings, embeddings).numpy()
    # Calculate mean similarity excluding diagonal (self-similarity)
    mean_similarity = np.array([
        np.mean(np.concatenate([similarity_matrix[i,:i], similarity_matrix[i,i+1:]]))
        for i in range(len(candidates))
    ])

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

def max_divese_desc_get_parent_fn(individuals: list[Individual], n_parent: int, n_offspring: int) -> list[Individual]:
    descs = [Population.get_handler_from_individual(ind).desc for ind in individuals]
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(descs, convert_to_tensor=False)
    similarity_matrix = util.cos_sim(embeddings, embeddings).numpy()

    mean_similarity = np.array([
        np.mean(np.concatenate([similarity_matrix[i,:i], similarity_matrix[i,i+1:]]))
        for i in range(len(descs))
    ])
    # get n_offspring individuals with the lowest similarity
    candidate_idxs = np.argsort(mean_similarity)[:n_offspring].tolist()
    parent_ids = []
    for i in candidate_idxs:
        parent_id = [i]
        others = np.argsort(similarity_matrix[i])[:n_parent-1].tolist()
        for j in others:
            parent_id.append(j)
        parent_ids.append(parent_id)
    parents = []
    for parent_id in parent_ids:
        parent = [individuals[i] for i in parent_id]
        parents.append(parent)
    return parents

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
