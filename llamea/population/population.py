import os
import pickle
import math
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable
import logging
from functools import cmp_to_key
import numpy as np
from ..individual import Individual


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

        self.debug_save_on_the_fly = False
        self.save_dir = None

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

    def get_offsprings(self, generation: int = None):
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

    def _safe_file_name(self, name):
        return name.replace(" ", "").replace(":", "_").replace("/", "_")

    def save_on_the_fly(self, individual: Individual, generation: int):
        if self.debug_save_on_the_fly:
            if self.save_dir is None:
                time_stamp = datetime.now().strftime("%m%d%H%M%S")
                file_name = self._safe_file_name(self.name)
                self.save_dir = f'Experiments/pop_temp/{self.__class__.__name__}_{file_name}_{time_stamp}'
            os.makedirs(self.save_dir, exist_ok=True)

            handler = Population.get_handler_from_individual(individual)
            code = handler.code
            name = handler.code_name
            index = self.get_population_size()
            fitness = individual.fitness
            code_path = f'{self.save_dir}/{generation}-{index}_{name}_{fitness:.4f}.py'
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(code)

            res = handler.eval_result
            res_path = f'{self.save_dir}/{generation}-{index}_{name}.pkl'
            with open(res_path, 'wb') as f:
                pickle.dump(res, f)

            prompt = handler.sys_prompt + '\n\n' + handler.prompt
            prompt_path = f'{self.save_dir}/{generation}-{index}_{name}_prompt.md'
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(prompt)

            raw_res = handler.raw_response + f'\n## Feedback\n {individual.feedback}'
            res_path = f'{self.save_dir}/{generation}-{index}_{name}.md'
            with open(res_path, 'w', encoding='utf-8') as f:
                f.write(raw_res)


    def save(self, filename=None, dirname=None):
        if dirname is None:
            dirname = self.save_dir
        if dirname is None:
            dirname = "Experiments/pop_temp"
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        if filename is None:
            filename = self.name
        filename = self._safe_file_name(filename)
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
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
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
    probability = 1 - probability  # Invert the curve
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
