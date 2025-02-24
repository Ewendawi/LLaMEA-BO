import os
import time
import pickle
import math
import difflib
import itertools
import concurrent.futures
from abc import ABC, abstractmethod
from datetime import datetime
from collections.abc import Callable
import logging
from functools import cmp_to_key
import numpy as np
from ..individual import Individual


class PopulationQueryItem:
    def __init__(self,qid=None, parent=None, offspring=None):
        if qid is None:
            self.qid = time.time()
        else:
            self.qid = qid
        self.generation = None
        self.is_initialized = False

        self.type = None
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
        self._save_dir_suffix = None
        self.save_per_generation = None

    def get_evaluator(self, query_item: PopulationQueryItem):
        return None

    def get_promptor(self, query_item: PopulationQueryItem):
        return None

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

    def revert_last_generation(self):
        pass

    @abstractmethod
    def get_offspring_queryitems(self, n_parent:int=None, max_n_offspring:int=None) -> list[PopulationQueryItem]:
        return None

    def get_next_queryitems(self, query_items: list[PopulationQueryItem]) -> list[PopulationQueryItem]:
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

    def _update_save_dir_if_need(self):
        if self.save_dir is None:
            self.save_dir = "Experiments/pop_temp"
        if self._save_dir_suffix is None:
            time_stamp = datetime.now().strftime("%m%d%H%M%S")
            file_name = self._safe_file_name(self.name)
            self._save_dir_suffix = f"{self.__class__.__name__}_{file_name}_{time_stamp}"
            

    def save_on_the_fly(self, individual: Individual, generation: int):
        if self.debug_save_on_the_fly:
            self._update_save_dir_if_need()
            _save_dir = self.save_dir + '/' + self._save_dir_suffix
            os.makedirs(_save_dir, exist_ok=True)


            handler = Population.get_handler_from_individual(individual)
            code = handler.code
            name = handler.code_name
            index = self.get_population_size()
            fitness = individual.fitness
            file_name = f'{generation}-{index}_{name}_{fitness:.4f}'

            code_path = f'{_save_dir}/{file_name}.py'
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(code)

            handler_path = f'{_save_dir}/{file_name}_handler.pkl'
            try:
                with open(handler_path, 'wb') as f:
                    pickle.dump(handler, f)
            except Exception as e:
                logging.error("Error saving handler: %s", e)

            prompt = handler.sys_prompt + '\n\n' + handler.prompt
            prompt_path = f'{_save_dir}/{file_name}_prompt.md'
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(prompt)

            raw_res = handler.raw_response
            if handler.feedback:
                raw_res += f'\n## Feedback\n {handler.feedback}'
            elif handler.error:
                raw_res += f'\n## Error\n {handler.error}'
            res_path = f'{_save_dir}/{file_name}_respond.md'
            with open(res_path, 'w', encoding='utf-8') as f:
                f.write(raw_res)

    def save(self, filename=None, dirname=None, suffix=None):
        self._update_save_dir_if_need()
        _save_dir = self.save_dir + '/' + self._save_dir_suffix
        if dirname is None:
            dirname = _save_dir
        os.makedirs(dirname, exist_ok=True)
        if filename is None:
            filename = self.name
        filename = self._safe_file_name(filename)
        if filename in dirname:
            filename = ''
        if suffix is not None:
            filename += f'{suffix}'
        if len(filename) > 0:
            filename = '_' + filename
        time_stamp = datetime.now().strftime("%m%d%H%M%S")
        file_path = os.path.join(dirname, f"{self.__class__.__name__}{filename}_{time_stamp}.pkl")
        try:
            with open(file_path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            logging.error("Error saving population: %s", e)
    
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
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# disable sentence transformer logging
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

def code_diff_similarity(inds:list[Individual], max_workers=0) -> tuple[np.ndarray, np.ndarray]:
    if not inds:
        return np.array([]), np.array([])

    codes = [Population.get_handler_from_individual(ind).code for ind in inds]
    return _code_diff_similarity(codes, max_workers)

def code_diff_similarity_from_handlers(handlers: list, max_workers=0) -> tuple[np.ndarray, np.ndarray]:
    if not handlers:
        return np.array([]), np.array([])

    codes = [handler.code for handler in handlers]
    return _code_diff_similarity(codes, max_workers)

def code_compare(code1, code2):
    diff = difflib.ndiff(code1.splitlines(), code2.splitlines())
    diffs = sum(1 for x in diff if x.startswith("- ") or x.startswith("+ "))
    total_lines = max(len(code1.splitlines()), len(code2.splitlines()))
    similarity_ratio = (total_lines - diffs) / total_lines if total_lines else 1
    return similarity_ratio

def _code_diff_similarity(codes: list, max_workers=0) -> tuple[np.ndarray, np.ndarray]:
    if not codes:
        return np.array([]), np.array([])

    logging.info("Calculating code diversity of %s", len(codes))

    if max_workers > 0:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            params = {}
            for i, code1 in enumerate(codes):
                for j, code2 in enumerate(codes):
                    params[(i,j)] = (code1, code2)

            similarity_matrix = np.zeros((len(codes), len(codes)))

            futures = {executor.submit(code_compare, *params[key]): key for key in params}
            for future in concurrent.futures.as_completed(futures):
                i, j = futures[future]
                similarity_matrix[i,j] = future.result()
                similarity_matrix[j,i] = similarity_matrix[i,j]
    else:
        similarity_matrix = np.zeros((len(codes), len(codes)))
        for i, code1 in enumerate(codes):
            for j, code2 in enumerate(codes):
                similarity_matrix[i,j] = code_compare(code1, code2)
                similarity_matrix[j,i] = similarity_matrix[i,j]

    # Calculate mean similarity excluding diagonal (self-similarity)
    mean_similarity = np.array([
        np.mean(np.concatenate([similarity_matrix[i,:i], similarity_matrix[i,i+1:]]))
        for i in range(len(codes))
    ])

    return mean_similarity, similarity_matrix

def code_bert_similarity(inds:list[Individual]) -> tuple[np.ndarray, np.ndarray]:
    if not inds:
        return np.array([]), np.array([])

    codes = [Population.get_handler_from_individual(ind).code for ind in inds]
    return _code_bert_similarity(codes)

def code_bert_similarity_from_handlers(handlers: list) -> tuple[np.ndarray, np.ndarray]:
    if not handlers:
        return np.array([]), np.array([])

    codes = [handler.code for handler in handlers]
    return _code_bert_similarity(codes)

def _code_bert_similarity(code_list: list[str], 
                        #   model_name="microsoft/codebert-base", 
                          model_name='dbernsohn/roberta-python',
                          batch_size=32, 
                          device=None
) -> tuple[np.ndarray, np.ndarray]:
    logging.info("Calculating code diversity of %s", len(code_list))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    num_batches = (len(code_list) + batch_size - 1) // batch_size
    all_embeddings = []

    for i in range(num_batches):
        batch_codes = code_list[i * batch_size : (i + 1) * batch_size]

        inputs = tokenizer(
            batch_codes, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :] # get CLS embedding.
            all_embeddings.append(embeddings.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)  # Combine batches
    similarity_matrix = cosine_similarity(all_embeddings)

    # Calculate mean similarity excluding diagonal (self-similarity)
    mean_similarity = np.array([
        np.mean(np.concatenate([similarity_matrix[i,:i], similarity_matrix[i,i+1:]]))
        for i in range(len(code_list))
    ])

    return mean_similarity, similarity_matrix

def desc_similarity(inds:list[Individual]) -> tuple[np.ndarray, np.ndarray]:
    if not inds:
        return np.array([]), np.array([])

    descs = [Population.get_handler_from_individual(ind).desc for ind in inds]
    return _desc_similarity(descs)

def desc_similarity_from_handlers(handlers: list) -> tuple[np.ndarray, np.ndarray]:
    if not handlers:
        return np.array([]), np.array([])

    descs = [handler.desc for handler in handlers]
    return _desc_similarity(descs)

def _desc_similarity(descs: list) -> tuple[np.ndarray, np.ndarray]:
    if not descs:
        return np.array([]), np.array([])

    logging.info("Calculating desc diversity of %s", len(descs))
    # Calculate the diversity of the candidates based on the description
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    embeddings = model.encode(descs, convert_to_tensor=False)
    similarity_matrix = util.cos_sim(embeddings, embeddings).numpy()
    # Calculate mean similarity excluding diagonal (self-similarity)
    mean_similarity = np.array([
        np.mean(np.concatenate([similarity_matrix[i,:i], similarity_matrix[i,i+1:]]))
        for i in range(len(descs))
    ])

    return mean_similarity, similarity_matrix

def family_competition_selection_fn(parent_size_threshold=1, is_aggressive=True):
    def _family_selection(child, all_ind_map, parent_size_threshold):
        # Check if the individual is selected in the next generation
        # if child > one of its parents, it will be selected
        # if parent < child, it will not be selected
        child_id = child.id
        if len(child.parent_id) == 0:
            return [child_id], []

        _ind_ids = [child_id] + child.parent_id       
        if len(child.parent_id) > parent_size_threshold:
            return _ind_ids, []
        
        sorted_ids = sorted([ind_id for ind_id in _ind_ids], key=lambda x: all_ind_map[x].fitness, reverse=True)
        child_index = sorted_ids.index(child_id)

        _selected_list = sorted_ids[:child_index+1]
        _unselected_list = sorted_ids[child_index+1:]

        return _selected_list, _unselected_list

    def _family_competition_selection_fn(ind_last_gen, ind_last_pop, n_parent, is_aggressive=is_aggressive, parent_size_threshold=parent_size_threshold):
        # Family competition selection
        # Select n_parent individuals from the last generation and the last population
        # The selected individuals are the best individuals in the family
        # The family is the individual and its parents

        all_ind_map = {ind.id: ind for ind in ind_last_gen + ind_last_pop}

        candidate_set = set()
        substitute_set = set()
        for ind in ind_last_gen:
            _selected_list, _unselected_list = _family_selection(ind, all_ind_map, parent_size_threshold)
            candidate_set.update(_selected_list)
            substitute_set.update(_unselected_list)

        for ind in ind_last_pop:
            if ind.id not in substitute_set:
                candidate_set.add(ind.id)

        intersection = candidate_set.intersection(substitute_set)
        if len(intersection) > 0:
            if is_aggressive:
                # aggressive strategy
                candidate_set = candidate_set - intersection
            else:
                # conservative strategy
                substitute_set = substitute_set - intersection
            
        _candidates = sorted(candidate_set, key=lambda x: all_ind_map[x].fitness, reverse=True)
        _substitutes = sorted(substitute_set, key=lambda x: all_ind_map[x].fitness, reverse=True)
        candidates = _candidates + _substitutes
        ind_candidates = [all_ind_map[ind_id] for ind_id in candidates]
        return ind_candidates[:n_parent]

    return _family_competition_selection_fn

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
    _score_list = []
    _sim_list = []
    sorted_candidates = []
    for ind, sim in tuple_cands:
        sorted_candidates.append(ind)
        _score_list.append(get_score(ind))
        _sim_list.append(sim)

    logging.info("diversity-awareness selection\n%s\n%s", _score_list, _sim_list)

    return sorted_candidates[:n_parent]

def max_divese_desc_get_parent_fn(individuals: list[Individual], n_parent: int, n_offspring: int) -> list[PopulationQueryItem]:
    mean_similarity, similarity_matrix = desc_similarity(individuals)

    # get n_offspring individuals with the lowest similarity
    candidate_idxs = np.argsort(mean_similarity)[:n_offspring].tolist()
    parent_set = set()
    for candidate_idx in candidate_idxs:
        # get n_parent-1 individuals with the lowest similarity to the candidate
        other_sim = similarity_matrix[candidate_idx]
        # remove the candidate itself
        others = np.argsort(other_sim)[:-1].tolist()
        other_comb = itertools.combinations(others, n_parent-1)
        for other in other_comb:
            parent_id = [candidate_idx]
            parent_id.extend(other)
            check_set = frozenset(parent_id)
            if check_set not in parent_set:
                parent_set.add(check_set)
                break

    query_items = []
    query_p_sim = []
    ind_names = [ind.name for ind in individuals]
    for parent_id in parent_set:
        parent = [individuals[i] for i in parent_id]
        query_item = PopulationQueryItem()
        query_item.parent = parent
        query_item.offspring = Individual()
        query_item.qid = 0
        query_items.append(query_item)

        _parent_sim = 0
        for comb in itertools.combinations(parent_id, 2):
            _parent_sim += similarity_matrix[comb[0]][comb[1]]
        _parent_sim /= len(parent_id)
        query_p_sim.append((list(parent_id),_parent_sim))
        
    logging.info("max_divese_desc_get_parent_fn\n%s\n%s\n%s", ind_names, similarity_matrix, query_p_sim)
        
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
