import random
import traceback
import re
import sys
import io
import json
import uuid
from typing import Callable, Any
from datetime import datetime
import logging
import contextlib
import threading
import pickle
import os
import itertools
import time
from abc import ABC, abstractmethod
import inspect
from tqdm import tqdm
import numpy as np
import torch
from botorch.test_functions import synthetic
from botorch.test_functions.synthetic import SyntheticTestFunction, ConstrainedSyntheticTestFunction
from .individual import Individual


class NoCodeException(Exception):
    """Could not extract generated code."""

    pass


def handle_timeout(signum, frame):
    raise TimeoutError

#========================================
#Logger
#========================================
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: grey + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(logger = None, level=logging.INFO, filename=None):
    if logger is None:
        logger = logging.getLogger()
    logger.setLevel(level)
    if filename:
        fh = logging.FileHandler(filename)
        fh.setLevel(level)
        logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    return logger

def get_logger(name = None, level=logging.INFO, filename=None):
    logger = logging.getLogger(name)
    setup_logger(logger, level, filename)
    return logger


class IndividualLogger:
    def __init__(self):
        self.individualMap:dict[str, Individual] = {}
        self.experimentMap:dict[str, dict] = {}
        self.log_extract_error = False
        self.should_log_population = True
        self.should_log_experiment = True
        self.auto_save = False
        self._file_name = "individual_set"
        self.dirname = "logs"

    @property
    def file_name(self):
        return self._file_name
    
    @file_name.setter
    def file_name(self, value):
        new_value = value
        if value is not None:
            new_value = value.replace(" ", "")
            new_value = new_value.replace(":", "_")
            new_value = new_value.replace("/", "_")
        self._file_name = new_value 

    def log_individual(self, individual):
        self.individualMap[individual.id] = individual

    def get_individual(self, ind_id):
        return self.individualMap.get(ind_id, None)

    def log_experiment(self, name, id_list):
        exp_id = str(uuid.uuid4())
        experiment = {
            "id": exp_id,
            "name": name,
            "id_list": id_list
        }
        self.experimentMap[exp_id] = experiment

    def get_experiment(self, experiment_id):
        return self.experimentMap.get(experiment_id, None)

    def save(self, filename=None, dirname=None):
        if dirname is None:
            dirname = self.dirname
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        if filename is None:
            filename = self.file_name
        filename = filename.replace(" ", "")
        filename = filename.replace(":", "_")
        filename = filename.replace("/", "_")
        time_stamp = datetime.now().strftime("%m%d%H%M%S")
        filename = os.path.join(dirname, f"{filename}_{time_stamp}.pkl")
        with open(filename, "wb") as f:
            pickle.dump((self.individualMap,self.experimentMap), f)
    
    def get_successful_individuals(self):
        successful_individuals = []
        for _, individual in self.individualMap.items():
            if isinstance(individual, dict):
                # No longer compatible with older formats
                continue
            if individual.error is None and "deprecated" not in individual.metadata:
                successful_individuals.append(individual)
        return successful_individuals

    def get_failed_individuals(self, error_type=None):
        failed_individuals = []
        for _, individual in self.individualMap.items():
            if isinstance(individual, dict):
                # No longer compatible with older formats
                continue
            if individual.error is None or "deprecated" in individual.metadata:
                continue
            if (error_type is None or individual.metadata["error_type"] == error_type):
                failed_individuals.append(individual)
        return failed_individuals

# {
#     "contents": {
#         "<id>": {
#             "id": "",
#             "solution": "", // code block
#             "name": "",
#             "description": "", // desc block, markdown
#             "fitness": "",
#             "feedback": "", // feedback and error block, markdown
#             "error": "",
#             "parent_id": "",
#             "metadata": {
#                 "error_type": "", // single-choice filter
#                 "model": "", // single-choice filter
#                 "prompt": "", // prompt block, foldable
#                 "raw_response": "", // response block, markdown
#                 "problem": "", // single-choice filter
#                 "tags": [] // multiple-choice filter
#             }
#         }
#     },
#     "experiments": {
#         "<experiment_id>": {
#             "id": "", // single-choice filter. retrieve all the content in the id_list. 
#             "name": "",
#             "id_list": [] // id: content_id
#         }
#     }
# }

    def save_reader_format(self, filename=None):
        json_str = self.covert_to_reader_format()
        time_stamp = datetime.now().strftime("%m%d%H%M%S")
        if filename is None:
            filename = self.file_name
        filename = filename.replace(" ", "")
        filename = filename.replace(":", "_")
        filename = filename.replace("/", "_")
        filepath = os.path.join(self.dirname, f"reader_format_{filename}_{time_stamp}.json")
        with open(f"{filepath}", "w") as f:
            f.write(json_str)

    def covert_to_reader_format(self) -> str:
        reader_format = {
            "experiments": self.experimentMap.copy()
        }
        contents = {}
        for ind_id, individual in self.individualMap.items():
            ind_dict = individual
            if not isinstance(ind_dict, dict):
                ind_dict = individual.to_dict()
            contents[ind_id] = ind_dict

        reader_format["contents"] = contents
            
        for _, individual in reader_format["contents"].items():
            individual["language"]= "python"

        json_str = json.dumps(reader_format, indent=4)
        return json_str

    @classmethod
    def load(cls, filepath=None):

        if filepath is None and not os.path.exists(filepath):
            return None
        logger = cls()
        logger.__load(filepath)
        return logger

    def __load(self, filepath):
        if filepath is None:
            return
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                # self.individualMap = pickle.load(f)
                self.individualMap, self.experimentMap = pickle.load(f)

    @classmethod
    def merge_logs(cls, log_dir, save=True):
        # check if the log_dir is a directory
        if not os.path.isdir(log_dir):
            return None

        log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".pkl")]
        loggers = []
        for log_file in log_files:
            logger = cls()
            logger.__load(log_file)
            loggers.append(logger)
        merged_logger = cls()
        for logger in loggers:
            merged_logger.individualMap.update(logger.individualMap)
            merged_logger.experimentMap.update(logger.experimentMap)
        if save:
            merged_logger.save()
        return merged_logger
#========================================
#BoTorch test functions
#========================================

def get_all_synthetic_test_function_from_botorch() -> list[SyntheticTestFunction]:
    test_functions = {}
    for module_name in [synthetic]: 
      for name, obj in inspect.getmembers(module_name):
        if name == "Cosine8":
            # Cosine8 is a maximation problem
            continue
        if inspect.isclass(obj) and issubclass(obj, SyntheticTestFunction) and not issubclass(obj, ConstrainedSyntheticTestFunction) and obj != SyntheticTestFunction and obj != ConstrainedSyntheticTestFunction:
            test_functions[name] = obj

    return test_functions

botorch_test_functions = []

def get_test_function_by_name(name: str = None) -> SyntheticTestFunction:
    global botorch_test_functions
    if len(botorch_test_functions) == 0:
        botorch_test_functions = get_all_synthetic_test_function_from_botorch()

    if name is None:
        return random.choice(list(botorch_test_functions.values()))
    if name not in botorch_test_functions:
        return None
    return botorch_test_functions[name]

#========================================
#Random search
#========================================

def random_search(objective_fn, bounds, budgets):
    """Random search."""
    X = np.random.uniform(bounds[0], bounds[1], size=(budgets, len(bounds[0])))
    Y = np.array([objective_fn(x) for x in X])
    
    return Y, X
    
    # best_value = float("inf")
    # best_params = None
    # for _ in range(budgets):
    #     x = [random.uniform(bounds[0][i], bounds[1][i]) for i in range(len(bounds[0]))]
    #     y = objective_fn(x)
    #     if y < best_value:
    #         best_value = y
    #         best_params = x
    # if isinstance(best_params, np.ndarray):
    #     if len(best_params.shape) == 1:
    #         best_value = best_value.item()
    #     else:
    #         best_params = best_params.tolist()
    # return best_value, best_params

#========================================
#Track exec
#========================================


def track_exec(code_string, name, _globals=None, _locals=None):
    compiled_code = compile(code_string, f'<{name}>', 'exec')
    exec(compiled_code, _globals, _locals)

def format_track_exec_with_code(name, code_str, exc_info, context_lines=2):
    trace_lines = traceback.format_exception(*exc_info)
    formatted_trace = ['']

    last_match_index = 0
    for i, line in enumerate(reversed(trace_lines)):
        match = re.search(rf'File "<{name}>", line (\d+), in', line)
        if match:
            last_match_index = len(trace_lines) - i - 1
            break
    
    for i, line in enumerate(trace_lines):
        formatted_trace.append(line)
        match = re.search(rf'File "<{name}>", line (\d+), in', line)
        if match:
            error_line = int(match.group(1))
            
            _context_lines = 0
            if i == last_match_index:
                _context_lines = context_lines
                
            formatted_trace.extend(get_code_snippet(code_str, error_line, _context_lines))
    
    return "".join(formatted_trace)


def get_code_snippet(code_str, error_line, context_lines):
    """Extracts code snippet around a specific line."""
    lines = code_str.splitlines()
    # return lines[error_line - 1] + '\n'
    start_line = max(0, error_line - context_lines - 1)
    end_line = min(len(lines), error_line + context_lines)

    formatted_code = []
    for i, line in enumerate(lines[start_line:end_line], start=start_line + 1):
        if i == error_line:
            formatted_code.append(f"{i:4}-> {line}\n")
        else:
            formatted_code.append(f"{i:4} | {line}\n")
    return formatted_code

def default_exec(code, cls_name, objective_fn, bounds, budget) -> tuple[any, str, str]:
    captured_output = io.StringIO()
    res = None
    err = None
    try:
        namespace: dict[str, Any] = {}
        track_exec(code, cls_name, namespace)
        
        if cls_name not in namespace:
            err = NameError(f"No '{cls_name}' found in the generated code")
        else:
            bo_cls = namespace[cls_name]
            bo = bo_cls()
            with contextlib.redirect_stderr(captured_output), contextlib.redirect_stdout(captured_output):
                res = bo.optimize(objective_fn=objective_fn, bounds=bounds, budget=budget)
    except Exception as e:
        formatted_traceback = format_track_exec_with_code(cls_name, code, sys.exc_info())
        err = e.__class__(formatted_traceback)

    return res, captured_output.getvalue(), err
            

#========================================
#Evaluator
#========================================

class ConvergenceCurveAnalyzer:
    """Analyzes optimization convergence curves and calculates AOC metric."""
    
    def __init__(self, max_y=None, min_y=None):
        self.max_y = max_y
        self.min_y = min_y
        
    def get_convergence_curve(self, y_history):
        """Calculate minimum values seen so far at each step."""
        if not isinstance(y_history, np.ndarray):
            y_history = np.array(y_history)
        return np.minimum.accumulate(y_history)

    def calculate_aoc(self, y_history):
        """Calculate area over convergence curve."""
        if len(y_history) == 0:
            return 0.0
            
        # Get convergence curve
        conv_curve = self.get_convergence_curve(y_history)
        
        local_max_y = np.max(y_history)
        max_y = self.max_y if self.max_y is not None else local_max_y

        local_min_y = np.min(y_history)
        min_y = self.min_y if self.min_y is not None else local_min_y
        
        # Normalize curve between 0 and 1
        norm_curve = (conv_curve - min_y) / (max_y - min_y)
        
        # Calculate AOC using trapezoidal rule
        x_vals = np.linspace(0, 1, len(norm_curve))
        aoc = np.trapz(1 - norm_curve, x_vals)
        
        return aoc

# y = np.array([100, 9, 8, 7, 6, 5, 5, 5, 5, 5])
# aoc = ConvergenceCurveAnalyzer(min_y=0).calculate_aoc(y)
# print(aoc)

class EvaluatorBasicResult:
    def __init__(self):
        self.name = None
        self.execution_time = 0
        self.y_hist:np.ndarray = None
        self.x_hist:np.ndarray = None

        self._surragate_model_losses:np.ndarray = None
        self.model_loss_name:str = None

        #TODO: add acquisition function values
        self.acquisition_function_values:np.ndarray = None 

        self.best_y = None
        self.best_x = None

        self.y_aoc = 0.0
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

        self.n_initial_points = 0

        # (initial, non-initial)
        self.x_mean_tuple = None  
        self.x_std_tuple = None
        self.y_mean_tuple = None
        self.y_std_tuple = None
        self.y_best_tuple = None
        self.non_init_y_aoc = 0.0

    @property
    def surragate_model_losses(self):
        return self._surragate_model_losses
    
    @surragate_model_losses.setter
    def surragate_model_losses(self, value):
        if isinstance(value, np.ndarray) and not value.dtype.hasobject:
            self._surragate_model_losses = value

    def to_dict(self):
        d = {}
        d["name"] = self.name
        d["execution_time"] = self.execution_time
        d["y_hist"] = self.y_hist.tolist() if self.y_hist is not None else None
        d["x_hist"] = self.x_hist.tolist() if self.x_hist is not None else None

        d["surragate_model_losses"] = self.surragate_model_losses.tolist() if self.surragate_model_losses is not None else None
        d["model_loss_name"] = self.model_loss_name

        d["best_y"] = self.best_y
        d["best_x"] = self.best_x.tolist() if self.best_x is not None else None

        d["y_aoc"] = self.y_aoc
        d["x_mean"] = self.x_mean.tolist() if self.x_mean is not None else None
        d["x_std"] = self.x_std.tolist() if self.x_std is not None else None
        d["y_mean"] = self.y_mean
        d["y_std"] = self.y_std

        d["n_initial_points"] = self.n_initial_points

        d["x_mean_tuple"] = (self.x_mean_tuple[0].tolist(), self.x_mean_tuple[1].tolist()) if self.x_mean_tuple is not None else None
        d["x_std_tuple"] = (self.x_std_tuple[0].tolist(), self.x_std_tuple[1].tolist()) if self.x_std_tuple is not None else None
        d["y_mean_tuple"] = self.y_mean_tuple
        d["y_std_tuple"] = self.y_std_tuple
        
        d["acquisition_function_values"] = np.nan_to_num(self.acquisition_function_values).tolist() if self.acquisition_function_values is not None else None
        return d

    def update_stats(self):
        if self.y_hist is None or self.x_hist is None:
            return

        best_index = np.argmin(self.y_hist)
        self.best_y = self.y_hist[best_index]
        self.best_x = self.x_hist[best_index]
        
        y_hist = self.y_hist
        self.y_mean = np.mean(y_hist)
        self.y_std = np.std(y_hist)
        
        if self.n_initial_points > 0 and len(y_hist) > self.n_initial_points:
            y_hist = self.y_hist[self.n_initial_points:]
            self.y_mean_tuple = (np.mean(self.y_hist[:self.n_initial_points]), np.mean(y_hist))
            self.y_std_tuple = (np.std(self.y_hist[:self.n_initial_points]), np.std(y_hist))
            self.y_best_tuple = (np.min(self.y_hist[:self.n_initial_points]), np.min(y_hist))
        
        x_hist = self.x_hist
        self.x_mean = np.mean(x_hist, axis=0)
        self.x_std = np.std(x_hist, axis=0)
        
        if self.n_initial_points > 0 and len(x_hist) > self.n_initial_points:
            x_hist = self.x_hist[self.n_initial_points:,:]
            self.x_mean_tuple = (np.mean(self.x_hist[:self.n_initial_points,:], axis=0), np.mean(x_hist, axis=0))
            self.x_std_tuple = (np.std(self.x_hist[:self.n_initial_points,:], axis=0), np.std(x_hist, axis=0))


    def update_aoc(self, optimal_value = None):
        if self.y_hist is None:
            return 
        
        y_hist = self.y_hist
        aoc = ConvergenceCurveAnalyzer(min_y=optimal_value).calculate_aoc(y_hist)
        self.y_aoc = aoc

        if self.n_initial_points > 0 and len(y_hist) > self.n_initial_points: 
            y_hist = self.y_hist[self.n_initial_points:]
            aoc = ConvergenceCurveAnalyzer(min_y=optimal_value).calculate_aoc(y_hist)
            self.non_init_y_aoc = aoc

class EvaluatorResult:
    """Result of evaluating an individual."""
    def __init__(self):
        self.optimal_value = None
        self.budget = None
        self.captured_output = None
        self.error = None
        self.error_type = None
        self.result:EvaluatorBasicResult = EvaluatorBasicResult()
        self.other_results:list[EvaluatorBasicResult] = []
        self.metadata = {}

    def set_capture_output(self, captured_output):
        if captured_output is None or captured_output.strip() == "":
            return

        self.metadata["ori_captured_output"] = captured_output
        
        captured_output_list = captured_output.split("\n")
        captured_output_list = [line for line in captured_output_list if line.strip() != ""] 

        # find the unique lines
        captured_output_list = list(set(captured_output_list))

        # filter do not contain anchor ":<number>:", then capture the sub string after the anchor. 
        new_captured_output_list = []
        for line in captured_output_list:
            match = re.search(r"\:\d+\:", line)
            if match:
                new_captured_output_list.append(line[match.end():])
    
        # strip the leading and trailing white spaces
        new_captured_output_list = [line.strip() for line in new_captured_output_list]
        new_captured_output = "\n".join(new_captured_output_list)
        self.captured_output = new_captured_output

class AbstractEvaluator(ABC):
    def __init__(self):
        self.return_checker:Callable[[tuple], str] = None
    
    @abstractmethod
    def problem_prompt(self) -> str:
        pass

    @abstractmethod
    def problem_name(self) -> str:
        pass

    @abstractmethod
    def problem_dim(self) -> int:
        pass

    @abstractmethod
    def eval_bugdet(self) -> int:
        pass

    @abstractmethod
    def evaluate(self, code, cls_name) -> EvaluatorResult:
        pass
        

class RandomBoTorchTestEvaluator(AbstractEvaluator):
    def __init__(self, budget: int = 40, dim: int = 6, obj_fn_name: str = None):
        self.obj_fn_cls = get_test_function_by_name(obj_fn_name)
        self.dim = dim
        self.budget = budget
        self.evaluating = False
        self.obj_fn = None

        if self.obj_fn_cls is not None:
            # signature = inspect.signature(self.obj_fn_cls)
            self.obj_name = self.obj_fn_cls.__name__
            try:
                self.obj_fn = self.obj_fn_cls(dim=self.dim)
            except Exception:
                self.obj_fn = self.obj_fn_cls()
                self.dim = self.obj_fn.dim
            
            self.bounds = self.obj_fn.bounds.numpy()

            logging.info(f"{self.obj_name}:{self.bounds[0], self.bounds[1]},budget: {self.budget}")

    def problem_dim(self) -> int:
        return self.dim

    def eval_bugdet(self) -> int:
        return self.budget

    def problem_name(self) -> str:
        return self.obj_name

    def problem_prompt(self) -> str:
        prompt = ''
        # if self.
        if self.obj_fn.__doc__ is not None:
            prompt += self.obj_fn.__doc__
        else:
            prompt += f"the {self.obj_name} function"
        prompt += f"\ndimensions:{self.dim}, Bounds: {self.bounds[0], self.bounds[1]}"
        return prompt

    def __loading_indicator(self, message):
        symbols = itertools.cycle(['|', '/', '-', '\\'])
        while self.evaluating:
            sys.stdout.write("\rEvaluating " + message + "... " + next(symbols))
            sys.stdout.flush()
            time.sleep(0.1)

    def loading_indicator(self, message):
        thread = threading.Thread(target=self.__loading_indicator, args=(message,))
        thread.start()

    def evaluate(self, code, cls_name) -> EvaluatorResult:
        """Evaluate an individual."""

        class botorchObjectivexFn:
            def __init__(self, obj_fn, budget=None):
                self.obj_fn = obj_fn
                self.x_hist = None
                self.y_hist = None
                self.budget = budget
                self.progres_bar = tqdm(total=budget, desc="Evaluating")

            def __call__(self, x):
                if self.x_hist is not None and self.budget is not None and len(self.x_hist) >= self.budget:
                    raise Exception("OverBudgetException")
                
                if self.x_hist is None:
                    self.x_hist = x
                else:
                    self.x_hist = np.vstack((self.x_hist, x))

                tensor_x = torch.tensor(x, dtype=torch.float64)
                tensor_y = self.obj_fn(tensor_x)

                y = tensor_y.reshape(-1,1).numpy()
                if self.y_hist is None:
                    self.y_hist = y
                else:
                    self.y_hist = np.append(self.y_hist, y)
                self.progres_bar.update(len(x))
                return y

        eval_result = EvaluatorResult()
        eval_result.budget = self.budget

        if code is None:
            eval_result.error = "No code generated"
            eval_result.error_type = "NoCodeGenerated"
            return eval_result

        bo_obj_fn = botorchObjectivexFn(self.obj_fn, budget=self.budget)
        # self.evaluating = True
        # self.loading_indicator(f"{cls_name}")
        start_time = time.perf_counter()
        res, captured_output, err = default_exec(code, cls_name, bo_obj_fn, self.bounds, self.budget)
        # self.evaluating = False
        eval_result.result.execution_time = time.perf_counter() - start_time
        eval_result.set_capture_output(captured_output)

        if err is not None:
            eval_result.error = str(err)
            eval_result.error_type = err.__class__.__name__
        
        if eval_result.error is None and self.return_checker is not None:
            # check the return value
            return_check_str = self.return_checker(res)
            if len(return_check_str) > 0:
                eval_result.error = return_check_str
                eval_result.error_type = "ReturnCheckError"

        if eval_result.error is None:
            
            y_hist, x_hist, surragate_model_losses, n_initial_points = res

            optimal_value = None
            try:
                optimal_value = self.obj_fn.optimal_value
            except Exception:
                pass
            eval_result.result.name = cls_name
            eval_result.optimal_value = optimal_value
            eval_result.result.y_hist = y_hist.reshape(-1) if len(y_hist.shape) > 1 else y_hist
            eval_result.result.x_hist = x_hist
            eval_result.result.surragate_model_losses = surragate_model_losses[0]
            eval_result.result.model_loss_name = surragate_model_losses[1]
            eval_result.result.n_initial_points = n_initial_points
            eval_result.result.update_stats()
            eval_result.result.update_aoc(optimal_value)

            # Random search
            random_obj_fn = botorchObjectivexFn(self.obj_fn)
            random_search_result = EvaluatorBasicResult()
            random_search_result.name = "Random Search"
            start_time = time.perf_counter()
            rs_Y, rs_X = random_search(random_obj_fn, self.bounds, self.budget)
            random_search_result.y_hist = rs_Y.reshape(-1) if len(rs_Y.shape) > 1 else rs_Y
            random_search_result.x_hist = rs_X
            random_search_result.execution_time = time.perf_counter() - start_time
            random_search_result.update_stats()
            random_search_result.update_aoc(optimal_value)
            eval_result.other_results.append(random_search_result)

        return eval_result

