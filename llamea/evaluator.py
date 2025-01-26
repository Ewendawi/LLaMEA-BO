import random
import traceback
import re
import sys
import io
import logging
from collections.abc import Callable
from typing import Any
import contextlib
import threading
import time
import os
import concurrent.futures
import itertools
from abc import ABC, abstractmethod
import inspect
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from botorch.test_functions import synthetic
from botorch.test_functions.synthetic import SyntheticTestFunction, ConstrainedSyntheticTestFunction

from .individual import Individual
from .utils import BOOverBudgetException, plot_result

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
    Y = objective_fn(X)
    # Y = np.array([objective_fn(x) for x in X])

    return Y, X

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


def __default_exec(code, cls_name, cls=None, init_kwargs=None, call_kwargs=None) -> tuple[any, str, str]:
    captured_output = io.StringIO()
    res = None
    err = None

    if init_kwargs is None:
        init_kwargs = {}
    if call_kwargs is None:
        call_kwargs = {}

    if cls is not None:
        # helper for debugging
        cls_instance = cls(**init_kwargs)
        with contextlib.redirect_stderr(captured_output), contextlib.redirect_stdout(captured_output):
            res = cls_instance(**call_kwargs)
    else:
        try:
            namespace: dict[str, Any] = {}
            track_exec(code, cls_name, namespace)

            if cls_name not in namespace:
                err = NameError(f"No '{cls_name}' found in the generated code")
            else:
                with contextlib.redirect_stderr(captured_output), contextlib.redirect_stdout(captured_output):
                    bo_cls = namespace[cls_name]
                    bo = bo_cls(**init_kwargs)
                    res = bo(**call_kwargs)
        except Exception as e:
            formatted_traceback = format_track_exec_with_code(cls_name, code, sys.exc_info())
            err = e.__class__(formatted_traceback)

    return res, captured_output.getvalue(), err

def default_exec(code, cls_name, init_kwargs=None, call_kwargs=None, time_out:float=None, cls=None) -> tuple[any, str, str]:
    if time_out is None:
        return __default_exec(code=code, cls_name=cls_name, cls=cls, init_kwargs=init_kwargs, call_kwargs=call_kwargs)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            params = {
                "code": code,
                "cls_name": cls_name,
                "cls": cls,
                "init_kwargs": init_kwargs,
                "call_kwargs": call_kwargs
            }
            future = executor.submit(__default_exec, **params)
            done, not_done = concurrent.futures.wait([future], timeout=time_out)
            if done:
                return future.result()
            if not_done:
                err = TimeoutError("Evaluation timed out")
                future.cancel()
                return None, None, err

#========================================
#Evaluator
#========================================


class ConvergenceCurveAnalyzer:
    """Analyzes optimization convergence curves and calculates AOC metric."""

    def __init__(self, max_y=None, min_y=None, log_scale=False, shift_value=0):
        self.max_y = max_y
        self.min_y = min_y
        self.log_scale = log_scale
        self.shift_value = shift_value

    def get_convergence_curve(self, y_history):
        """Calculate minimum values seen so far at each step."""
        if not isinstance(y_history, np.ndarray):
            y_history = np.array(y_history)
        return np.minimum.accumulate(y_history)

    def calculate_aoc(self, y_history):
        """Calculate area over convergence curve."""
        if len(y_history) == 0:
            return 0.0

        shift_y = y_history - self.shift_value
        max_y = self.max_y if self.max_y is not None else np.max(y_history)
        min_y = self.min_y if self.min_y is not None else np.min(y_history)
        clip_y = np.clip(shift_y, min_y, max_y)
        conv_curve = self.get_convergence_curve(clip_y)

        if self.log_scale:
            log_conv_curve = np.log10(conv_curve)
            log_max_y = np.log10(max_y)
            log_min_y = np.log10(min_y)
            norm_curve = (log_conv_curve - log_min_y) / (log_max_y - log_min_y)
        else:
            norm_curve = (conv_curve - min_y) / (max_y - min_y)

        # Calculate AOC using trapezoidal rule
        x_vals = np.linspace(0, 1, len(norm_curve))
        aoc = np.trapz(1-norm_curve, x_vals)

        return aoc

# y = np.array([100, 79, 81, 71, 65, 15, -5, 45, 5, 5])
# aoc = ConvergenceCurveAnalyzer(max_y=200, min_y=0, log_scale=False, shift_value=-10).calculate_aoc(y)
# print(aoc)

# aoc1 = ConvergenceCurveAnalyzer(max_y=1e2, min_y=1e-8, log_scale=True, shift_value=shift_value).calculate_aoc(y)
# print(aoc1)

class EvaluatorBasicResult:
    def __init__(self):
        self.id = None
        self.name = None
        self.optimal_value = None
        self.bounds = None
        self.budget = None
        self.captured_output = None
        self.error = None
        self.error_type = None
        
        self.execution_time = 0
        self.y_hist:np.ndarray = None
        self.x_hist:np.ndarray = None

        self._surrogate_model_losses:np.ndarray = None
        self.model_loss_name:str = None

        self.acquisition_function_values:np.ndarray = None

        self.best_y = None
        self.best_x = None

        self.y_aoc = 0.0
        self.y_aoc_from_ioh = 0.0
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
    def surrogate_model_losses(self):
        return self._surrogate_model_losses

    @surrogate_model_losses.setter
    def surrogate_model_losses(self, value):
        self._surrogate_model_losses = value

    def __to_json__(self):
        d = {}
        d["name"] = self.name
        d["optimal_value"] = self.optimal_value
        d["bounds"] = self.bounds.tolist() if self.bounds is not None else None
        d["budget"] = self.budget
        d["captured_output"] = self.captured_output
        d["error"] = self.error
        d["error_type"] = self.error_type
        
        d["execution_time"] = self.execution_time
        d["y_hist"] = self.y_hist.tolist() if self.y_hist is not None else None
        d["x_hist"] = self.x_hist.tolist() if self.x_hist is not None else None

        d["surrogate_model_losses"] = np.nan_to_num(self.surrogate_model_losses).tolist() if self.surrogate_model_losses is not None else None
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

    def update_aoc(self, optimal_value = None, log_scale=False, min_y=None, max_y=None):
        if self.y_hist is None:
            return

        y_hist = self.y_hist
        y_aoc = ConvergenceCurveAnalyzer(max_y=max_y, min_y=min_y, log_scale=log_scale, shift_value=optimal_value).calculate_aoc(y_hist)
        self.y_aoc = y_aoc

        if self.n_initial_points > 0 and len(y_hist) > self.n_initial_points:
            y_hist = self.y_hist[self.n_initial_points:]
            non_init_y_aoc = ConvergenceCurveAnalyzer(max_y=max_y, min_y=min_y, log_scale=log_scale, shift_value=optimal_value).calculate_aoc(y_hist)
            self.non_init_y_aoc = non_init_y_aoc

    def set_capture_output(self, captured_output):
        if captured_output is None or captured_output.strip() == "":
            return

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

    def __str__(self):
        return f"{self.name}\nbest_y:{self.best_y:.2f}, aoc:{self.y_aoc:.2f}, time:{self.execution_time:.2f}"


class EvaluatorResult:
    """Result of evaluating an individual."""
    def __init__(self):
        self.name = None
        self.score = None
        self.similarity = None
        self.error = None
        self.error_type = None
        
        self.result:list[EvaluatorBasicResult] = []

    def __to_json__(self):
        d = {}
        d["name"] = self.name
        d["error"] = self.error
        d["error_type"] = self.error_type
        if hasattr(self, "score"):
            d["score"] = self.score
        d["result"] = [r.__to_json__() for r in self.result]
        return d

    def __str__(self):
        if self.error is not None:
            return f"{self.name}\n{self.error}\n"
        else:
            return f"{self.name}, score:{self.score:.4f}"

class AbstractEvaluator(ABC):
    def __init__(self):
        self.return_checker:Callable[[tuple], str] = lambda x: ""

    @abstractmethod
    def problem_prompt(self) -> str:
        pass

    @abstractmethod
    def is_maximization(self) -> bool:
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
    def evaluate(self, code, cls_name, cls=None, max_eval_workers:int = 0, timeout:int=None) -> EvaluatorResult:
        pass

    def evaluate_others(self) -> list[EvaluatorResult]:
        pass

    @classmethod
    def plot_results(cls, results:list[tuple[str,list[EvaluatorResult]]], 
                     other_results:list[EvaluatorResult] = None, **kwargs):
        pass

class BotorchObjectivexFn:
    def __init__(self, obj_fn, budget=None):
        self.obj_fn = obj_fn
        self.x_hist = None
        self.y_hist = None
        self.budget = budget
        self.progress_bar = tqdm(total=budget, desc="Evaluating")

    def __call__(self, x):
        if self.x_hist is not None and self.budget is not None and len(self.x_hist) >= self.budget:
            raise BOOverBudgetException("OverBudgetException", "The total number(during the whole process) of the sample points which evaluated by objective_fn should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the objective_fn to evaluate the points is a alternative option.")

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
        self.progress_bar.update(len(x))
        return y

class RandomBoTorchTestEvaluator(AbstractEvaluator):
    def __init__(self, budget: int = 40, dim: int = 6, obj_fn_name: str = None):
        super().__init__()
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

            logging.info("%s:%s,budget: %s", self.obj_name, (self.bounds[0], self.bounds[1]), self.budget)

        self.optimal_value = None
        try:
            self.optimal_value = self.obj_fn.optimal_value
        except Exception:
            pass

    def is_maximization(self) -> bool:
        return False

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

    def evaluate_others(self) -> dict[str, EvaluatorResult]:
        # Random search
        other_results = []

        eval_result = EvaluatorResult()
        eval_result.name = "Random Search"

        random_obj_fn = BotorchObjectivexFn(self.obj_fn)
        random_search_result = EvaluatorBasicResult()
        start_time = time.perf_counter()
        rs_Y, rs_X = random_search(random_obj_fn, self.bounds, self.budget)
        random_search_result.y_hist = rs_Y.reshape(-1) if len(rs_Y.shape) > 1 else rs_Y
        random_search_result.x_hist = rs_X
        random_search_result.execution_time = time.perf_counter() - start_time
        random_search_result.update_stats()
        random_search_result.update_aoc(optimal_value=self.optimal_value)

        eval_result.result.append(random_search_result)
        
        other_results.append(eval_result)

        return other_results

    def evaluate(self, code, cls_name, cls=None, max_eval_workers:int = -1, timeout:int=None) -> EvaluatorResult:
        """Evaluate an individual."""

        eval_result = EvaluatorResult()
        eval_result.name = cls_name

        if code is None:
            eval_result.error = "No code generated"
            eval_result.error_type = "NoCodeGenerated"
            return eval_result

        eval_basic_result = EvaluatorBasicResult()
        eval_basic_result.budget = self.budget

        bo_obj_fn = BotorchObjectivexFn(self.obj_fn, budget=self.budget)
        # self.evaluating = True
        # self.loading_indicator(f"{cls_name}")
        start_time = time.perf_counter()
        init_kwargs = {}
        call_kwargs = {
            "objective_fn": bo_obj_fn,
            "bounds": self.bounds,
            "budget": self.budget
        }
        if cls is not None:
            # helper for debugging
            cls_instance = cls(**init_kwargs)
            captured_output_stream = io.StringIO()
            with contextlib.redirect_stderr(captured_output_stream), contextlib.redirect_stdout(captured_output_stream):
                res = cls_instance.optimize(**call_kwargs)
            captured_output = captured_output_stream.getvalue()
            err = None
        else:
            res, captured_output, err = default_exec(code=code, cls_name=cls_name, cls=cls, init_kwargs=init_kwargs, call_kwargs=call_kwargs, time_out=timeout)
        # self.evaluating = False
        eval_basic_result.execution_time = time.perf_counter() - start_time
        eval_basic_result.set_capture_output(captured_output)

        if err is not None:
            eval_basic_result.error = str(err)
            eval_basic_result.error_type = err.__class__.__name__
            

        if eval_basic_result.error is None and self.return_checker is not None:
            # check the return value
            return_check_str = self.return_checker(res)
            if len(return_check_str) > 0:
                eval_basic_result.error = return_check_str
                eval_basic_result.error_type = "ReturnCheckError"

        if eval_basic_result.error is None:
            y_hist, x_hist, surrogate_model_losses, n_initial_points = res

            eval_basic_result.bounds = self.bounds
            eval_basic_result.optimal_value = self.optimal_value
            eval_basic_result.y_hist = y_hist.reshape(-1) if len(y_hist.shape) > 1 else y_hist
            eval_basic_result.x_hist = x_hist
            eval_basic_result.surrogate_model_losses = surrogate_model_losses[0]
            eval_basic_result.model_loss_name = surrogate_model_losses[1]
            eval_basic_result.n_initial_points = n_initial_points
            eval_basic_result.update_stats()
            eval_basic_result.update_aoc(optimal_value=self.optimal_value)
        else:
            eval_result.error = eval_basic_result.error
            eval_result.error_type = eval_basic_result.error_type

        eval_result.result.append(eval_basic_result)
        eval_result.score = eval_basic_result.best_y

        return eval_result


    @classmethod
    def evaluate_individual(cls, ind:Individual, problem:str=None, dim:int = 6, budget:int = 40):
        problem_name = problem
        if problem_name is None:
            problem_name = ind.metadata["problem"]

        evaluator = cls(dim=dim, budget=budget, obj_fn_name=problem_name)
        return evaluator.evaluate(ind.solution, ind.name)

    @classmethod
    def evaluate_from_cls(cls, bo_cls, problem:str=None, dim:int = 6, budget:int = 40, eval_others:bool = False):
        evaluator = cls(dim=dim, budget=budget, obj_fn_name=problem)
        res = evaluator.evaluate("code", bo_cls.__name__, cls=bo_cls)
        other_results = None
        if eval_others:
            other_results = evaluator.evaluate_others()
        return res, other_results


    
from ioh import get_problem, logger
from misc import aoc_logger, correct_aoc, OverBudgetException

class IOHObjectiveFn:
    def __init__(self, problem_id, instance_id, exec_id, dim, budget, show_progress_bar=False):
        self.problem_id = problem_id
        self.instance_id = instance_id
        self.exec_id = exec_id
        self.dim = dim
        self.budget = budget

        self.obj_fn = get_problem(problem_id, instance_id, dim)
        self.optimal_value = self.obj_fn.optimum.y
        self.name = f"F{problem_id}-{self.obj_fn.problems[problem_id]}"

        lb = self.obj_fn.bounds.lb
        ub = self.obj_fn.bounds.ub
        p_bounds = np.array([lb, ub])
        self.bounds = p_bounds

        self.x_hist = None
        self.y_hist = None
        self.aoc = 0.0

        self.progress_bar = None
        self.show_progress_bar = show_progress_bar

    def reset(self):
        self.obj_fn = None
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None

    @property
    def show_progress_bar(self):
        return self._show_progress_bar
    
    @show_progress_bar.setter
    def show_progress_bar(self, value):
        self._show_progress_bar = value
        if self._show_progress_bar:
            self.progress_bar = tqdm(total=self.budget, desc=f"Evaluating {self.name}") 
        else:
            if self.progress_bar is not None:
                self.progress_bar.close()
                self.progress_bar = None

    def __call__(self, x):
        if self.obj_fn is not None and self.budget is not None and self.obj_fn.state.evaluations >= self.budget:
            raise BOOverBudgetException("OverBudgetException", "The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.")

        if self.x_hist is None:
            self.x_hist = x
        else:
            self.x_hist = np.vstack((self.x_hist, x))

        y = self.obj_fn(x)

        if self.y_hist is None:
            if isinstance(y, list):
                self.y_hist = np.array(y).reshape(-1,1)
            else:
                self.y_hist = np.array([y]).reshape(-1,1)
        else:
            if isinstance(y, list):
                self.y_hist = np.append(self.y_hist, np.array(y).reshape(-1,1))
            else:
                self.y_hist = np.append(self.y_hist, np.array([y]).reshape(-1,1))

        if self.show_progress_bar:
            progress = 1
            if len(x.shape) > 1:
                progress = x.shape[0]
            self.progress_bar.update(progress)
        else:
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                progress = len(self.x_hist)
                interval = self.budget // 4
                if progress % interval == 0:
                    msg = f"{self.name}-{self.instance_id}-{self.exec_id}:{progress}/{self.budget} evaluations completed"
                    logging.debug(msg)

        if isinstance(y, list):
            return np.array(y).reshape(-1,1)
        return y

def ioh_evaluate_block(problem_id, instance_id, exec_id, dim, budget, code, cls_name, cls=None, time_out:int=None) -> EvaluatorBasicResult:

    obj_fn = IOHObjectiveFn(problem_id=problem_id, instance_id=instance_id, exec_id=exec_id, dim=dim, budget=budget, show_progress_bar=False)

    l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
    obj_fn.obj_fn.attach_logger(l2)
        
    start_time = time.perf_counter()

    init_kwargs = {
        "dim": dim,
        "budget": budget
    }
    call_kwargs = {
        "func": obj_fn
    }

    res, captured_output, err = default_exec(code=code, cls_name=cls_name, cls=cls, init_kwargs=init_kwargs, call_kwargs=call_kwargs, time_out=time_out)
    exec_time = time.perf_counter() - start_time

    # unset the unpicklable object
    aoc = correct_aoc(obj_fn.obj_fn, l2, budget)
    obj_fn.aoc = aoc
    obj_fn.reset()

    return res, captured_output, err, exec_time, obj_fn
    

class IOHEvaluator(AbstractEvaluator):

    def __str__(self):
        return f"IOHEvaluator: {self._problem_name}_dim-{self.dim}_budget-{self.budget}_instances-{self.instances}_repeat-{self.reapeat}"
    
    def __init__(self, dim:int = 5, budget:int = 40, problems:list[int]= None, instances:list[list[int]]=None, repeat:int = 1):
        super().__init__()
        if problems is not None and instances is not None and len(problems) != len(instances):
            raise ValueError("The length of problems and instances should be the same")
        
        feasible_dim = [2, 3, 5, 10, 20, 40]
        if dim not in feasible_dim:
            raise ValueError(f"dim should be in {feasible_dim}")

        self.problems = None
        feasible_problems = list(range(1, 25))
        if problems is not None:
            for problem in problems:
                if problem not in feasible_problems:
                    raise ValueError("problem should be in range(1, 25)")
            self.problems = problems
        else:
            # https://numbbo.github.io/coco/testsuites/bbob
            # separable_problems = list(range(1, 6))
            low_conditioning_problems = list(range(6, 10))
            high_conditioning_problems = list(range(10, 15))
            adequate_structure_problems = list(range(15, 20))
            weak_structure_problems = list(range(20, 25))
            group_problems = [low_conditioning_problems, high_conditioning_problems, adequate_structure_problems, weak_structure_problems]

            selected_problems = [random.choice(group) for group in group_problems]
            self.problems = random.sample(selected_problems, 1)

        feasible_instances = list(range(1, 15))
        self.instances = None
        if instances is not None:
            for p_instances in instances:
                for instance in p_instances:
                    if instance not in feasible_instances:
                        raise ValueError(f"instance should be in {feasible_instances}")
            self.instances = instances
        else:
            self.instances = [random.sample(feasible_instances, 1)] * len(self.problems)
        
        self.reapeat = repeat
        self.dim = dim
        self.budget = budget

        obj_fn_params = []
        for problem, instances in zip(self.problems, self.instances):
            for instance in instances:
                for i in range(self.reapeat):
                    params = {
                        "problem_id": problem,
                        "instance_id": instance,
                        "exec_id": i,
                        "dim": self.dim,
                        "budget": self.budget
                    }
                    obj_fn_params.append(params)
        
        self.obj_fn_params = obj_fn_params

        problem_name = "bbob_" + "_".join([f"f{problem}" for problem in self.problems])
        self._problem_name = problem_name

    def is_maximization(self) -> bool:
        return True

    def problem_dim(self) -> int:
        return self.dim

    def eval_bugdet(self) -> int:
        return self.budget

    def problem_name(self) -> str:
        return self._problem_name

    def problem_prompt(self) -> str:
        prompt = f'Problems from the BBOB test suite with dimensions {self.dim}\n'
        return prompt

    def evaluate_others(self) -> list[EvaluatorResult]:
        # Random search
        other_results = []

        eval_result = EvaluatorResult()
        eval_result.name = "Random Search"
        progress_bar = tqdm(total=len(self.obj_fn_params), desc="Evaluating Random Search")
        for params in self.obj_fn_params:
            random_obj_fn = IOHObjectiveFn(**params)
            optimal_value = random_obj_fn.obj_fn.optimum.y
            rs_result = EvaluatorBasicResult()
            rs_result.budget = self.budget
            rs_result.optimal_value = optimal_value
            rs_result.bounds = random_obj_fn.bounds
            rs_result.name = random_obj_fn.name
            start_time = time.perf_counter()
            rs_y, rs_x = random_search(random_obj_fn, random_obj_fn.bounds, self.budget)
            rs_result.y_hist = rs_y.reshape(-1) if len(rs_y.shape) > 1 else rs_y
            rs_result.x_hist = rs_x
            rs_result.execution_time = time.perf_counter() - start_time
            rs_result.update_stats()
            rs_result.update_aoc(optimal_value=optimal_value, log_scale=True, min_y=1e-8, max_y=1e2)

            eval_result.result.append(rs_result)
            progress_bar.update(1)

        progress_bar.close()
        eval_result.score = np.mean([r.y_aoc for r in eval_result.result])

        other_results.append(eval_result)

        return other_results

    def __process_results(self, res, captured_output, err, exec_time, obj_fn):
        eval_basic_result = EvaluatorBasicResult()
        eval_basic_result.id = f"{obj_fn.problem_id}-{obj_fn.instance_id}-{obj_fn.exec_id}"
        eval_basic_result.budget = obj_fn.budget
        eval_basic_result.name = obj_fn.name
        eval_basic_result.bounds = obj_fn.bounds
        eval_basic_result.execution_time = exec_time
        eval_basic_result.set_capture_output(captured_output)

        if err is not None:
            eval_basic_result.error = str(err)
            eval_basic_result.error_type = err.__class__.__name__

        if eval_basic_result.error is None and self.return_checker is not None:
            # check the return value
            return_check_str = self.return_checker(res)
            if len(return_check_str) > 0:
                eval_basic_result.error = return_check_str
                eval_basic_result.error_type = "ReturnCheckError"

        if eval_basic_result.error is None:
            # best_y, best_x = res
            y_hist = obj_fn.y_hist
            x_hist = obj_fn.x_hist

            # y_hist, x_hist, surrogate_model_losses, n_initial_points = res
            eval_basic_result.y_aoc_from_ioh = obj_fn.aoc

            eval_basic_result.name = obj_fn.name
            eval_basic_result.bounds = obj_fn.bounds
            eval_basic_result.optimal_value = obj_fn.optimal_value
            eval_basic_result.y_hist = y_hist.reshape(-1) if len(y_hist.shape) > 1 else y_hist
            eval_basic_result.x_hist = x_hist
            # eval_basic_result.surrogate_model_losses = surrogate_model_losses[0]
            # eval_basic_result.model_loss_name = surrogate_model_losses[1]
            # eval_basic_result.n_initial_points = n_initial_points
            eval_basic_result.update_stats()
            eval_basic_result.update_aoc(optimal_value=obj_fn.optimal_value, log_scale=False, min_y=1e-8, max_y=1e2)

        return eval_basic_result

    def evaluate(self, code, cls_name, cls=None, max_eval_workers:int = -1, timeout:int=None) -> EvaluatorResult:
        """Evaluate an individual."""
        eval_result = EvaluatorResult()
        eval_result.name = cls_name
        if code is None:
            eval_result.error = "No code generated"
            eval_result.error_type = "NoCodeGenerated"
            return eval_result

        params = []
        for param in self.obj_fn_params:
            new_param = {
                "code": code,
                "cls_name": cls_name,
                "cls": cls,
                "time_out": timeout
            }
            new_param.update(param)
            params.append(new_param)

        total_tasks = len(params)
        interval = min(max(1, total_tasks // 4), 20)

        if max_eval_workers is None or max_eval_workers > 0:
            max_workers = min(os.cpu_count() - 1, max_eval_workers)

            # if cuda is available, use thread pool executor
            # if torch.cuda.is_available() and "cuda" in code:
            #     logging.info("Evaluating %s: %s tasks, using ThreadPoolExecutor with %s max_workers", cls_name, total_tasks, max_workers)
            #     executor_cls = concurrent.futures.ThreadPoolExecutor
            # else:
            #     logging.info("Evaluating %s: %s tasks, using ProcessPoolExecutor with %s max_workers", cls_name, total_tasks, max_workers)
            #     executor_cls = concurrent.futures.ProcessPoolExecutor

            logging.info("Evaluating %s: %s tasks, using ThreadPoolExecutor with %s max_workers", cls_name, total_tasks, max_workers)
            executor_cls = concurrent.futures.ThreadPoolExecutor
                
            with executor_cls(max_workers=max_workers) as executor:
                futures = {executor.submit(ioh_evaluate_block, **param): param for param in params}
                for future in concurrent.futures.as_completed(futures.keys()):
                    res, captured_output, err, exec_time, obj_fn = future.result()
                    eval_basic_result = self.__process_results(res, captured_output, err, exec_time, obj_fn)
                    if eval_basic_result.error is not None:
                        eval_result.error = eval_basic_result.error
                        eval_result.error_type = eval_basic_result.error_type
                        logging.info("Evaluating %s: Got Error - %s", cls_name, eval_basic_result.error_type)
                        executor.shutdown(wait=False)
                        break
                    else:
                        eval_result.result.append(eval_basic_result)
                        done_tasks = len(eval_result.result)
                        if done_tasks % interval == 0:
                            logging.info("Evaluating %s: %s/%s", cls_name, done_tasks, total_tasks)
        else:
            logging.info("Evaluating %s: %s tasks in sequence", cls_name, total_tasks)

            for param in params:
                res, captured_output, err, exec_time, obj_fn = ioh_evaluate_block(**param)
                eval_basic_result = self.__process_results(res, captured_output, err, exec_time, obj_fn)
                if eval_basic_result.error is not None:
                    eval_result.error = eval_basic_result.error
                    eval_result.error_type = eval_basic_result.error_type
                    logging.info("Evaluating %s: Got Error - %s", cls_name, eval_basic_result.error_type)
                    break
                else:
                    eval_result.result.append(eval_basic_result)
                    done_tasks = len(eval_result.result)
                    if done_tasks % interval == 0:
                        logging.info("Evaluating %s: %s/%s", cls_name, done_tasks, total_tasks)

        if eval_result.error is None:
            # eval_result.score = np.mean([r.y_aoc for r in eval_result.result])
            eval_result.score = np.mean([r.y_aoc_from_ioh for r in eval_result.result])
            logging.info("Evaluated %s: %s", cls_name, eval_result.score)
        else:                           
            eval_result.score = 0.0

        return eval_result

    @classmethod
    def plot_results(cls, results:list[tuple[str,list[EvaluatorResult]]], 
                     other_results:list[EvaluatorResult] = None, **kwargs):
        #results: (n_strategies, n_generations, n_evaluations)
        column_names = [
            'strategy',
            'problem_id',
            'instance_id',
            'exec_id',
            'n_gen',
            "log_y_aoc",
            "y_aoc",
            "best_y",
            'loss'
            ]

        def res_to_row(res, gen:int):
            res_id = res.id
            res_split = res_id.split("-")
            problem_id = int(res_split[0])
            instance_id = int(res_split[1])
            repeat_id = int(res_split[2])
            row = {
                'strategy': strategy_name,
                'problem_id': problem_id,
                'instance_id': instance_id,
                'exec_id': repeat_id,
                'n_gen': gen+1,
                "log_y_aoc": res.y_aoc_from_ioh,
                "y_aoc": res.y_aoc,
                "best_y": res.best_y,
                'loss': abs(res.optimal_value - res.best_y)
            }
            return row
        
        res_df = pd.DataFrame(columns=column_names)
        for res_tuple in results:
            strategy_name, gen_list = res_tuple 
            for i, gen_res in enumerate(gen_list):
                if gen_res.error is not None:
                    continue
                for res in gen_res.result:
                    row = res_to_row(res, i) 
                    res_df.loc[len(res_df)] = row
        
        if other_results is not None:
            for other_res in other_results:
                for res in other_res.result:
                    row = res_to_row(res, -1)
                    res_df.loc[len(res_df)] = row

        # strategy-wise
        log_y_aocs, log_y_aoc_labels = [], []
        baseline_y_aoc, baseline_y_aoc_labels = [], []
        y_aocs, y_aoc_labels = [], []
        baseline_y_aoc, baseline_y_aoc_labels = [], []
        g_log_y_aoc = res_df.groupby(['strategy', 'n_gen'])[["log_y_aoc", "y_aoc"]].agg(np.mean).reset_index()
        max_gen = g_log_y_aoc['n_gen'].max()
        for name, group in g_log_y_aoc.groupby('strategy'):
            gens = group['n_gen'].values
            if len(gens) == 1 and gens[0] == -1:
                baseline_y_aoc.append(group['y_aoc'].values[0])
                baseline_y_aoc_labels.append(name)
                continue

            # fill the missing generations with 0
            aoc = np.zeros(max_gen)
            log_aoc = np.zeros(max_gen)
            for gen in gens:
                log_aoc[gen-1] = group[group['n_gen'] == gen]['log_y_aoc'].values[0]
                aoc[gen-1] = group[group['n_gen'] == gen]['y_aoc'].values[0]

            log_y_aocs.append(log_aoc)
            y_aocs.append(aoc)
            log_y_aoc_labels.append(name)
            y_aoc_labels.append(name)

        # problem-wise
        loss_list = [[] for _ in range(1, 25)]
        loss_labels = [[] for _ in range(1, 25)]
        aoc_list = [[] for _ in range(1, 25)]
        aoc_labels = [[] for _ in range(1, 25)]
        g_best_y = res_df.groupby(['strategy', 'n_gen', 'problem_id'])[['y_aoc', 'loss']].agg(np.mean).reset_index()
        max_gen = g_best_y['n_gen'].max()
        for (name, p_id), group in g_best_y.groupby(['strategy', 'problem_id']):
            gens = group['n_gen'].values
            if len(gens) == 1 and gens[0] == -1:
                continue

            aoc = np.zeros(max_gen)
            loss = np.zeros(max_gen)
            max_loss = group['loss'].max()
            missing_gens = set(range(1, max_gen+1)) - set(gens)
            for missing_gen in missing_gens:
                loss[missing_gen-1] = max_loss
            for gen in gens:
                loss[gen-1] = group[group['n_gen'] == gen]['loss'].values[0]
                aoc[gen-1] = group[group['n_gen'] == gen]['y_aoc'].values[0]

            aoc_list[p_id-1].append(aoc)
            aoc_labels[p_id-1].append(name)
            loss_list[p_id-1].append(loss)
            loss_labels[p_id-1].append(name)
            
        # plot aoc
        y = np.maximum.accumulate(np.array([log_y_aocs, y_aocs]), axis=2)
        base_x = np.arange(1, max_gen+1, dtype=int)
        x = np.tile(base_x, (y.shape[0], y.shape[1], 1))
        sub_titles = ["Log AOC", "AOC"]
        labels = [log_y_aoc_labels] * 2
        plot_result(y=y, x=x, labels=labels,
                    title=None, 
                    sub_titles=sub_titles, n_cols=2,
                    **kwargs)

        # plot loss
        # y = np.minimum.accumulate(np.array(loss_list), axis=2)
        # base_x = np.arange(1, max_gen+1, dtype=int)
        # x = np.tile(base_x, (y.shape[0], y.shape[1], 1))
        # sub_titles = [f"F{p_id}" for p_id in range(1, 25)]
        # labels = loss_labels * len(loss_list)
        # x_labels = ["Generations"] * len(loss_list)
        # n_cols = 6
        # for i, _ in enumerate(x_labels):
        #     if i < len(x_labels) - n_cols:
        #         x_labels[i] = ""
        # y_labels = ["Loss"] * len(loss_list)
        # for i, _ in enumerate(y_labels):
        #     if i % n_cols != 0:
        #         y_labels[i] = ""
        # plot_result(y=y, x=x, labels=labels, 
        #             title=None, figsize=(14, 8),
        #             x_labels=x_labels, y_labels=y_labels,
        #             sub_titles=sub_titles, n_cols=n_cols,
        #             **kwargs)
            
        # plot aoc
        # y = np.maximum.accumulate(np.array(aoc_list), axis=2)
        # base_x = np.arange(1, max_gen+1, dtype=int)
        # x = np.tile(base_x, (y.shape[0], y.shape[1], 1))
        # sub_titles = [f"F{p_id}" for p_id in range(1, 25)]
        # labels = aoc_labels * len(aoc_list)
        # plot_result(y=y, x=x, labels=labels, 
        #             title=None, figsize=(14, 8),
        #             sub_titles=sub_titles, n_cols=6,
        #             **kwargs)



        
    @classmethod
    def evaluate_from_cls(cls, bo_cls, problems:list[int]=None, dim:int = 5, budget:int = 40, eval_others:bool=False):
        evaluator = cls(dim=dim, budget=budget, problems=problems)
        res = evaluator.evaluate("code", bo_cls.__name__, cls=bo_cls)
        other_results = None
        if eval_others:
            other_results = evaluator.evaluate_others()
        return res, other_results
