import io
import sys
import time
import contextlib
import itertools
import threading
import logging
import inspect
import random
from tqdm import tqdm
import numpy as np
import torch
from botorch.test_functions import synthetic
from botorch.test_functions.synthetic import SyntheticTestFunction, ConstrainedSyntheticTestFunction
from llamea.utils import BOOverBudgetException
from .evaluator import AbstractEvaluator 
from .evaluator_result import EvaluatorResult, EvaluatorBasicResult
from .exec_utils import default_exec

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

    
    
# Botorch test functions
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
            y_hist, x_hist, n_initial_points = res

            eval_basic_result.bounds = self.bounds
            eval_basic_result.optimal_value = self.optimal_value
            eval_basic_result.y_hist = y_hist.reshape(-1) if len(y_hist.shape) > 1 else y_hist
            eval_basic_result.x_hist = x_hist
            eval_basic_result.n_initial_points = n_initial_points
            eval_basic_result.update_stats()
            eval_basic_result.update_aoc(optimal_value=self.optimal_value)
        else:
            eval_result.error = eval_basic_result.error
            eval_result.error_type = eval_basic_result.error_type

        eval_result.result.append(eval_basic_result)
        eval_result.score = eval_basic_result.best_y

        return eval_result