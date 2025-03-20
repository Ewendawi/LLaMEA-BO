import random
import logging
from typing import Any
import time
import os
import concurrent.futures
from tqdm import tqdm
import numpy as np
from ioh import get_problem, logger
from misc import aoc_logger, correct_aoc

from llamea.utils import BOOverBudgetException

from .evaluator import AbstractEvaluator
from .evaluator_result import EvaluatorResult, EvaluatorBasicResult
from .exec_utils import default_exec

_logger = logging.getLogger(__name__)

class IOHObjectiveFn:
    def __init__(self, problem_id, instance_id, exec_id, dim, budget, show_progress_bar=False):
        self.problem_id = problem_id
        self.instance_id = instance_id
        self.exec_id = exec_id
        self.dim = dim
        self.budget = budget
        self.maximize = False

        self.obj_fn = get_problem(problem_id, instance_id, dim)
        self.optimal_value = self.obj_fn.optimum.y
        self.optimal_x = self.obj_fn.optimum.x
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
        self.ignore_over_budget = False

    def reset(self):
        self.obj_fn = None
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None

    def stateless_call(self, x):
        new_obj_fn = get_problem(self.problem_id, self.instance_id, self.dim)
        y = new_obj_fn(x)
        if self.maximize:
            y = -y
        return y

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
        if self.obj_fn is not None and self.budget is not None and self.obj_fn.state.evaluations > self.budget:
            _logger.error("%s Over budget: %s/%s", self.name, self.obj_fn.state.evaluations, self.budget)
            if not self.ignore_over_budget:
                raise BOOverBudgetException("OverBudgetException", "The total number(during the whole process) of the sample points which evaluated by func should not exceed the budget. Using the surrogate model, accquisition function or any other methods suited your purposes instead of the func to evaluate the points is a alternative option.")
        
        # check if x are nan
        if np.isnan(x).any():
            raise ValueError(f"x({x}) contains nan values")

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
            if _logger.isEnabledFor(logging.DEBUG):
                progress = len(self.x_hist)
                interval = self.budget // 4
                if progress % interval == 0:
                    msg = f"{self.name}-{self.instance_id}-{self.exec_id}:{progress}/{self.budget} evaluations completed"
                    _logger.debug(msg)
        
        if self.maximize:
            y = -y

        if isinstance(y, list):
            return np.array(y).reshape(-1,1)
        return y

def ioh_evaluate_block(problem_id, instance_id, exec_id, dim, budget, code, cls_name, cls=None, cls_init_kwargs=None, cls_call_kwargs=None, ignore_over_budget:bool=False, inject_critic:bool=False, ignore_capture:bool=True, ignore_metric=False) -> tuple[Any, str, str, float, IOHObjectiveFn, Any]: 

    obj_fn = IOHObjectiveFn(problem_id=problem_id, instance_id=instance_id, exec_id=exec_id, dim=dim, budget=budget, show_progress_bar=False)
    obj_fn.ignore_over_budget = ignore_over_budget

    l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
    obj_fn.obj_fn.attach_logger(l2)
        
    start_time = time.perf_counter()

    init_kwargs = {
        "dim": dim,
        "budget": budget,
    }
    if cls_init_kwargs is not None:
        init_kwargs.update(cls_init_kwargs)
    
    call_kwargs = {
        "func": obj_fn
    }
    if cls_call_kwargs is not None:
        call_kwargs.update(cls_call_kwargs)

    res, captured_output, err, critic = default_exec(code=code, cls_name=cls_name, cls=cls, init_kwargs=init_kwargs, call_kwargs=call_kwargs, inject_critic=inject_critic, ignore_metric=ignore_metric)
    exec_time = time.perf_counter() - start_time

    # unset the unpicklable object
    aoc = correct_aoc(obj_fn.obj_fn, l2, budget)
    obj_fn.aoc = aoc
    obj_fn.reset()

    if ignore_capture:
        captured_output = None

    return res, captured_output, err, exec_time, obj_fn, critic


class IOHEvaluator(AbstractEvaluator):
    def __str__(self):
        return f"IOHEvaluator: {self._problem_name}_dim-{self.dim}_budget-{self.budget}_instances-{self.instances[0]}_repeat-{self.reapeat}"
    
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
            separable_problems = list(range(1, 6))
            low_conditioning_problems = list(range(6, 10))
            high_conditioning_problems = list(range(10, 15))
            adequate_structure_problems = list(range(15, 20))
            weak_structure_problems = list(range(20, 25))
            group_problems = [
                separable_problems,
                low_conditioning_problems, 
                high_conditioning_problems, 
                adequate_structure_problems, 
                weak_structure_problems
            ]

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
            # self.instances = [random.sample(feasible_instances, 1)] * len(self.problems)
            self.instances = [[1]] * len(self.problems)
        
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

        problem_name = "_".join([f"f{problem}" for problem in self.problems])
        self._problem_name = problem_name

    def problem_dim(self) -> int:
        return self.dim

    def eval_bugdet(self) -> int:
        return self.budget

    def problem_name(self) -> str:
        return self._problem_name

    def problem_prompt(self) -> str:
        prompt = f'Problems from the BBOB test suite with dimensions {self.dim}\n'
        return prompt

    def __process_results(self, res, captured_output, err, exec_time, obj_fn, critic) -> EvaluatorBasicResult:
        eval_basic_result = EvaluatorBasicResult()
        eval_basic_result.id = f"{obj_fn.problem_id}-{obj_fn.instance_id}-{obj_fn.exec_id}"
        eval_basic_result.budget = obj_fn.budget
        eval_basic_result.name = obj_fn.name
        eval_basic_result.bounds = obj_fn.bounds
        eval_basic_result.execution_time = exec_time
        eval_basic_result.set_capture_output(captured_output)

        if err is not None:
            eval_basic_result.error = str(err)
            eval_basic_result.error_type = err.error_type

        if eval_basic_result.error is None and self.return_checker is not None:
            # check the return value
            return_check_str = self.return_checker(res)
            if len(return_check_str) > 0:
                eval_basic_result.error = return_check_str
                eval_basic_result.error_type = "ReturnCheckError"

        if eval_basic_result.error is None:
            # best_y, best_x = res
            y_hist = obj_fn.y_hist if len(obj_fn.y_hist) <= self.budget else obj_fn.y_hist[:self.budget] 
            x_hist = obj_fn.x_hist if len(obj_fn.x_hist) <= self.budget else obj_fn.x_hist[:self.budget]

            eval_basic_result.name = obj_fn.name
            eval_basic_result.bounds = obj_fn.bounds
            eval_basic_result.optimal_value = obj_fn.optimal_value
            eval_basic_result.optimal_x = obj_fn.optimal_x
            eval_basic_result.y_hist = y_hist.reshape(-1) if len(y_hist.shape) > 1 else y_hist
            eval_basic_result.x_hist = x_hist

            if critic is not None:
                eval_basic_result.n_initial_points = critic.n_init
                eval_basic_result.r2_list = critic.r_2_list
                eval_basic_result.r2_list_on_train = critic.r_2_list_on_train
                eval_basic_result.uncertainty_list = critic.uncertainty_list
                eval_basic_result.uncertainty_list_on_train = critic.uncertainty_list_on_train
                eval_basic_result.search_result = critic.search_result
                if not self.ignore_metric:
                    eval_basic_result.update_coverage()
                eval_basic_result.fill_short_data(obj_fn.budget)

            eval_basic_result.y_aoc_from_ioh = obj_fn.aoc
            eval_basic_result.update_stats()
            eval_basic_result.update_aoc(optimal_value=obj_fn.optimal_value, min_y=1e-8)

        return eval_basic_result

    def _check_timeout(self, start_time, timeout):
        if timeout is not None:
            _current_eval_time = time.perf_counter()
            _time_diff = _current_eval_time - start_time
            if _time_diff > timeout:
                return True
        return False

    def _post_process_error_check(self, eval_result: EvaluatorResult, eval_basic_result: EvaluatorBasicResult, timeout, start_time):
        _err = eval_basic_result.error
        _err_type = eval_basic_result.error_type
        if _err is None and self._check_timeout(start_time, timeout):
            _err = TimeoutError("Evaluation timed out (%d)", timeout)
            _err_type = "Timeout"
        if _err is not None:
            eval_result.error = _err 
            eval_result.error_type = _err_type
        else:
            eval_result.result.append(eval_basic_result)

    def _logging_eval_process(self, eval_result, interval, total_tasks):
        cls_name = eval_result.name
        if eval_result.error is not None:
            _logger.error("Evaluating %s: %s", cls_name, eval_result.error_type)
        else:
            done_tasks = len(eval_result.result)
            if done_tasks % interval == 0:
                _logger.info("Evaluating %s: %s/%s", cls_name, done_tasks, total_tasks)

    def start_as_completed(self,eval_result, futures, timeout, task_manager=None, executor=None, cls_name=None, interval=None, total_tasks=None):
        _should_cancel = False
        _as_completed = None
        if task_manager is not None:
            _as_completed = task_manager.as_completed(futures.keys(), timeout=timeout)
        else:
            _as_completed = concurrent.futures.as_completed(futures.keys(), timeout=timeout)
        try:
            for future in _as_completed:
                res = future.result()
                eval_basic_result = self.__process_results(*res)

                _err = eval_basic_result.error
                _err_type = eval_basic_result.error_type
                if _err is not None:
                    eval_result.error = _err 
                    eval_result.error_type = _err_type
                else:
                    eval_result.result.append(eval_basic_result)

                self._logging_eval_process(eval_result, interval, total_tasks)

                if eval_result.error is not None:
                    _should_cancel = True
                    break
        except TimeoutError as e:
            _err = TimeoutError("Evaluation timed out (%d)", timeout)
            _err_type = "Timeout"
            eval_result.error = _err
            eval_result.error_type = _err_type
            self._logging_eval_process(eval_result, interval, total_tasks)
            _should_cancel = True
        
        _logger.info("Evaluating %s: Shutting down executor", cls_name)
        # better to wait for all running tasks to finish in case of resource competition
        if task_manager is not None:
            task_manager.shutdown(wait=True, cancel_futures=_should_cancel)
        else:
            executor.shutdown(wait=True, cancel_futures=_should_cancel)
        _logger.info("Evaluating %s: Executor shut down", cls_name)

        return eval_result
            
    def evaluate(self, code, cls_name, cls=None, cls_init_kwargs:dict[str, Any]=None, cls_call_kwargs:dict[str, Any]=None) -> EvaluatorResult:
        """Evaluate an individual."""
        eval_result = EvaluatorResult()
        eval_result.name = cls_name
        if code is None and cls is None:
            eval_result.error = "No code generated"
            eval_result.error_type = "NoCodeGenerated"
            return eval_result

        params = []
        for param in self.obj_fn_params:
            new_param = {
                "code": code,
                "cls_name": cls_name,
                "cls": cls,
                "ignore_over_budget": self.ignore_over_budget,
                "inject_critic": self.inject_critic,
                "cls_init_kwargs": cls_init_kwargs,
                "cls_call_kwargs": cls_call_kwargs,
                'ignore_capture': self.ignore_capture,
                'ignore_metric': self.ignore_metric,
            }
            new_param.update(param)
            params.append(new_param)

        total_tasks = len(params)
        interval = min(max(1, total_tasks // 4), 20)

        _all_eval_time_start = time.perf_counter()
        
        max_eval_workers = self.max_eval_workers
        use_multi_process = self.use_multi_process
        use_mpi = self.use_mpi
        use_mpi_future = self.use_mpi_future
        timeout = self.timeout

        if use_mpi:
            from mpi4py import MPI
            from llamea.evaluator.MPITaskManager import MPITaskManager, MPIFuture

            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            _logger.info("Evaluating %s: %s tasks, using MPI with %s workers", cls_name, total_tasks, size-1)

            task_manager = MPITaskManager()
            futures = {task_manager.submit(ioh_evaluate_block, **param): param for param in params}
            self.start_as_completed(eval_result, futures, timeout, task_manager=task_manager, cls_name=cls_name, interval=interval, total_tasks=total_tasks)
        
        elif use_mpi_future:
            from mpi4py import MPI
            from mpi4py.futures import MPIPoolExecutor
            from mpi4py.futures import get_comm_workers
            
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()

            if size < 2:
                raise ValueError("Requires at least 2 MPI processes.")

            _logger.info("Evaluating %s: %s tasks, using MPI.futures with %s max_workers", cls_name, total_tasks, size)
                
            executor = MPIPoolExecutor(max_workers=size)  
            futures = {executor.submit(ioh_evaluate_block, **param): param for param in params}
            self.start_as_completed(eval_result, futures, timeout, executor=executor, cls_name=cls_name, interval=interval, total_tasks=total_tasks)

        elif max_eval_workers is None or max_eval_workers > 0:
            max_workers = min(os.cpu_count() - 1, max_eval_workers)
            if use_multi_process:
                _logger.info("Evaluating %s: %s tasks, using ProcessPoolExecutor with %s max_workers", cls_name, total_tasks, max_workers)
                executor_cls = concurrent.futures.ProcessPoolExecutor
            else:
                _logger.info("Evaluating %s: %s tasks, using ThreadPoolExecutor with %s max_workers", cls_name, total_tasks, max_workers)
                executor_cls = concurrent.futures.ThreadPoolExecutor

            executor = executor_cls(max_workers=max_workers)
            futures = {executor.submit(ioh_evaluate_block, **param): param for param in params}
            self.start_as_completed(eval_result, futures, timeout, executor=executor, cls_name=cls_name, interval=interval, total_tasks=total_tasks)
        else:
            _logger.info("Evaluating %s: %s tasks in sequence", cls_name, total_tasks)

            for param in params:
                res = ioh_evaluate_block(**param)
                eval_basic_result = self.__process_results(*res)

                _err = eval_basic_result.error
                _err_type = eval_basic_result.error_type
                if _err is None and self._check_timeout(_all_eval_time_start, timeout):
                    _err = TimeoutError("Evaluation timed out (%d)", timeout)
                    _err_type = "Timeout"
                if _err is not None:
                    eval_result.error = _err 
                    eval_result.error_type = _err_type
                else:
                    eval_result.result.append(eval_basic_result)

                self._logging_eval_process(eval_result, interval, total_tasks)
                if eval_result.error is not None:
                    break

        _all_eval_time = time.perf_counter() - _all_eval_time_start
        if eval_result.error is None:
            eval_result.score = np.mean([r.log_y_aoc for r in eval_result.result])
            eval_result.total_execution_time = np.sum([r.execution_time for r in eval_result.result])
            _logger.info("Evaluated %s: %.4f executed %.2fs in %.2fs", cls_name, eval_result.score, eval_result.total_execution_time, _all_eval_time)
        else:                           
            _logger.error("Evaluated %s: Failed in %.2fs", cls_name, _all_eval_time)
            eval_result.score = 0.0
            eval_result.total_execution_time = 0.0

        return eval_result

    @classmethod
    def evaluate_from_cls(cls, bo_cls_list:list, problems:list[int]=None, dim:int = 5, budget:int = 40, repeat:int=1, instances:list[list[int]]=None) -> list[EvaluatorResult]:
        res_list = []
        for bo_cls in bo_cls_list:
            evaluator = cls(dim=dim, budget=budget, problems=problems, repeat=repeat, instances=instances)
            evaluator.ignore_over_budget = True
            evaluator.inject_critic = True
            cls_call_kwargs = {
                "capture_output": False,
            }
            res = evaluator.evaluate(None, bo_cls.__name__, cls=bo_cls, cls_call_kwargs=cls_call_kwargs)
            res_list.append(res)
        return res_list
