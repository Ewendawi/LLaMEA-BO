import logging
import time
import functools
import inspect
import re
from types import FunctionType
import numpy as np
import torch
import gpytorch
import botorch
import sklearn
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import qmc
from .evaluator_result import EvaluatorSearchResult
from .exec_utils import ExecInjector

class FunctionProfiler:
    def __init__(self):
        self.name = None
        self._execution_times = {}
        self._call_counts = {}

    def profile(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time

            func_name = func.__name__
            if func_name not in self._execution_times:
                self._execution_times[func_name] = []
                self._call_counts[func_name] = 0

            self._execution_times[func_name].append(execution_time)
            self._call_counts[func_name] += 1
            return result
        return wrapper

    def wrap_class(self, cls):
        for name, attr in cls.__dict__.items():
            if isinstance(attr, FunctionType): 
                setattr(cls, name, self.profile(attr)) 
        return cls

    def print_report(self):
        print(f"\n--- {self.name} Profiling Report ---")
        for func_name, times in self._execution_times.items():
            call_count = self._call_counts[func_name]
            total_time = sum(times)
            avg_time = total_time / call_count if call_count > 0 else 0

            print(f"Fun: {func_name}")
            print(f"  Call Count: {call_count}")
            print(f"  Total Time: {total_time:.4f} seconds")
            print(f"  Ave Time: {avg_time:.4f} seconds")
        print("--- End of Report ---")

def critic_wrapper(func):
    functools.wraps(func)

    def to_numpy_if_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x
    
    def injected_wrapper(self, *args, **kwargs):
        _injected_critic = None
        if hasattr(self, "_injected_critic"):
            _injected_critic = getattr(self, "_injected_critic")
            if _injected_critic.n_init == 0 and hasattr(self, "n_init"):
                _injected_critic.n_init = self.n_init

        if _injected_critic is not None and not _injected_critic.ignore_metric:
            if func.__name__ == "_update_eval_points":
                try:
                    next_X = args[0]
                    next_y = args[1]
                    next_X = to_numpy_if_tensor(next_X)
                    next_y = to_numpy_if_tensor(next_y)
                    X = to_numpy_if_tensor(self.X)
                    y = to_numpy_if_tensor(self.y)

                    n_evals = None
                    if hasattr(self, "n_evals"):
                        n_evals = self.n_evals
                    _injected_critic.update_after_eval(X, y, next_X, next_y, n_evals)
                except Exception as e:
                    logging.error("Error in _update_eval_points wrapper: %s", e)
            elif func.__name__ == "_fit_model":
                n_evals = None
                if hasattr(self, "n_evals"):
                    n_evals = self.n_evals
                if hasattr(self, "kappa"):
                    kappa = self.kappa
                    if n_evals and len(_injected_critic.search_result.kappa_list) < n_evals:
                        len_diff = n_evals - len(_injected_critic.search_result.kappa_list)
                        _injected_critic.search_result.kappa_list.extend([kappa] * len_diff)
                    _injected_critic.search_result.kappa_list.append(kappa)

                if hasattr(self, "trust_region_radius"):
                    trust_region_radius = self.trust_region_radius
                    if n_evals and len(_injected_critic.trust_region_radius_list.kappa_list) < n_evals:
                        len_diff = n_evals - len(_injected_critic.search_result.trust_region_radius_list)
                        _injected_critic.search_result.trust_region_radius_list.extend([trust_region_radius] * len_diff)
                    _injected_critic.search_result.trust_region_radius_list.append(trust_region_radius)

        res = func(self, *args, **kwargs)

        if _injected_critic is not None and not _injected_critic.ignore_metric:
            if func.__name__ == "_fit_model":
                try:
                    new_X = args[0]
                    new_y = args[1]
                    new_X = to_numpy_if_tensor(new_X)
                    new_y = to_numpy_if_tensor(new_y)
                    n_evals = len(new_X)
                    model = res
                    if hasattr(self, "n_evals"):
                        n_evals = self.n_evals
                    _injected_critic.update_after_model_fit(model, n_evals, new_X, new_y)
                except Exception as e:
                    logging.error("Error in _fit_model wrapper: %s", e)
        return res
    return injected_wrapper

def set_inject_maximize(cls_instance, maximize):
    if cls_instance is None:
        return
    if hasattr(cls_instance, "_inject_maximize"):
        setattr(cls_instance, "_inject_maximize", maximize)
    else:
        setattr(cls_instance, "_inject_maximize", maximize)

def get_inject_maximize(cls_instance):
    if cls_instance is None:
        return False
    if hasattr(cls_instance, "is_maximization"):
        return cls_instance.is_maximization()
    if hasattr(cls_instance, "_inject_maximize"):
        return getattr(cls_instance, "_inject_maximize")
    return False

class BOInjector(ExecInjector):
    def __init__(self):
        self.ignore_metric = True
        self.critic = None

    def inject_cls(self, cls, code):
        methods = ['_update_eval_points', '_fit_model']
        for method in methods:
            original_method = getattr(cls, method, None)
            if original_method is None or original_method.__name__ == 'injected_wrapper':
                continue
            decorated_method = critic_wrapper(original_method)
            setattr(cls, method, decorated_method)
        return code

    def inject_instance(self, cls_instance, code, init_kwargs, call_kwargs):
        if not hasattr(cls_instance, "_injected_critic"):
            dim = init_kwargs.get("dim", 1)
            func = call_kwargs.get("func", None)
            bounds = func.bounds if func is not None else None
            critic = AlgorithmCritic(dim=dim, bounds=bounds, optimal_value=func.optimal_value, critic_y_range=400)
            critic.update_test_y(func)
            critic.search_result.init_grid(bounds=bounds, dim=dim, budget=func.budget)
            critic.ignore_metric = self.ignore_metric

            setattr(cls_instance, "_injected_critic", critic)

            is_maximize = get_inject_maximize(cls_instance)
            if not is_maximize and code is not None and 'botorch' in code:
                is_maximize = True
            
            if is_maximize:
                critic.maximize = True
                obj_fn = call_kwargs.get("func", None)
                if obj_fn is not None and hasattr(obj_fn, "maximize"):
                    obj_fn.maximize = True

            self.critic = critic

    def inject_code(self, code: str) -> str:
        # Add the critic_wrapper function to the code
        critic_wrapper_code = inspect.getsource(critic_wrapper)
        critic_wrapper_code_lines = critic_wrapper_code.splitlines(keepends=True)

        lines = code.splitlines(keepends=True)
        new_lines = []

        for line in lines:
            # find the first class in the code. then inject the critic_wrapper function above it
            if re.search(r'^class\s+(\w+)', line):
                new_lines.extend(critic_wrapper_code_lines)
                new_lines.append("\n")
            elif re.search(r'def\s+_update_eval_points', line):
                stripped_text   = line.lstrip()
                n_blank_spaces  = len(line) - len(stripped_text)
                decrator_line = " " * n_blank_spaces + '@critic_wrapper'
                new_lines.append(decrator_line)
                new_lines.append('\n')
            elif re.search(r'def\s+_fit_model', line):
                stripped_text   = line.lstrip()
                n_blank_spaces  = len(line) - len(stripped_text)
                decrator_line = " " * n_blank_spaces + '@critic_wrapper'
                new_lines.append(decrator_line)
                new_lines.append('\n')

            new_lines.append(line)

        return "".join(new_lines)

    def clear(self):
        if self.critic is not None:
            self.critic.clear()

class AlgorithmCritic:
    # - r2 from surrogate model
    # - uncertainty from surrogate model of the same samples
    def __init__(self, dim:int, bounds:np.ndarray, optimal_value=None, critic_y_range=None):
        self.dim = dim
        self.bounds = bounds
        self.n_init = 0
        self.maximize = False
        self.ignore_metric = False
        
        self.n_test_x = 1000
        self.test_x = None
        self.test_y = None

        self.r_2_list = []
        self.r_2_list_on_train = []
        self.uncertainty_list = []
        self.uncertainty_list_on_train = []

        self.search_result = EvaluatorSearchResult()
        self.search_result.init_acq_score(optimal_value=optimal_value, y_range=critic_y_range)

        self.temp_r2_list = []
        self.temp_r2_list_on_train = []
        self.temp_uncertainty_list = []
        self.temp_uncertainty_list_on_train = []

    def _sample_points(self, n_points) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def clear(self):
        self.test_x = None
        self.test_y = None

    def update_test_y(self, func):
        self.test_x = self._sample_points(self.n_test_x)
        self.test_y = func.stateless_call(self.test_x)

    def update_after_eval(self, x, y, next_x, next_y, n_evals):
        if self.ignore_metric:
            return

        # n_evals should include the evaluation of next_x
        
        # inverse the y to treat the problem as minimization
        if self.maximize:
            y = -y if y is not None else None
            next_y = -next_y if next_y is not None else None
        
        self.search_result.update_next_grid_coverage(X=x, next_X=next_x, bounds=self.bounds, n_evals=n_evals)
        self.search_result.update_next_dbscan_coverage(X=x, next_X=next_x, bounds=self.bounds, n_evals=n_evals)
        self.search_result.update_next_exploitation(X=x, next_X=next_x, fX=y, next_fX=next_y, n_evals=n_evals)
        self.search_result.update_next_acq_score(fX=y, next_fX=next_y, n_evals=n_evals)

# model related
    def update_after_model_fit_temp(self, model, new_X, new_y):
        if self.ignore_metric:
            return
        r2 = self._get_model_r2(model)
        self.temp_r2_list.append(r2)
        r2_on_train = self._get_model_r2(model, new_X, new_y)
        self.temp_r2_list_on_train.append(r2_on_train)
        
        uncertainty = self._get_model_uncertainty(model)
        self.temp_uncertainty_list.append(uncertainty)
        uncertainty_on_train = self._get_model_uncertainty(model, new_X)
        self.temp_uncertainty_list_on_train.append(uncertainty_on_train)

    def update_after_model_fit_with_temp(self, n_evals):
        if self.ignore_metric:
            return
        mean_r2 = np.mean(self.temp_r2_list)
        self._update_r2(mean_r2, n_evals, self.r_2_list)
        self.temp_r2_list = []

        mean_r2_on_train = np.mean(self.temp_r2_list_on_train)
        self._update_r2(mean_r2_on_train, n_evals, self.r_2_list_on_train)
        self.temp_r2_list_on_train = []
        
        mean_uncertainty = np.mean(self.temp_uncertainty_list)
        self._update_uncertainty(mean_uncertainty, n_evals, self.uncertainty_list)
        self.temp_uncertainty_list = []

        mean_uncertainty_on_train = np.mean(self.temp_uncertainty_list_on_train)
        self._update_uncertainty(mean_uncertainty_on_train, n_evals, self.uncertainty_list_on_train)
        self.temp_uncertainty_list_on_train = []

    def update_after_model_fit(self, model, n_evals, new_X, new_y):
        if self.ignore_metric:
            return

        r_squared_on_train = self._get_model_r2(model, new_X, new_y)
        self._update_r2(r_squared_on_train, n_evals, self.r_2_list_on_train)
        r_squared = self._get_model_r2(model)
        self._update_r2(r_squared, n_evals, self.r_2_list)

        uncertainty = self._get_model_uncertainty(model)
        self._update_uncertainty(uncertainty, n_evals, self.uncertainty_list)
        uncertainty_on_train = self._get_model_uncertainty(model, new_X)
        self._update_uncertainty(uncertainty_on_train, n_evals, self.uncertainty_list_on_train)

# r squared 
    def _get_model_r2(self, model, new_X=None, new_y=None):
        def _get_single_model_r2(model, x, y):
            _r_squared = 0.0
            if isinstance(model, sklearn.base.RegressorMixin):
                _r_squared = model.score(x, y)
            # elif isinstance(model, botorch.models.SingleTaskGP):
            #     with torch.no_grad():
            #         tensor_x = torch.tensor(x, dtype=torch.float32)
            #         mean = model.posterior(tensor_x).mean.detach().numpy()
            #         _r_squared = r2_score(y, mean)
            elif isinstance(model, gpytorch.models.GP):
                model.eval()
                model.likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    tensor_x = torch.tensor(x, dtype=torch.float32)
                    posterior = model.likelihood(model(tensor_x))
                    mean = posterior.mean.detach().numpy()
                    _r_squared = r2_score(y, mean)
            return _r_squared

        eval_x = new_X
        eval_y = new_y
        if eval_x is None or eval_y is None:
            eval_x = self.test_x
            eval_y = self.test_y 

        r_squared = 0.0
        if isinstance(model, list) or isinstance(model, tuple):
            r_squared_list = [_get_single_model_r2(m, eval_x, eval_y) for m in model]
            r_squared = np.mean(r_squared_list)
        else:
            r_squared = _get_single_model_r2(model, eval_x, eval_y)
        return r_squared

    def _update_r2(self, r_squared, n_evals, target_list):
        if len(target_list) > 0:
            n_new_points = n_evals - len(target_list)
        else:
            n_new_points = n_evals

        n_fill = n_new_points - 1
        target_list.extend([np.nan] * n_fill)
        target_list.append(r_squared)

# uncertainty
    def _get_model_uncertainty(self, model, new_X=None):
        def _get_single_model_uncertainty(model, x):
            _uncertainty = 0.0
            # if isinstance(model, botorch.models.SingleTaskGP):
            #     with torch.no_grad():
            #         x = torch.tensor(x, dtype=torch.float32)
            #         _uncertainty = model.posterior(x).variance.detach().numpy()
            if isinstance(model, gpytorch.models.GP):
                model.eval()
                model.likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    x = torch.tensor(x, dtype=torch.float32)
                    posterior = model.likelihood(model(x))
                    _variance = posterior.variance.cpu().numpy()
                    _uncertainty = np.sqrt(_variance)
            elif isinstance(model, GaussianProcessRegressor):
                _, _uncertainty = model.predict(x, return_std=True)
            mean_uncertainty = np.mean(_uncertainty)
            return mean_uncertainty
        
        eval_x = new_X
        if eval_x is None:
            eval_x = self.test_x
        
        uncertainty = 0.0
        if isinstance(model, list):
            uncertainty_list = [_get_single_model_uncertainty(m, eval_x) for m in model]
            uncertainty = np.mean(uncertainty_list)
        else:
            uncertainty = _get_single_model_uncertainty(model, eval_x)
        return uncertainty

    def _update_uncertainty(self, uncertainty, n_evals, target_list=None):
        if len(target_list) > 0:
            n_new_points = n_evals - len(target_list)
        else:
            n_new_points = n_evals
        n_fill = n_new_points - 1
        target_list.extend([np.nan] * n_fill)
        target_list.append(uncertainty)
