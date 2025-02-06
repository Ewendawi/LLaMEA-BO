from typing import Any
import io
import contextlib
import concurrent.futures
import sys
import traceback
import re
import inspect
from .injected_critic import AlgorithmCritic, critic_wrapper, set_inject_maximize, get_inject_maximize

class TrackExecExceptionWrapper:
    def __init__(self, error, _traceback):
        self.error = error
        self.traceback = _traceback

    @property
    def error_type(self):
        return type(self.error).__name__

    def __str__(self):
        if self.traceback is None:
            return str(self.error)
        return self.traceback


def track_exec(code_string, name, _globals=None, _locals=None):
    compiled_code = compile(code_string, f'<{name}>', 'exec')
    exec(compiled_code, _globals, _locals)

def format_track_exec_with_code(name, code_str, exc_info, context_lines=2):
    trace_lines = traceback.format_exception(*exc_info)

    # remove local traceback
    new_trace_lines = []
    for line in trace_lines:
        match = re.search(r'File ".*\.py", line (\d+), in', line)
        if not match:
            new_trace_lines.append(line)
    trace_lines = new_trace_lines

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

def __inject_critic_code(code: str) -> str:
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
        elif re.search(r'def\s+_update_sample_points', line):
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

def inject_critic_cls(cls):
    methods = ['_update_sample_points', '_fit_model']
    for method in methods:
        original_method = getattr(cls, method, None)
        if original_method is None or original_method.__name__ == 'injected_wrapper':
            continue
        decorated_method = critic_wrapper(original_method)
        setattr(cls, method, decorated_method)

def inject_critic_func(cls_instance, init_kwargs, call_kwargs) -> any:
    critic = None
    if not hasattr(cls_instance, "_injected_critic"):
        setattr(cls_instance, "_injected_critic", None)
    if cls_instance._injected_critic is None:
        dim = init_kwargs.get("dim", 1)
        func = call_kwargs.get("func", None)
        bounds = func.bounds if func is not None else None
        critic = AlgorithmCritic(dim=dim, bounds=bounds, optimal_value=func.optimal_value, critic_y_range=400)
        critic.update_test_y(func)
        critic.search_result.init_grid(bounds=bounds, dim=dim, budget=func.budget)

        cls_instance._injected_critic = critic

    return critic

def __default_exec(code, cls_name, cls=None, init_kwargs=None, call_kwargs=None, inject_critic=False) -> tuple[any, str, str, any]:
    captured_output = io.StringIO()
    res = None
    err = None

    if init_kwargs is None:
        init_kwargs = {}
    if call_kwargs is None:
        call_kwargs = {}

    def _inject_critic_and_init(cls, init_kwargs, call_kwargs, code=None):
        if inject_critic:
            inject_critic_cls(cls)
            cls_instance = cls(**init_kwargs)
            critic = inject_critic_func(cls_instance, init_kwargs, call_kwargs)

            if code is None:
                try:
                    code = inspect.getsource(cls)
                except:
                    pass

            is_maximize = get_inject_maximize(cls_instance)
            if not is_maximize and 'botorch' in code:
                is_maximize = True
            
            if is_maximize:
                critic.maximize = True
                obj_fn = call_kwargs.get("func", None)
                if obj_fn is not None and hasattr(obj_fn, "maximize"):
                    obj_fn.maximize = True
        else:
            cls_instance = cls(**init_kwargs)
            critic = None
        return cls_instance, critic

    critic = None
    if cls is not None:
        # helper for debugging
        cls_instance, critic = _inject_critic_and_init(cls, init_kwargs, call_kwargs, code)

        should_capture_output = call_kwargs.pop("capture_output", True) 
        if should_capture_output:
            with contextlib.redirect_stderr(captured_output), contextlib.redirect_stdout(captured_output):
                res = cls_instance(**call_kwargs)
        else:
            res = cls_instance(**call_kwargs)
    else:
        try:
            namespace: dict[str, Any] = {}
            track_exec(code, cls_name, namespace)

            if cls_name not in namespace:
                err = NameError(f"No '{cls_name}' found in the generated code")
            else:
                with contextlib.redirect_stderr(captured_output), contextlib.redirect_stdout(captured_output):
                    _cls = namespace[cls_name]
                    cls_instance, critic = _inject_critic_and_init(_cls, init_kwargs, call_kwargs, code)
                    res = cls_instance(**call_kwargs)
        except Exception as e:
            formatted_traceback = format_track_exec_with_code(cls_name, code, sys.exc_info())
            err = TrackExecExceptionWrapper(e, formatted_traceback)

    return res, captured_output.getvalue(), err, critic

def default_exec(code, cls_name, init_kwargs=None, call_kwargs=None, time_out:float=None, cls=None, inject_critic=False):
    params = {
            "code": code,
            "cls_name": cls_name,
            "cls": cls,
            "init_kwargs": init_kwargs,
            "call_kwargs": call_kwargs,
            "inject_critic": inject_critic
    }
    if time_out is None:
        return __default_exec(**params)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(__default_exec, **params)
            done, not_done = concurrent.futures.wait([future], timeout=time_out)
            if done:
                return future.result()
            if not_done:
                _err = TimeoutError("Evaluation timed out")
                err = TrackExecExceptionWrapper(_err, None)
                future.cancel()
                return None, None, err, None