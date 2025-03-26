from typing import Any
import io
import contextlib
import sys
import traceback
import re
import inspect
import logging
from abc import ABC 
import torch

class ExecInjector(ABC):
    @classmethod
    def inject_code_with_device(cls, code, device) -> str:
        # search whether the code include "cuda"
        if torch.cuda.is_available():
            if "cuda" not in code:
                raise Exception("CUDA is available but the code does not use 'cuda'.")
            else:
                if device is not None and device not in code:
                    code = code.replace("\"cuda\"", f"\"{device}\"")
                    logging.info("replaced 'cuda' with '%s'", device)
        return code

    def inject_cls(self, cls, code) -> str:
        return code

    def inject_instance(self, cls_instance, code, init_kwargs, call_kwargs) -> Any:
        return cls_instance

    def clear(self):
        pass

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

def __default_exec(code, cls_name, cls=None, init_kwargs=None, call_kwargs=None, injector:ExecInjector=None) -> tuple[any, str, str, any]:
    captured_output = io.StringIO()
    res = None
    err = None

    if init_kwargs is None:
        init_kwargs = {}
    if call_kwargs is None:
        call_kwargs = {}

    def _inject_and_init(cls, code, init_kwargs, call_kwargs):
        if injector:
            if code is None:
                try:
                    code = inspect.getsource(cls)
                except Exception:
                    pass

            injector.inject_cls(cls, code)
            cls_instance = cls(**init_kwargs)
            injector.inject_instance(cls_instance, code, init_kwargs, call_kwargs)
        else:
            cls_instance = cls(**init_kwargs)
        return cls_instance

    if cls is not None:
        # helper for debugging
        cls_instance = _inject_and_init(cls, code, init_kwargs, call_kwargs)

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
                    cls_instance = _inject_and_init(_cls, code, init_kwargs, call_kwargs)
                    res = cls_instance(**call_kwargs)
        except Exception as e:
            formatted_traceback = format_track_exec_with_code(cls_name, code, sys.exc_info())
            err = TrackExecExceptionWrapper(e, formatted_traceback)

    if injector is not None:
        injector.clear()

    return res, captured_output.getvalue(), err, injector

def default_exec(code, cls_name, init_kwargs=None, call_kwargs=None, cls=None, injector:ExecInjector=None) -> tuple[any, str, str, any]:
    params = {
            "code": code,
            "cls_name": cls_name,
            "cls": cls,
            "init_kwargs": init_kwargs,
            "call_kwargs": call_kwargs,
            "injector": injector
    }
    return __default_exec(**params)