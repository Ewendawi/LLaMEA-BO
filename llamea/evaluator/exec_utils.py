from typing import Any
import io
import contextlib
import concurrent.futures
import sys
import traceback
import re

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