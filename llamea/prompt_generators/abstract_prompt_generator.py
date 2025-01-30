
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any
from ..evaluator import EvaluatorResult
from ..population import Population  #FIXME: prompt generators should not be coupled with the individual and population classes

class GenerationTask(Enum):
    """Enum class for generation tasks."""
    INITIALIZE_SOLUTION = 0
    FIX_ERRORS = 1
    FIX_ERRORS_FROM_ERROR = 2
    OPTIMIZE_PERFORMANCE = 3

class ResponseHandler(ABC):
    """Abstract base class for response handler."""

    def __init__(self):
        self.code = ""
        self.code_name = ""
        self.raw_response = ""
        self.error = None
        self.error_type = None
        self._eval_result:EvaluatorResult = None

    @property
    def eval_result(self) -> EvaluatorResult:
        return self._eval_result
    
    @eval_result.setter
    def eval_result(self, value:EvaluatorResult):
        if value is not None and value.error is not None:
            self.error = value.error
            self.error_type = value.error_type
        self._eval_result = value

    @abstractmethod
    def extract_response(self, response:str, task:GenerationTask) -> None:
        pass

    def __to_json__(self) -> dict:
        d = {
            "code": self.code,
            "code_name": self.code_name,
            "raw_response": self.raw_response,
            "error": self.error,
            "error_type": self.error_type,
            "eval_result": self.eval_result.__to_json__()
        }
        return d

class ResponseImpReturnChecker(ABC):
    """Abstract base class for response return checkers."""
    @abstractmethod
    def __call__(self, imp_return:tuple) -> str:
        pass

class PromptGenerator(ABC):
    """Abstract base class for prompt generators."""
    def task_description(self, task:GenerationTask, extra:str="") -> str:
        pass

    def task_instruction(self, task:GenerationTask, extra:str="") -> str:
        """explicit COT of the task accomplishment"""

    def code_structure(self, extra:str="") -> str:
        pass

    def response_format(self, task:GenerationTask, extra:str="") -> str:
        pass

    def evaluation_feedback_prompt(self, eval_res:EvaluatorResult, 
                                   other_results:tuple[EvaluatorResult,list[EvaluatorResult]]= None) -> str:
        pass

    @abstractmethod
    def get_prompt(self, task:GenerationTask, problem_desc:str,
                   candidates:list[ResponseHandler]= None,
                   population:Population = None,
                   other_results:tuple[EvaluatorResult,list[EvaluatorResult]]= None,
                   sharedborad=None) -> tuple[str, str]:
        pass

    @abstractmethod
    def get_response_handler(self) -> ResponseHandler:
        pass

    def get_return_checker(self) -> ResponseImpReturnChecker:
        return None

    def update_sharedbroad(self, sharedbroad: Any, handlers:list[ResponseHandler], population:Population) -> None:
        pass

    def get_prompt_sharedbroad(self) -> Any:
        return None
