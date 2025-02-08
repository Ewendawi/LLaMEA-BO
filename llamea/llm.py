"""LLM manager to connect to different types of models.
"""
import time
import os
import sys
from abc import ABC, abstractmethod
import logging
from typing import Optional, Any
from collections.abc import Callable
import requests
import openai

#TODO: aisuite?
class LLMClient(ABC):
    """Abstract base class for LLM backends."""
    def __init__(self, api_key: str, model_name: str, base_url: Optional[str] = None):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key

    @abstractmethod
    def raw_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Generate a raw completion from the LLM.
        """

    @property
    def name(self) -> str:
        """Return the name of the LLM backend."""
        return f"{self.model_name}"

class OpenAIClient(LLMClient):
    """OpenAI GPT backend implementation."""

    def __init__(self, api_key: str, model_name: str, base_url: Optional[str] = None):
        super().__init__(api_key, model_name, base_url)
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def raw_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Generate a raw completion using OpenAI's API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            return response
        except Exception as e:
            return e

class RequestClient(LLMClient):
    def raw_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        url = self.base_url + "/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            **kwargs
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.post(url, json=payload, headers=headers, timeout=30)

        return response.json()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
ONEHUB_API_KEY = os.getenv("ONEHUB_API_KEY")
ONEHUB_BASE_URL = os.getenv("ONEHUB_BASE_URL")

LLMS = {
    "mixtral-8x7b-32768": ('mixtral-8x7b-32768', GROQ_API_KEY, GROQ_BASE_URL, 2),
    "llama-3.1-70b-versatile": ('llama-3.1-70b-versatile', GROQ_API_KEY, GROQ_BASE_URL, 6),
    "llama-3.3-70b-versatile": ('llama-3.3-70b-versatile', GROQ_API_KEY, GROQ_BASE_URL, 15),

    "gemini-1.5-flash": ('gemini-1.5-flash', GEMINI_API_KEY, GEMINI_BASE_URL, 5),
    "gemini-2.0-flash-exp": ('gemini-2.0-flash-exp', GEMINI_API_KEY, GEMINI_BASE_URL, 10),
    'gemini-exp-1206': ('gemini-exp-1206', GEMINI_API_KEY, GEMINI_BASE_URL, 5),
    'gemini-2.0-pro' : ('gemini-2.0-pro-exp-02-05', GEMINI_API_KEY, GEMINI_BASE_URL, 5),
    'gemini-2.0-flash-thinking': ('gemini-2.0-flash-thinking-exp-01-21', GEMINI_API_KEY, GEMINI_BASE_URL, 5),

    'onehub-gemini-2.0-flash': ('gemini-2.0-flash-exp', ONEHUB_API_KEY, ONEHUB_BASE_URL, 5),
    'onehub-gemma2-9b-it': ('gemma2-9b-it', ONEHUB_API_KEY, ONEHUB_BASE_URL, 5),

    'deepseek/deepseek-chat': ('deepseek/deepseek-chat', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 2),
    'o_gemini-flash-1.5-8b-exp': ('google/gemini-flash-1.5-8b-exp', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 5),
    'o_gemini-flash-1.5-exp': ('google/gemini-flash-1.5-exp', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 5),
    'o_gemini-2.0-flash-exp': ('google/gemini-2.0-flash-exp:free', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 5),
    'o_gemini-exp-1206': ('google/gemini-exp-1206:free', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 5),
    'o_gemini-exp-1121': ('google/gemini-exp-1121:free', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 5),
    'o_gemini-2.0-flash-thinking-exp': ('google/gemini-2.0-flash-thinking-exp:free', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 5),
    'o_llama-3.1-405b-instruct': ('meta-llama/llama-3.1-405b-instruct:free', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 5),
}

class LLMmanager:
    """LLM manager, currently only supports ChatGPT models."""

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None, client_str: Optional[str]=None, max_interval: int = 0):
        """Initialize the LLM manager with an api key and model name.

        Args:
            api_key (str): api key for authentication.
            model (str, optional): model abbreviation. Defaults to "gpt-4-turbo".
                Options are: gpt-3.5-turbo, gpt-4-turbo, gpt-4o, llama3, codellama, deepseek-coder-v2, gemma2, codegemma,
        """
        if client_str is not None:
            cls = globals()[client_str]
            if issubclass(cls, LLMClient):
                self.client = cls(api_key, model, base_url)
            else:
                raise ValueError(f"Invalid client string: {client_str}")
        else:
            self.client = OpenAIClient(api_key, model, base_url)

        self.max_interval = max_interval

        self.mock_res_provider:Callable[..., str] = None

    def model_name(self) -> str:
        return self.client.name

    def chat(self, session_messages, temperature=0.7, **kwargs):
        if self.mock_res_provider is not None:
            return self.mock_res_provider(session_messages, temperature, **kwargs)
        
        response = self.client.raw_completion(
            session_messages,
            temperature=temperature,
            **kwargs
        )
        try:
            content = response.choices[0].message.content
            return content
        except Exception:
            if hasattr(response, "error"):
                logging.error("LLM: %s, %s", self.model_name(), response.error)
            else:
                logging.error("LLM: %s, %s", self.model_name(), response)
            return ""
