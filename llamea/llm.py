"""LLM manager to connect to different types of models.
"""
import time
import os
from datetime import datetime
from abc import ABC, abstractmethod
import logging
from typing import Optional, Any
from collections.abc import Callable
import requests
import openai
import aisuite as ai
from aisuite.providers.groq_provider import GroqMessageConverter
from google import genai
from google.genai import types


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL_FOR_OPENAI_CLIENT = os.getenv("GROQ_BASE_URL_FOR_OPENAI_CLIENT", None)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
ONEHUB_API_KEY = os.getenv("ONEHUB_API_KEY")
ONEHUB_BASE_URL = os.getenv("ONEHUB_BASE_URL")

LLMS = {
    "mixtral-8x7b-32768": ('mixtral-8x7b-32768', GROQ_API_KEY, GROQ_BASE_URL_FOR_OPENAI_CLIENT, 2, 'groq'),
    'llama3-70b-8192': ('llama3-70b-8192', GROQ_API_KEY, GROQ_BASE_URL_FOR_OPENAI_CLIENT, 2, 'groq'),
    "llama-3.3-70b-versatile": ('llama-3.3-70b-versatile', GROQ_API_KEY, GROQ_BASE_URL_FOR_OPENAI_CLIENT, 15, 'groq'),
    'deepseek-r1-distill-llama-70b': ('deepseek-r1-distill-llama-70b', GROQ_API_KEY, GROQ_BASE_URL_FOR_OPENAI_CLIENT, 5, 'groq'),
    'deepseek-r1-distill-qwen-32b': ('deepseek-r1-distill-qwen-32b', GROQ_API_KEY, GROQ_BASE_URL_FOR_OPENAI_CLIENT, 5, 'groq'),

    "gemini-1.5-flash": ('gemini-1.5-flash', GEMINI_API_KEY, GEMINI_BASE_URL, 5, 'google'),
    "gemini-2.0-flash-exp": ('gemini-2.0-flash-exp', GEMINI_API_KEY, GEMINI_BASE_URL, 10, 'google'),
    'gemini-exp-1206': ('gemini-exp-1206', GEMINI_API_KEY, GEMINI_BASE_URL, 5, 'google'),
    'gemini-2.0-pro' : ('gemini-2.0-pro-exp-02-05', GEMINI_API_KEY, GEMINI_BASE_URL, 5, 'google'),
    'gemini-2.0-flash-thinking': ('gemini-2.0-flash-thinking-exp-01-21', GEMINI_API_KEY, GEMINI_BASE_URL, 5, 'google'),

    'onehub-gemini-2.0-flash': ('gemini-2.0-flash-exp', ONEHUB_API_KEY, ONEHUB_BASE_URL, 5, None),
    'onehub-gemma2-9b-it': ('gemma2-9b-it', ONEHUB_API_KEY, ONEHUB_BASE_URL, 5, None),

    'deepseek/deepseek-chat': ('deepseek/deepseek-chat', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 2, 'openrouter'),
    'o_gemini-flash-1.5-8b-exp': ('google/gemini-flash-1.5-8b-exp', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 5, 'openrouter'),
    'o_gemini-flash-1.5-exp': ('google/gemini-flash-1.5-exp', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 5, 'openrouter'),
    'o_gemini-2.0-flash-exp': ('google/gemini-2.0-flash-exp:free', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 5, 'openrouter'),
    'o_gemini-exp-1206': ('google/gemini-exp-1206:free', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 5, 'openrouter'),
    'o_gemini-exp-1121': ('google/gemini-exp-1121:free', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 5),
    'o_gemini-2.0-flash-thinking-exp': ('google/gemini-2.0-flash-thinking-exp:free', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 5, 'openrouter'),
    'o_llama-3.1-405b-instruct': ('meta-llama/llama-3.1-405b-instruct:free', OPENROUTER_API_KEY, OPENROUTER_BASE_URL, 5, 'openrouter'),
}

class LLMClientResponse:
    def __init__(self, response):
        self.response = response
        self.text = None
        self.prompt_token_count = 0
        self.response_token_count = 0
        self.error = None

    def __str__(self):
        if self.error is not None:
            return str(self.error)
        return str(self.response)

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

class GoogleGenAIClient(LLMClient):
    def __init__(self, api_key: str, model_name: str, base_url: Optional[str] = None):
        super().__init__(api_key, model_name, base_url)
        self.client = genai.Client(api_key=api_key)

    def raw_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Generate a raw completion using Google's GenAI API."""
        try:
            sys_prompt = None
            user_contents = []
            for _msg in messages:
                if _msg['role'] == 'system':
                    sys_prompt = _msg['content']
                else:
                    user_contents.append(types.Part.from_text(text=_msg['content']))
                    user_contents.append(_msg['content'])
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents= user_contents,
                config=types.GenerateContentConfig(
                    system_instruction=sys_prompt,
                    **kwargs
                ),
            )   
            res = LLMClientResponse(response)
            res.text = response.text
            res.prompt_token_count = response.usage_metadata.prompt_token_count
            res.response_token_count = response.usage_metadata.candidates_token_count
            return res
        except Exception as e:
            res = LLMClientResponse(None)
            res.error = e
            return res

class CustomGroqMessageConverter(GroqMessageConverter):
    @staticmethod
    def convert_response(response_data):

        res = GroqMessageConverter.convert_response(response_data)

        current_timestamp = time.time()
        setattr(res, "_response_timestamp", current_timestamp)

        AISuiteClient.raw_response_map[current_timestamp] = response_data

        return res
    
class AISuiteClient(LLMClient):
    # content should be deleted after use
    raw_response_map = {}
    
    def __init__(self, model_key: str):
        if model_key not in LLMS:
            raise ValueError(f"Invalid model key: {model_key}")
        _model = LLMS[model_key]
        api_key = _model[1]
        model_name = _model[0]
        base_url = _model[2]

        self.provider = _model[4]
        
        super().__init__(api_key, model_name, base_url)
        _config = {
            "groq": {
                "api_key": GROQ_API_KEY,
            },
        }
        
        self.client = ai.Client(provider_configs=_config)

        if self.provider == "groq":
            provider = self.client.providers.get(self.provider)
            provider.transformer = CustomGroqMessageConverter()

    def raw_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Generate a raw completion using AISuite's API."""
        res = None
        try:
            _aisuite_model = f"{self.provider}:{self.model_name}"
            response = self.client.chat.completions.create(
                model=_aisuite_model,
                messages=messages,
                **kwargs
            )

            res = LLMClientResponse(response)
            res.text = response.choices[0].message.content

            raw_response = None
            if getattr(response, "_response_timestamp", None) is not None:
                _response_timestamp = getattr(response, "_response_timestamp")
                raw_response = AISuiteClient.raw_response_map.pop(_response_timestamp, None)
                res.response = raw_response
                if self.provider == "groq":
                    res.prompt_token_count = raw_response["usage"]["prompt_tokens"]
                    res.response_token_count = raw_response["usage"]["completion_tokens"]

            return res
        except Exception as e:
            res.error = e
            return res

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
        res = None
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            res = LLMClientResponse(response)
            res.text = response.choices[0].message.content
            res.prompt_token_count = response.usage.prompt_tokens
            res.response_token_count = response.usage.completion_tokens
            return res
        except Exception as e:
            if hasattr(response, "error"):
                res.error = response.error
            else:
                res.error = e
            return res

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

        res = None
        try:
            json_res = response.json()
            res = LLMClientResponse(json_res)
            res.text = json_res["choices"][0]["message"]["content"]
            if "usage" in json_res:
                res.prompt_token_count = json_res["usage"]["prompt_tokens"]
                res.response_token_count = json_res["usage"]["completion_tokens"]
            return res
        except Exception as e:
            if res is None:
                res = LLMClientResponse(None)
                
            if hasattr(response, "error"):
                res.error = response.error
            else:
                res.error = e
            return res

class LLMmanager:
    def __init__(self, model_key: str):
        if model_key not in LLMS:
            raise ValueError(f"Invalid model key: {model_key}")

        _model = LLMS[model_key]
        api_key = _model[1]
        model_name = _model[0]
        base_url = _model[2]
        client_str = _model[4]

        if client_str is None:
            self.client = OpenAIClient(api_key, model_name, base_url)
        elif client_str == "openrouter":
            self.client = OpenAIClient(api_key, model_name, base_url)
        elif client_str == "request":
            self.client = RequestClient(api_key, model_name, base_url)
        elif client_str == "google":
            self.client = GoogleGenAIClient(api_key, model_name, base_url)
        else:
            self.client = AISuiteClient(model_key)

        self.max_interval = _model[3]
        self.mock_res_provider:Callable[..., str] = None

    def model_name(self) -> str:
        return self.client.name

    def chat(self, session_messages, **kwargs):
        if self.mock_res_provider is not None:
            _content = self.mock_res_provider(session_messages, **kwargs)
            res = LLMClientResponse(None)
            res.text = _content
            res.response_token_count = len(_content.split()) 
            return res
        
        response = self.client.raw_completion(
            session_messages,
            **kwargs
        )

        if response.error is not None:
            logging.error("LLM: %s, %s", self.model_name(), response.error)
        return response
