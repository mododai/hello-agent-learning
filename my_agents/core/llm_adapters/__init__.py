"""
LLM 适配器包，可以根据需求进行扩展

"""
from .base_adapter import BaseLLMAdapter
from .openai_adapter import OpenAIAdapter

from .llm_response import *
