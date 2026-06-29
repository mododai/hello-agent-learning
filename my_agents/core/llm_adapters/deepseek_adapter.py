from typing import Optional, Dict

from my_agents.core.llm_adapters import BaseLLMAdapter, OpenAIAdapter


class DeepSeekerAdapter(OpenAIAdapter):

    def __init__(self,
                 api_key: str,
                 base_url: Optional[str],
                 model: str,
                 timeout: int,
                 ):
        super().__init__(api_key, base_url, model, timeout)
        self.last_stats = None
        self.provider = "DeepSeek"

    def _is_thinking_model(self, **kwargs) -> True:

        # deepseek系列模型, 参考官方api文档(https://api-docs.deepseek.com/zh-cn/)
        if hasattr(kwargs, "extra_body") and hasattr(kwargs["extra_body"], "thinking") and getattr(kwargs["extra_body"], "thinking") == "enable":
            return True
        return super()._is_thinking_model(**kwargs)

    def _get_reasoning_content(self, response) -> str:
        return response.choices[0].message.reasoning_content


    def enable_thinking_model(self, reasoning_effort: Optional[str] = None ,**kwargs) -> Dict[str, str]:
        if self.model.startswith("deepseek"):
            kwargs["extra_body"] = {
                "thinking": {
                    "type": "enabled"
                }
            }
            kwargs["reasoning_effort"] = reasoning_effort if reasoning_effort else "high"

        return kwargs

    def disable_thinking_model(self, **kwargs) -> Dict[str, str]:
        if self.model.startswith("deepseek"):
            kwargs["extra_body"] = {
                "thinking": {
                    "type": "disabled"
                }
            }


        return kwargs