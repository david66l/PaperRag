"""LLM generation providers."""

from abc import ABC, abstractmethod

import httpx

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response given system + user prompts."""
        ...


class OpenAICompatibleProvider(BaseLLMProvider):
    """Works with OpenAI, vLLM, Qwen, DeepSeek, Ollama, etc."""

    def __init__(self, api_url: str, api_key: str, model: str, temperature: float = 0.1, max_tokens: int = 1024):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.api_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        answer = data["choices"][0]["message"]["content"]
        logger.info("LLM generated %d chars", len(answer))
        return answer


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6", temperature: float = 0.1, max_tokens: int = 1024):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        answer = data["content"][0]["text"]
        logger.info("Anthropic LLM generated %d chars", len(answer))
        return answer


def create_llm_provider(settings: Settings) -> BaseLLMProvider:
    if settings.llm_provider == "anthropic":
        return AnthropicProvider(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    return OpenAICompatibleProvider(
        api_url=settings.llm_api_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
