from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .models import Intention


LIVE_LLM_MODES = {"deepseek", "openai"}


def is_live_llm_mode(mode: str) -> bool:
    return mode in LIVE_LLM_MODES


class DeepSeekClient:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        provider: str = "deepseek",
    ) -> None:
        if provider not in {"deepseek", "openai"}:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        self.provider = provider
        if provider == "openai":
            self.model = model or os.environ.get("OPENAI_MODEL", "Qwen/Qwen3.8-27B-FP8")
            self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
            self.api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        else:
            self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
            self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
            self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.last_usage: dict[str, int] | None = None
        self.last_raw_content: str | None = None  # 调试用:最后一次响应的原始 content
        self.last_reasoning_content: str | None = None
        self.last_finish_reason: str | None = None
        self.last_call_status = "not_called"
        self.last_error_type: str | None = None
        self.usage_totals: dict[str, int] = {
            "prompt_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat_json(self, messages: list[dict[str, str]], max_tokens: int = 900) -> dict[str, Any] | None:
        self.last_usage = None
        self.last_raw_content = None
        self.last_reasoning_content = None
        self.last_finish_reason = None
        self.last_call_status = "not_called"
        self.last_error_type = None
        if not self.api_key:
            self.last_call_status = "missing_api_key"
            return None
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.model.startswith("deepseek"):
            payload["temperature"] = 0
            payload["thinking"] = {"type": "disabled"}
        elif self.model.lower().startswith("qwen"):
            enable_thinking = _env_bool("OPENAI_ENABLE_THINKING", False)
            reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT", "medium")
            if reasoning_effort not in {"low", "medium", "xhigh"}:
                raise ValueError(
                    "OPENAI_REASONING_EFFORT must be low, medium, or xhigh"
                )
            payload["temperature"] = float(os.environ.get("OPENAI_TEMPERATURE", "0"))
            payload["seed"] = int(os.environ.get("OPENAI_SEED", "42"))
            payload["max_tokens"] = int(
                os.environ.get("OPENAI_MAX_TOKENS", str(max_tokens))
            )
            payload["chat_template_kwargs"] = {
                "enable_thinking": enable_thinking,
                "preserve_thinking": False,
            }
            if enable_thinking:
                # Applying a JSON grammar from the first generated token prevents
                # Qwen from completing its think block. The qwen3 reasoning parser
                # separates the unconstrained reasoning from the final JSON.
                payload.pop("response_format", None)
                payload["chat_template_kwargs"]["reasoning_effort"] = reasoning_effort
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw_response = resp.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            self.last_call_status = "request_error"
            self.last_error_type = type(exc).__name__
            return None
        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            self.last_call_status = "invalid_response_json"
            self.last_error_type = type(exc).__name__
            return None
        self._record_usage(data.get("usage"))
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        self.last_finish_reason = choice.get("finish_reason")
        content = message.get("content", "")
        self.last_raw_content = content
        self.last_reasoning_content = message.get("reasoning_content")
        if not content:
            self.last_call_status = (
                "length_truncated"
                if self.last_finish_reason == "length"
                else "empty_content"
            )
            return None
        try:
            parsed = json.loads(_extract_json(content))
        except json.JSONDecodeError as exc:
            self.last_call_status = (
                "length_truncated"
                if self.last_finish_reason == "length"
                else "invalid_content_json"
            )
            self.last_error_type = type(exc).__name__
            return None
        self.last_call_status = "success"
        return parsed

    def _record_usage(self, usage: Any) -> None:
        fields = {
            "prompt_tokens",
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "total_tokens",
        }
        if not isinstance(usage, dict) or not usage:
            self.last_usage = None
            return
        parsed = {field: int(usage.get(field, 0) or 0) for field in fields}
        completion_details = usage.get("completion_tokens_details")
        if isinstance(completion_details, dict):
            parsed["reasoning_tokens"] = int(
                completion_details.get("reasoning_tokens", 0) or 0
            )
        if parsed["total_tokens"] == 0:
            parsed["total_tokens"] = parsed["prompt_tokens"] + parsed["completion_tokens"]
        if not any(parsed.values()):
            self.last_usage = None
            return
        self.last_usage = parsed
        for field, value in parsed.items():
            self.usage_totals[field] = self.usage_totals.get(field, 0) + value


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(line for line in cleaned.splitlines() if not line.strip().startswith("```"))
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_intention_or_none(data: dict[str, Any] | None) -> Intention | None:
    if not data:
        return None
    try:
        return Intention.model_validate(data)
    except Exception:
        return None
