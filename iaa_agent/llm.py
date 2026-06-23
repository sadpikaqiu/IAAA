from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .models import Intention


class DeepSeekClient:
    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat_json(self, messages: list[dict[str, str]], max_tokens: int = 900) -> dict[str, Any] | None:
        if not self.api_key:
            return None
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
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
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            return None
        try:
            return json.loads(_extract_json(content))
        except json.JSONDecodeError:
            return None


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(line for line in cleaned.splitlines() if not line.strip().startswith("```"))
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def parse_intention_or_none(data: dict[str, Any] | None) -> Intention | None:
    if not data:
        return None
    try:
        return Intention.model_validate(data)
    except Exception:
        return None

