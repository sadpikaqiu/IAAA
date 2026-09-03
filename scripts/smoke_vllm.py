from __future__ import annotations

import json
import os
import urllib.request


base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
model = os.environ.get("OPENAI_MODEL", "Qwen/Qwen3.8-27B-FP8")
payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "Return one JSON object only."},
        {"role": "user", "content": 'Return {"ok": true, "backend": "vllm"}.'},
    ],
    "max_tokens": 64,
    "temperature": 0,
    "seed": 42,
    "response_format": {"type": "json_object"},
    "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
}
request = urllib.request.Request(
    f"{base_url}/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=300) as response:
    result = json.loads(response.read().decode("utf-8"))

content = result["choices"][0]["message"]["content"]
parsed = json.loads(content)
if parsed.get("ok") is not True:
    raise RuntimeError(f"Unexpected model response: {content}")

print(json.dumps({"model": result.get("model"), "content": parsed, "usage": result.get("usage")}, indent=2))
