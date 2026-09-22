"""Per-request budgets, durable LLM attempts, and exact-token prompt packing."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from pydantic import ValidationError

from .llm import DeepSeekClient, _extract_json


@dataclass
class RepairRequest:
    messages: list
    schema: dict
    validator: object
    context: str
    metadata: dict


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def strict_response_json(content):
    """Reject repeated object keys before JSON parsing can silently discard them."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON object key: {key}. Each key may occur only once.")
            result[key] = value
        return result
    if not isinstance(content, str):
        raise ValueError("Structured response has no raw JSON text")
    return json.loads(_extract_json(content), object_pairs_hook=unique_object)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


class PromptBudget:
    def __init__(self, tokenizer, limit=11500):
        self.tokenizer, self.limit = tokenizer, limit

    def count(self, messages):
        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False,
                    preserve_thinking=False, return_dict=False)
        # Newer Transformers may return BatchEncoding by default: len(mapping)
        # counts fields (usually 2), not tokens. Also tolerate that return shape
        # from adapters which ignore return_dict=False.
        ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        if not isinstance(ids, (list, tuple)):
            raise ValueError("Expected token IDs for one chat prompt")
        if ids and isinstance(ids[0], (list, tuple)):
            if len(ids) != 1:
                raise ValueError("Prompt budget expects exactly one conversation")
            ids = ids[0]
        return len(ids)

    def clip(self, text, limit):
        tokens = self.tokenizer.encode(str(text), add_special_tokens=False)
        return self.tokenizer.decode(tokens[:limit], skip_special_tokens=True)

    def messages(self, system, payload):
        """Only shorten excerpts. Candidate IDs and mandatory facts are never dropped."""
        payload = copy.deepcopy(payload)
        def trim(value, cap):
            if isinstance(value, dict):
                for key, item in list(value.items()):
                    if key == "text" and isinstance(item, str):
                        clipped = self.clip(item, cap)
                        value[key] = clipped
                        value["excerpt_truncated"] = value.get("excerpt_truncated", False) or clipped != item
                    else:
                        trim(item, cap)
            elif isinstance(value, list):
                for item in value:
                    trim(item, cap)
        for cap in (80, 40, 20, 8, 0):
            trim(payload, cap)
            messages = [{"role": "system", "content": system}, {"role": "user", "content": canonical(payload)}]
            # Reserve room for one bounded repair instruction within the same input cap.
            if self.count(messages) <= self.limit - min(1024, self.limit // 2):
                return messages
        raise ValueError("context_budget_exceeded: candidate identifiers/facts cannot fit")


class JournaledModel:
    def __init__(self, directory, identity, budget: PromptBudget, config, *, client=None, heartbeat=None):
        self.directory = Path(directory)
        self.identity, self.budget, self.config = identity, budget, config
        self.client = client or DeepSeekClient(provider="openai", timeout_seconds=config.request_timeout)
        self.heartbeat = heartbeat or (lambda *_: None)
        self.started = time.monotonic()
        self.used_labels: list[str] = []

    def _records(self):
        return [read_json(p) for p in sorted(self.directory.glob("*.json"))]

    def accounting(self):
        records = self._records()
        attempts = [a for r in records for a in r["attempts"]]
        totals = {}
        for a in attempts:
            for k, v in (a.get("usage") or {}).items():
                totals[k] = totals.get(k, 0) + int(v)
        return {"logical_calls": len(records), "requests": len(attempts),
                "retries": sum(max(0, len(r["attempts"]) - 1) for r in records),
                "invalid_attempts": sum(not a.get("accepted", False) for a in attempts),
                "usage_missing_count": sum(not a.get("usage") for a in attempts),
                "usage": totals, "model_seconds": sum(a.get("elapsed_seconds", 0) for a in attempts)}

    def repair_messages(self, messages, attempts, max_tokens, *, structured):
        previous = attempts[-1]
        limit = min(16384 - max_tokens, self.budget.limit) if structured else 16384 - max_tokens
        diagnostic = self.budget.clip(previous.get("error", "Previous request was interrupted"), 300)
        feedback = (f"Repair attempt {len(attempts)}. The previous response was rejected. "
                    "Return a complete corrected JSON response, using the original facts and allowed IDs. "
                    "Meet the required number of DISTINCT POIs; never pad a list by repeating an ID. "
                    "Choose any additional POIs yourself from the supplied candidates. "
                    "Validation diagnostic: " + diagnostic)
        parsed = previous.get("parsed")
        if parsed is not None:
            candidate = messages + [{"role": "assistant", "content": canonical(parsed)},
                                    {"role": "user", "content": feedback}]
            if self.budget.count(candidate) <= limit:
                return candidate, "full_previous_response"
        projection = {}
        if isinstance(parsed, dict):
            if "working_poi_selection" in parsed:
                projection["working_poi_selection"] = parsed["working_poi_selection"]
            if "working_poi_ids" in parsed:
                projection["working_poi_ids"] = parsed["working_poi_ids"]
            if isinstance(parsed.get("ranked_pois"), list):
                projection["ranked_pois"] = [{k: p.get(k) for k in ("poi_idx", "evidence_refs")}
                                             for p in parsed["ranked_pois"] if isinstance(p, dict)]
            if isinstance(parsed.get("ranked_pois_by_id"), dict):
                projection["ranked_pois_by_id"] = {idx: {k: p.get(k) for k in ("rank", "evidence_refs")}
                    for idx, p in parsed["ranked_pois_by_id"].items() if isinstance(p, dict)}
        for cap in (640, 320, 160, 0):
            previous_text = ("\nPrevious invalid answer, identifier summary: " +
                             self.budget.clip(canonical(projection), cap)) if projection and cap else ""
            candidate = messages + [{"role": "user", "content": feedback + previous_text}]
            if self.budget.count(candidate) <= limit:
                return candidate, "identifier_summary" if previous_text else "diagnostic_only"
        raise ValueError("context_budget_exceeded: cannot fit repair feedback without dropping original facts")

    def call(self, label, messages, max_tokens, validator, *, schema=None, repair_builder=None):
        options = {"max_tokens": max_tokens, "temperature": 0, "seed": 42,
                   "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False}}
        if schema is not None:
            options["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "iaaa_" + label, "schema": schema, "strict": True}}
        identity = {"experiment": self.identity, "model": self.client.model,
                    "base_url": self.client.base_url, "messages": digest(messages), "options": options}
        if repair_builder is not None:
            identity["repair_protocol"] = repair_builder.protocol

        def request_for(prior):
            current = copy.deepcopy(messages)
            current_options = copy.deepcopy(options)
            current_validator, context, metadata = validator, None, None
            if prior:
                custom = repair_builder(prior) if repair_builder is not None else None
                if custom is not None:
                    current, current_validator = custom.messages, custom.validator
                    context, metadata = custom.context, custom.metadata
                    current_options["response_format"]["json_schema"]["schema"] = custom.schema
                else:
                    current, context = self.repair_messages(current, prior, max_tokens, structured=schema is not None)
            return current, current_options, current_validator, context, metadata
        path = self.directory / f"{label}.json"
        if path.exists():
            record = read_json(path)
            if record["identity"] != identity or record["messages"] != messages:
                raise ValueError(f"Resume request identity mismatch: {label}")
        else:
            record = {"identity": identity, "messages": messages, "attempts": []}
        self.used_labels.append(label)
        for index, a in enumerate(record["attempts"]):
            if a.get("accepted"):
                if schema is not None and strict_response_json(a.get("raw_content")) != a["parsed"]:
                    raise ValueError(f"Cached raw/parsed response mismatch: {label}")
                if repair_builder is not None:
                    current, current_options, current_validator, context, metadata = request_for(record["attempts"][:index])
                    if (a.get("request_messages") != current or a.get("request_options") != current_options
                            or a.get("repair_context") != context or a.get("repair_metadata") != metadata):
                        raise ValueError(f"Cached repair request mismatch: {label}")
                    return current_validator(a["parsed"])
                return validator(a["parsed"])
        while True:
            retries = self.accounting()["retries"]
            if record["attempts"] and retries >= self.config.retry_budget:
                raise RuntimeError(f"Repair budget exhausted at {label}")
            remaining = self.config.case_timeout - (time.monotonic() - self.started)
            if remaining <= 0:
                raise TimeoutError("Agent case deadline exceeded")
            current, current_options, current_validator, repair_context, metadata = request_for(record["attempts"])
            prompt_count = self.budget.count(current)
            if prompt_count + max_tokens > 16384 or (schema is not None and prompt_count > self.budget.limit):
                raise ValueError("context_budget_exceeded at request dispatch")
            attempt = {"started_at": now(), "status": "inflight_or_interrupted", "accepted": False,
                       "request_messages": current, "repair_context": repair_context, "usage": None,
                       "estimated_prompt_tokens": prompt_count, "request_options": current_options,
                       "repair_metadata": metadata}
            record["attempts"].append(attempt)
            atomic_json(path, record)
            self.heartbeat("model_started", label)
            started = time.monotonic()
            try:
                parsed = self.client.chat_json(current, max_tokens=max_tokens, request_options=current_options,
                                               timeout_seconds=min(remaining, self.config.request_timeout))
                attempt.update(parsed=parsed, raw_content=self.client.last_raw_content,
                               status=self.client.last_call_status, finish_reason=self.client.last_finish_reason,
                               usage=self.client.last_usage, error_type=self.client.last_error_type)
                attempt["http_status"] = getattr(self.client, "last_http_status", None)
                if self.client.last_call_status != "success" or self.client.last_finish_reason != "stop":
                    raise ValueError(f"Model response status={self.client.last_call_status}; finish={self.client.last_finish_reason}")
                if not self.client.last_usage or not self.client.last_usage.get("total_tokens"):
                    raise ValueError("Model response has no token usage")
                if schema is not None:
                    parsed = strict_response_json(self.client.last_raw_content)
                    attempt["parsed"] = parsed
                result = current_validator(parsed)
                attempt["accepted"] = True
            except Exception as exc:
                attempt["exception_type"] = type(exc).__name__
                if isinstance(exc, ValidationError):
                    errors = exc.errors(include_url=False, include_context=False, include_input=False)
                    attempt["error"] = canonical(errors)[:2500]
                else:
                    attempt["error"] = str(exc)[:2500]
                # Unexpected program errors are not model repair opportunities.
                if not isinstance(exc, (ValueError, ValidationError)):
                    raise
            finally:
                attempt.update(finished_at=now(), elapsed_seconds=time.monotonic() - started)
                atomic_json(path, record)
                self.heartbeat("model_finished", label)
            if attempt["accepted"]:
                return result


def load_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(Path(path).expanduser()), local_files_only=True)
