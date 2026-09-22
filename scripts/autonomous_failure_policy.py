"""Terminal arm failures and a latched, request-level service circuit breaker.

Only known model/transport failures may be scored as no recommendation. Unknown
exceptions and integrity failures remain fatal. No prediction is fabricated.
"""
from collections import Counter, deque
import threading

from iaa_agent.agent_runtime import read_json, now


POLICY = {
    "version": "terminal_arm_failures_v1",
    "strict_phases": ["development", "smoke50"],
    "tolerant_phases": ["validation", "full"],
    "retries": "unchanged shared agent repair budget; no outer retries",
    "ranking_score": "no valid recommendation => zero on all scheduled sessions",
    "candidate_metrics": "observed successful outputs only; coverage reported",
    "service_consecutive_failures": 5,
    "service_window": 20,
    "service_window_failures": 8,
    "service_action": "latch; stop admission and new arms; drain inflight arms; exit paused",
}

SOFT_KINDS = {"model", "infrastructure", "dependency"}


def attempt_kind(attempt):
    """Classify the actual rejection, never all RuntimeError/ValueError alike."""
    if attempt.get("accepted"):
        return "success"
    status = attempt.get("status")
    if status == "request_error":
        code = attempt.get("http_status")
        if code in {408, 429} or isinstance(code, int) and 500 <= code <= 599:
            return "infrastructure"
        if code is None and attempt.get("error_type") in {"URLError", "TimeoutError", "ConnectionError"}:
            return "infrastructure"
        return "fatal"  # Unknown HTTP codes, auth/config errors require investigation.
    if status in {"invalid_response_json", "empty_content"}:
        return "infrastructure"
    if status in {"length_truncated", "invalid_content_json"}:
        return "model"
    if status != "success":
        return "fatal"  # An interrupted request is not a known model failure.
    error = attempt.get("error", "")
    if error == "Model response has no token usage":
        return "infrastructure"
    if attempt.get("finish_reason") == "length":
        return "model"
    if attempt.get("exception_type") in {"ValidationError", "JSONDecodeError"}:
        return "model"
    # Older journals did not retain exception_type. Their Pydantic diagnostic
    # has a structured list of loc/type/msg entries, not an arbitrary message.
    if error.startswith("[{"):
        import json
        try:
            errors = json.loads(error)
            if errors and all(isinstance(e, dict) and {"type", "loc", "msg"} <= e.keys() for e in errors):
                return "model"
        except ValueError:
            pass
    prefixes = (
        "Duplicate JSON object key:", "Ranking has ", "Ranking contains IDs outside",
        "Invalid/unseen or wrong-POI evidence references", "ranked_pois.poi_idx:",
        "working_poi_ids:", "Return working_poi_selection", "working_poi_selection must",
        "Working candidate limit exceeded", "Working set contains an unregistered POI",
        "Working set has ", "Stop needs at least ", "Budget requires consolidation:",
        "Tool call budget exceeded", "stop=false requires", "Fixed schedule requires exactly",
        "Fixed expansion requires", "Supply one parameter object per fixed tool slot",
    )
    return "model" if error.startswith(prefixes) else "fatal"


def records_and_accounting(directory):
    records = [read_json(p) for p in sorted((directory / "calls").glob("*.json"))]
    attempts = [a for r in records for a in r["attempts"]]
    usage = Counter()
    for a in attempts:
        usage.update(a.get("usage") or {})
    return records, {
        "logical_calls": len(records), "requests": len(attempts),
        "retries": sum(max(0, len(r["attempts"]) - 1) for r in records),
        "invalid_attempts": sum(not a.get("accepted") for a in attempts),
        "usage_missing_count": sum(not a.get("usage") for a in attempts),
        "usage": dict(usage), "model_seconds": sum(a.get("elapsed_seconds", 0) for a in attempts),
    }


def classify_failure(exc, records):
    message = str(exc)
    # A previous unknown error must never be hidden by a later timeout/repair.
    attempts = [a for r in records for a in r["attempts"]]
    kinds = [attempt_kind(a) for a in attempts if not a.get("accepted")]
    if "fatal" in kinds:
        return "fatal"
    if isinstance(exc, TimeoutError) and message == "Agent case deadline exceeded":
        return "model"
    if isinstance(exc, ValueError) and message.startswith("context_budget_exceeded"):
        return "model"
    if isinstance(exc, RuntimeError) and message in {
        "Insufficient discovered candidates at consolidation deadline",
        "Insufficient valid candidates after decision budget",
    }:
        return "model"
    if isinstance(exc, RuntimeError) and message.startswith("Repair budget exhausted at ") and kinds:
        # The last rejected request caused termination; retain the whole
        # per-attempt breakdown separately when causes were mixed.
        rejected = sorted((a for a in attempts if not a.get("accepted")), key=lambda a: a.get("started_at", ""))
        return attempt_kind(rejected[-1])
    return "fatal"


def failure_row(exc, directory, *, elapsed_seconds=0, dependency=None, baseline_accounting=None):
    from iaa_agent.agent_evaluation import combined_accounting
    records, exclusive = records_and_accounting(directory)
    kind = "dependency" if dependency else classify_failure(exc, records)
    accounting = combined_accounting(baseline_accounting, exclusive) if baseline_accounting else exclusive
    return {"status": "failed", "failure_kind": kind, "error_type": type(exc).__name__,
            "error": str(exc), "dependency": dependency, "finished_at": now(),
            "attempt_causes": dict(Counter(attempt_kind(a) for r in records for a in r["attempts"])),
            "predictions": [], "rank": None, "valid": False, "heuristic_fallback": False,
            "accounting": accounting, "exclusive_accounting": exclusive,
            "elapsed_seconds": max(elapsed_seconds, accounting["model_seconds"]),
            "elapsed_basis": "max(current_invocation_wall_seconds, all_recorded_model_seconds)", "tool_calls": None}


class ServiceCircuit:
    def __init__(self, policy=POLICY):
        self.policy = policy
        self.lock = threading.Lock()
        self.window = deque(maxlen=policy["service_window"])
        self.consecutive = 0
        self.reason = None

    @property
    def opened(self):
        with self.lock:
            return self.reason is not None

    def observe(self, attempt):
        bad = attempt_kind(attempt) == "infrastructure"
        with self.lock:
            self.window.append(bad)
            self.consecutive = self.consecutive + 1 if bad else 0
            if self.reason is None and (self.consecutive >= self.policy["service_consecutive_failures"]
                    or sum(self.window) >= self.policy["service_window_failures"]):
                self.reason = {"at": now(), "consecutive": self.consecutive,
                               "window_requests": len(self.window), "window_failures": sum(self.window)}
            return self.reason
