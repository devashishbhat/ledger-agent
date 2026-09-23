"""
ledger/tracing.py — records what happened during every agent run.

A "trace" is the record of one run (one question). It contains "spans":
one per step (planning, a search, an LLM call...), each with its timing,
and for LLM calls the tokens used and the cost.

Every finished trace is appended as one JSON line to data/traces/<date>.jsonl.
Later, `uv run python -m ledger.stats` summarizes those files.

How the code finds "the current trace" without passing it everywhere:
a ContextVar. Think of it as a global variable that is private to each
run, even when many runs happen at once (as in the eval harness).
"""
import contextvars
import json
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from ledger import config

_current_trace: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar(
    "ledger_current_trace", default=None)
_write_lock = threading.Lock()   # stops two threads writing the file at the same moment


class Trace:
    def __init__(self, question: str, tags: list[str] | None = None):
        self.run_id = uuid.uuid4().hex[:12]
        self.question = question
        self.tags = tags or []
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._t0 = time.perf_counter()
        self.spans: list[dict] = []
        self.error: str | None = None
        self.latency_ms: float | None = None

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    def totals(self) -> dict:
        llm = [s for s in self.spans if s["kind"] == "llm"]
        costs = [s.get("cost_usd") for s in llm]
        return {
            "run_id": self.run_id,
            "latency_ms": round(self.latency_ms or self.elapsed_ms(), 1),
            "llm_calls": len(llm),
            "input_tokens": sum(s.get("input_tokens", 0) for s in llm),
            "output_tokens": sum(s.get("output_tokens", 0) for s in llm),
            # If ANY call has an unknown price, the total is unknown too
            # (a partial sum would look precise and be wrong).
            "cost_usd": (round(sum(costs), 6) if costs and None not in costs
                         else (0.0 if not costs else None)),
        }

    def to_dict(self) -> dict:
        return {"question": self.question, "tags": self.tags, "started_at": self.started_at,
                "error": self.error, **self.totals(), "spans": self.spans}


def current_trace() -> Trace | None:
    return _current_trace.get()


@contextmanager
def start_trace(question: str, tags: list[str] | None = None):
    """
    Usage:
        with start_trace("What was Apple's revenue?") as trace:
            ... run the agent ...
        # when the block ends, the trace is saved to disk
    """
    trace = Trace(question, tags)
    token = _current_trace.set(trace)
    try:
        yield trace
    except Exception as exc:
        trace.error = repr(exc)
        raise
    finally:
        trace.latency_ms = trace.elapsed_ms()
        _current_trace.reset(token)
        _save(trace)


@contextmanager
def span(kind: str, name: str, **attributes):
    """
    Time one step and attach it to the current trace.
        with span("node", "retrieve", queries=3) as s:
            ...
            s["results"] = 12        # you can add fields while inside
    """
    trace = current_trace()
    record = {"kind": kind, "name": name,
              "start_ms": round(trace.elapsed_ms(), 1) if trace else 0, **attributes}
    t0 = time.perf_counter()
    try:
        yield record
    except Exception as exc:
        record["error"] = repr(exc)
        raise
    finally:
        record["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if trace is not None:
            trace.spans.append(record)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price_in, price_out = config.PRICES_PER_MTOK.get(model, (None, None))
    if price_in is None or price_out is None:
        return None
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def record_llm_call(name: str, model: str, input_tokens: int, output_tokens: int,
                    duration_ms: float, stop_reason: str | None = None) -> None:
    trace = current_trace()
    if trace is None:
        return
    trace.spans.append({
        "kind": "llm", "name": name, "model": model,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "cost_usd": cost_usd(model, input_tokens, output_tokens),
        "duration_ms": round(duration_ms, 1), "stop_reason": stop_reason,
    })


def _save(trace: Trace) -> None:
    config.TRACES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TRACES_DIR / f"{trace.started_at[:10]}.jsonl"
    line = json.dumps(trace.to_dict(), ensure_ascii=False, default=str)
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
