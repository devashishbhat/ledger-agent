"""
ledger/llm.py — The one place that talks to Claude.

Every model call in the project goes through call_tool(). Keeping it in one
place means retries, token logging, and cost tracking happen everywhere
automatically.

The trick used here is "forced tool use": we describe the exact JSON shape
we want as a tool, and force the model to call that tool. The model's
answer then arrives as structured data (a Python dict) instead of free
text we'd have to parse.
"""
import time

import anthropic

from ledger import config
from ledger.tracing import record_llm_call

_client: anthropic.Anthropic | None = None


class LLMError(RuntimeError):
    """Raised when the model's response is unusable."""


def client() -> anthropic.Anthropic:
    """Create the API client once and reuse it (it reads ANTHROPIC_API_KEY itself)."""
    global _client
    if _client is None:
        # max_retries: the SDK automatically retries rate limits (429),
        # overloads (529) and server errors (5xx) with exponential backoff.
        _client = anthropic.Anthropic(max_retries=4, timeout=120.0)
    return _client


def call_tool(*, name: str, model: str, system: str, user: str,
              tool_name: str, tool_description: str, schema: dict,
              max_tokens: int = 1500) -> dict:
    """
    Ask the model to fill in `schema` and return the result as a dict.

    name:   a label for this call in traces ("plan", "answer", ...)
    schema: a JSON Schema describing the fields we want back

    Note: current versions of the Anthropic SDK no longer take a
    `temperature` argument on messages.create. Runs are therefore not
    perfectly repeatable, which is one reason the eval suite reports rates
    over many cases rather than trusting any single run.
    """
    started = time.perf_counter()
    response = client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        tools=[{"name": tool_name, "description": tool_description, "input_schema": schema}],
        tool_choice={"type": "tool", "name": tool_name},   # force THIS tool
        messages=[{"role": "user", "content": user}],
    )
    record_llm_call(
        name=name, model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        duration_ms=(time.perf_counter() - started) * 1000,
        stop_reason=response.stop_reason,
    )
    # If the model ran out of room, the JSON may be cut off. Fail loudly
    # instead of silently using half an answer.
    if response.stop_reason == "max_tokens":
        raise LLMError(f"{name}: hit max_tokens={max_tokens}; output may be truncated")
    for block in response.content:
        if block.type == "tool_use":
            return block.input          # already a Python dict
    raise LLMError(f"{name}: model did not return a tool call")
