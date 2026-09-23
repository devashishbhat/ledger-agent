"""
ledger/agent_simple.py — Stage 8: an agent with NO framework.

Run it with:
    uv run python -m ledger.agent_simple "What were Apple's total net sales in fiscal 2024?"

This is the raw loop that every agent framework (LangGraph included) is
built on. The model decides which tool to call; OUR code runs the tool;
the result goes back to the model; repeat until it answers.
You'll replace this with the LangGraph version in Stage 9, but you
should understand every line here first.
"""
import json
import sys
import time

from ledger import config
from ledger.llm import client
from ledger.prompts import SIMPLE_AGENT_SYSTEM
from ledger.retrieval import get_retriever
from ledger.tracing import record_llm_call, span, start_trace

# The tool's description is what the model reads to decide WHEN to use it.
# The input_schema says what arguments it must provide.
TOOLS = [{
    "name": "search_filings",
    "description": ("Search companies' annual reports (Form 10-K) and return matching passages. "
                    "Use one search per fact. Include the company and fiscal year when known."),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for"},
            "ticker": {"type": "string", "description": "Stock ticker, e.g. AAPL (optional)"},
            "fiscal_year": {"type": "integer", "description": "Fiscal year, e.g. 2024 (optional)"},
        },
        "required": ["query"],
    },
}]


def search_filings(query: str, ticker: str | None = None, fiscal_year: int | None = None) -> dict:
    """The actual Python function behind the tool."""
    results = get_retriever().search(
        query,
        tickers=[ticker.upper()] if ticker else None,
        years=[fiscal_year] if fiscal_year else None,
        k=5,
    )
    if not results:
        # Return errors/emptiness AS DATA so the model can adjust its next search.
        return {"passages": [], "note": "No matches. Try different words or no year filter."}
    return {"passages": [
        {"source": f"{c['ticker']} fiscal {c['fiscal_year']}, {c['section']}", "text": c["text"]}
        for c in results
    ]}


def run(question: str, max_iterations: int = 6) -> str:
    messages = [{"role": "user", "content": question}]

    with start_trace(question, tags=["simple-agent"]):
        for iteration in range(1, max_iterations + 1):     # ALWAYS bound the loop
            started = time.perf_counter()
            response = client().messages.create(
                model=config.MAIN_MODEL, max_tokens=1500,
                system=SIMPLE_AGENT_SYSTEM, tools=TOOLS, messages=messages,
            )
            record_llm_call("agent_step", config.MAIN_MODEL, response.usage.input_tokens,
                            response.usage.output_tokens,
                            (time.perf_counter() - started) * 1000, response.stop_reason)
            print(f"[step {iteration}] stop_reason={response.stop_reason} "
                  f"input_tokens={response.usage.input_tokens}")

            # Add the model's turn to the history (we must send it back next time).
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "max_tokens":
                return "Stopped: the response was cut off (max_tokens)."
            if response.stop_reason != "tool_use":
                # No tool requested -> this is the final answer.
                return "".join(block.text for block in response.content if block.type == "text")

            # The model asked for one or more tool calls. Run each one.
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                print(f"    -> {block.name}({json.dumps(block.input)})")
                with span("tool", block.name, arguments=block.input):
                    try:
                        output = search_filings(**block.input)
                    except Exception as exc:
                        output = {"error": str(exc)}      # an error is still a result
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,              # links this result to that request
                    "content": json.dumps(output),
                })
            # Tool results go back to the model as a "user" message.
            messages.append({"role": "user", "content": tool_results})

        return f"Stopped: no answer after {max_iterations} steps."


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('Usage: uv run python -m ledger.agent_simple "your question"')
    print("\n" + run(sys.argv[1]))
