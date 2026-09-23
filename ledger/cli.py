"""
ledger/cli.py — ask Ledger a question from the terminal.

    uv run python -m ledger.cli "What were Apple's total net sales in fiscal 2024?"
    uv run python -m ledger.cli --approve "..."    # pause for your OK before saving a report
"""
import argparse
import uuid

from ledger.graph import ask, build_graph
from ledger.tracing import start_trace


def print_result(result: dict) -> None:
    print()
    print(f"STATUS: {result['status'].upper()}")
    print()
    print(result.get("answer", ""))
    if result.get("citations"):
        print("\nSOURCES:")
        for c in result["citations"]:
            print(f'  [{c["source"]}]')
            print(f'    "{c["quote"]}"')
            print(f'    {c["source_url"]}  (chars {c["doc_char_start"]}-{c["doc_char_end"]})')
    if result.get("flagged_chunk_ids"):
        print(f"\n  ! {len(result['flagged_chunk_ids'])} suspicious passage(s) were blocked")
    cost = result.get("cost_usd")
    print(f"\n  run {result.get('run_id', '-')} | {result.get('latency_ms', 0) / 1000:.1f}s | "
          f"{result.get('llm_calls', 0)} LLM calls | "
          f"{result.get('input_tokens', 0):,} in / {result.get('output_tokens', 0):,} out tokens | "
          f"cost {'unknown (set prices in config.py)' if cost is None else f'${cost:.4f}'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask Ledger about company 10-K filings.")
    parser.add_argument("question")
    parser.add_argument("--approve", action="store_true",
                        help="pause for human approval before saving a report")
    args = parser.parse_args()

    if not args.approve:
        print_result(ask(args.question, tags=["cli"]))
        return

    # Human-in-the-loop version: run until the graph pauses before "export".
    graph = build_graph(with_approval=True)
    # The thread_id tells the checkpointer which saved run to resume later.
    thread = {"configurable": {"thread_id": uuid.uuid4().hex}}
    with start_trace(args.question, tags=["cli", "approve"]) as trace:
        state = graph.invoke({"question": args.question}, thread)
        print_result({**state["result"], **trace.totals()})

        if graph.get_state(thread).next == ("export",):     # it paused, waiting for us
            reply = input("\nSave this answer as a report? [y/N] ").strip().lower()
            if reply == "y":
                state = graph.invoke(None, thread)            # None = "resume where you stopped"
                print(f"Saved: {state['result']['report_path']}")
            else:
                print("Not saved.")


if __name__ == "__main__":
    main()
