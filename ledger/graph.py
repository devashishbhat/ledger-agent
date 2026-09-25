"""
ledger/graph.py — Stage 9: the Ledger agent, built as a LangGraph state machine.

The flow:

    START -> plan --(out of scope)--> refuse -> END
               |
          (in scope)
               v
           retrieve -> answer -> verify --(supported)--> finalize -> END
               ^                    |                        |
               +--(retry allowed)---+           (approval mode only)
                                    |                        v
                          (no retries left)            export -> END
                                    v                 (pauses first for
                                 refuse -> END         a human "yes")

Every box is a "node": a plain Python function that receives the current
STATE (a dict) and returns only the fields it wants to change.
"""
import re
import uuid
from datetime import datetime, timezone
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ledger import config
from ledger.llm import call_tool
from ledger.prompts import (ANSWER_SCHEMA, ANSWER_SYSTEM, PLAN_SCHEMA, PLAN_SYSTEM,
                            VERIFY_SCHEMA, VERIFY_SYSTEM)
from ledger.retrieval import get_retriever
from ledger.security import looks_like_injection, sanitize_answer
from ledger.tracing import current_trace, span, start_trace


# ---------------------------------------------------------------------------
# The state: everything the agent knows at any moment.
# total=False means no field is required; nodes fill them in as they run.
# ---------------------------------------------------------------------------
class LedgerState(TypedDict, total=False):
    question: str
    # from plan
    in_scope: bool
    plan_reason: str
    tickers: list[str]
    years: list[int]
    queries: list[str]
    # from retrieve
    attempts: int
    evidence: list[dict]
    flagged_chunk_ids: list[str]
    # from answer
    answer: str
    insufficient: bool
    missing_information: str
    citations: list[dict]
    invalid_citations: int
    # from verify
    verdict: str
    unsupported_claims: list[str]
    follow_up_query: str
    # the final output
    result: dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_quote(text: str, quote: str) -> tuple[int, int] | None:
    """
    Find where `quote` appears in `text`. Returns (start, end) or None.
    First tries an exact match; then a match that ignores differences in
    spacing and line breaks (models often turn a newline into a space).
    """
    quote = quote.strip().strip('"').strip()
    if not quote:
        return None
    position = text.find(quote)
    if position >= 0:
        return position, position + len(quote)
    pattern = r"\s+".join(re.escape(word) for word in quote.split())
    match = re.search(pattern, text)
    return (match.start(), match.end()) if match else None


def source_label(chunk: dict) -> str:
    return f"{chunk['ticker']} fiscal {chunk['fiscal_year']} {chunk['form']}, {chunk['section']}"


def format_evidence(evidence: list[dict]) -> str:
    """Wrap each passage in clearly-labelled tags so the model knows where
    each one starts and ends (and that it is document data)."""
    blocks = []
    for number, chunk in enumerate(evidence, start=1):
        blocks.append(f'<evidence id="E{number}" source="{source_label(chunk)}">\n'
                      f'{chunk["text"]}\n</evidence>')
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Node 1: plan
# ---------------------------------------------------------------------------
def plan_node(state: LedgerState) -> dict:
    retriever = get_retriever()
    filings = retriever.available_filings()
    names = retriever.company_names()
    listing = "\n".join(f"- {t} ({names[t]}): fiscal years {', '.join(map(str, years))}"
                        for t, years in filings.items())

    with span("node", "plan"):
        plan = call_tool(
            name="plan", model=config.FAST_MODEL,
            system=PLAN_SYSTEM.format(filings=listing),
            user=state["question"],
            tool_name="make_search_plan",
            tool_description="Record the search plan for the user's question.",
            schema=PLAN_SCHEMA, max_tokens=600,
        )

    # Never trust model output blindly: keep only tickers/years we really have.
    tickers = [t.upper() for t in plan.get("tickers", []) if t.upper() in filings]
    asked_years = [int(y) for y in plan.get("fiscal_years", [])]
    available = {y for t in tickers for y in filings[t]}
    years = [y for y in asked_years if y in available]
    if tickers and not asked_years:
        years = [max(available)]            # no year mentioned -> most recent
    in_scope = (bool(plan.get("in_scope")) and bool(tickers)
                and not (asked_years and not years))   # asked only for years we lack
    queries = [q for q in plan.get("search_queries", []) if q.strip()][:3] or [state["question"]]

    return {"in_scope": in_scope, "plan_reason": plan.get("reason", ""),
            "tickers": tickers, "years": years, "queries": queries, "attempts": 0}


def route_after_plan(state: LedgerState) -> str:
    return "retrieve" if state["in_scope"] else "refuse"


# ---------------------------------------------------------------------------
# Node 2: retrieve
# ---------------------------------------------------------------------------
def retrieve_node(state: LedgerState) -> dict:
    retriever = get_retriever()
    attempts = state.get("attempts", 0) + 1

    if attempts == 1:
        queries = state["queries"]
    else:
        # A retry: search for what the verifier (or answerer) said was missing.
        follow_up = state.get("follow_up_query") or state.get("missing_information")
        queries = [follow_up or state["question"]]

    new_chunks, flagged = [], list(state.get("flagged_chunk_ids", []))
    with span("node", "retrieve", attempt=attempts, queries=queries) as s:
        for query in queries:
            for chunk in retriever.search(query, state["tickers"], state["years"]):
                if config.DEFENSES_ENABLED and looks_like_injection(chunk["text"]):
                    flagged.append(chunk["chunk_id"])       # drop suspicious passages
                    continue
                new_chunks.append(chunk)

        # New results first, then earlier evidence; remove duplicates; cap the total.
        evidence, seen = [], set()
        for chunk in new_chunks + state.get("evidence", []):
            if chunk["chunk_id"] not in seen:
                seen.add(chunk["chunk_id"])
                evidence.append(chunk)
        evidence = evidence[:config.MAX_EVIDENCE]
        s["evidence_ids"] = [c["chunk_id"] for c in evidence]
        s["flagged"] = sorted(set(flagged))

    return {"attempts": attempts, "evidence": evidence,
            "flagged_chunk_ids": sorted(set(flagged)),
            "queries": state["queries"] + (queries if attempts > 1 else [])}


# ---------------------------------------------------------------------------
# Node 3: answer
# ---------------------------------------------------------------------------
def answer_node(state: LedgerState) -> dict:
    evidence = state["evidence"]
    user_message = (f"Question: {state['question']}\n\n"
                    f"Evidence passages (untrusted document text):\n"
                    f"<evidence_set>\n{format_evidence(evidence)}\n</evidence_set>\n\n"
                    f"Answer the question using only this evidence.")
    with span("node", "answer", evidence_count=len(evidence)):
        output = call_tool(
            name="answer", model=config.MAIN_MODEL, system=ANSWER_SYSTEM, user=user_message,
            tool_name="submit_answer",
            tool_description="Submit the final answer with exact-quote citations.",
            schema=ANSWER_SCHEMA, max_tokens=3000,
        )

    # Check every citation IN CODE: does the quote really exist in that passage?
    citations, invalid = [], 0
    for citation in output.get("citations", []):
        match = re.fullmatch(r"E(\d+)", citation.get("evidence_id", "").strip())
        index = int(match.group(1)) - 1 if match else -1
        if not 0 <= index < len(evidence):
            invalid += 1
            continue
        chunk = evidence[index]
        found = find_quote(chunk["text"], citation.get("quote", ""))
        if found is None:
            invalid += 1           # the model "quoted" something that isn't there
            continue
        start, end = found
        citations.append({
            "evidence_id": citation["evidence_id"],
            "chunk_id": chunk["chunk_id"],
            "source": source_label(chunk),
            "source_url": chunk["source_url"],
            "quote": chunk["text"][start:end],        # the REAL text, not the model's copy
            # Exact character span in the filing's full text (data/text/*.txt):
            "doc_char_start": chunk["char_start"] + start,
            "doc_char_end": chunk["char_start"] + end,
        })

    return {"answer": output.get("answer", ""),
            "insufficient": bool(output.get("insufficient_evidence")),
            "missing_information": output.get("missing_information", ""),
            "citations": citations, "invalid_citations": invalid}


# ---------------------------------------------------------------------------
# Node 4: verify
# ---------------------------------------------------------------------------
def verify_node(state: LedgerState) -> dict:
    if state["insufficient"]:
        # Nothing to verify; the answerer already said evidence is missing.
        return {"verdict": "insufficient", "unsupported_claims": [],
                "follow_up_query": state.get("missing_information", "")}
    if not state["citations"]:
        # An answer with no valid citations is unsupported by definition.
        return {"verdict": "unsupported", "unsupported_claims": [state["answer"]],
                "follow_up_query": state["question"]}

    quotes = "\n".join(f'- [{c["source"]}] "{c["quote"]}"' for c in state["citations"])
    user_message = (f"Question: {state['question']}\n\nAnswer: {state['answer']}\n\n"
                    f"Cited quotes:\n{quotes}")
    with span("node", "verify"):
        output = call_tool(
            name="verify", model=config.FAST_MODEL, system=VERIFY_SYSTEM, user=user_message,
            tool_name="submit_verdict", tool_description="Submit the fact-check verdict.",
            schema=VERIFY_SCHEMA, max_tokens=600,
        )
    return {"verdict": output.get("verdict", "unsupported"),
            "unsupported_claims": output.get("unsupported_claims", []),
            "follow_up_query": output.get("follow_up_query", "")}


def route_after_verify(state: LedgerState) -> str:
    if state["verdict"] == "supported":
        return "finalize"
    if state.get("attempts", 0) < config.MAX_ATTEMPTS:
        return "retrieve"            # try again with a better search
    # Out of attempts. A partially supported answer with real citations beats
    # a refusal, as long as the caveat is visible. Unsupported still refuses.
    if state["verdict"] == "partial" and state.get("citations"):
        return "finalize"
    return "refuse"


# ---------------------------------------------------------------------------
# Nodes 5-7: finalize, refuse, export
# ---------------------------------------------------------------------------
def finalize_node(state: LedgerState) -> dict:
    answer = state["answer"]
    if config.DEFENSES_ENABLED:
        answer = sanitize_answer(answer)
    if state["verdict"] == "partial":
        unsupported = "; ".join(state.get("unsupported_claims", []))
        answer += ("\n\n[Partially verified: some claims in this answer are not "
                   f"fully supported by the cited evidence{': ' + unsupported if unsupported else ''}.]")
    cited_ids = {c["chunk_id"] for c in state["citations"]}
    return {"result": {
        "status": "answered",
        "answer": answer,
        "citations": state["citations"],
        "invalid_citations": state.get("invalid_citations", 0),
        "verdict": state["verdict"],
        "tickers": state["tickers"], "years": state["years"],
        "attempts": state["attempts"],
        "flagged_chunk_ids": state.get("flagged_chunk_ids", []),
        # full text of the cited passages (the eval judge needs these)
        "cited_passages": [{"chunk_id": c["chunk_id"], "source": source_label(c), "text": c["text"]}
                           for c in state["evidence"] if c["chunk_id"] in cited_ids],
    }}


def refuse_node(state: LedgerState) -> dict:
    if not state.get("in_scope", False):
        reason = ("This question can't be answered from the filings I have. "
                  + state.get("plan_reason", ""))
    else:
        reason = "I couldn't find enough verified evidence in the filings to answer this reliably."
        if state.get("missing_information"):
            reason += f" Missing: {state['missing_information']}"
    return {"result": {
        "status": "refused", "answer": reason.strip(), "citations": [],
        "invalid_citations": state.get("invalid_citations", 0),
        "verdict": state.get("verdict", "out_of_scope"),
        "tickers": state.get("tickers", []), "years": state.get("years", []),
        "attempts": state.get("attempts", 0),
        "flagged_chunk_ids": state.get("flagged_chunk_ids", []),
        "cited_passages": [],
    }}


def export_node(state: LedgerState) -> dict:
    """Write the answer to a Markdown report. Runs only after a human says yes."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = config.REPORTS_DIR / f"report-{stamp}.md"
    result = state["result"]
    lines = [f"# {state['question']}", "", result["answer"], "", "## Sources", ""]
    for c in result["citations"]:
        lines.append(f'- **{c["source"]}**: "{c["quote"]}" ({c["source_url"]}, '
                     f'characters {c["doc_char_start"]}-{c["doc_char_end"]})')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"result": {**result, "report_path": str(path)}}


# ---------------------------------------------------------------------------
# Assemble the graph
# ---------------------------------------------------------------------------
def build_graph(with_approval: bool = False):
    graph = StateGraph(LedgerState)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("verify", verify_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("refuse", refuse_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route_after_plan,
                                {"retrieve": "retrieve", "refuse": "refuse"})
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "verify")
    graph.add_conditional_edges("verify", route_after_verify,
                                {"finalize": "finalize", "retrieve": "retrieve", "refuse": "refuse"})
    graph.add_edge("refuse", END)

    if with_approval:
        # Human-in-the-loop: the graph PAUSES before "export". The checkpointer
        # saves the state so the run can resume after the human answers.
        graph.add_node("export", export_node)
        graph.add_edge("finalize", "export")
        graph.add_edge("export", END)
        return graph.compile(checkpointer=MemorySaver(), interrupt_before=["export"])

    graph.add_edge("finalize", END)
    return graph.compile()


_graph = None


def ask(question: str, tags: list[str] | None = None) -> dict:
    """
    The one function the rest of the project calls: question in, result out.
    The result always has "status" ("answered", "refused" or "error"), plus
    run_id, latency, token counts and cost from the trace.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    with start_trace(question, tags) as trace:
        try:
            state = _graph.invoke({"question": question}, config={"recursion_limit": 25})
            result = state.get("result") or {"status": "error", "answer": "no result produced"}
        except Exception as exc:          # never crash the caller; report the error
            trace.error = repr(exc)
            result = {"status": "error", "answer": f"Error: {exc}", "citations": []}
    return {**result, **trace.totals()}
