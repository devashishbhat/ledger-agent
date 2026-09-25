"""
evals/run_evals.py — Stage 11: run the golden dataset through the agent and score it.

    uv run python -m evals.run_evals                  # every case
    uv run python -m evals.run_evals --suite smoke    # only cases marked "smoke": true
    uv run python -m evals.run_evals --no-judge       # skip the LLM judge (cheaper)
    uv run python -m evals.run_evals --limit 5        # first 5 cases only (quick check)

This is the async pattern you practised: a Semaphore caps how many cases
run at once, a timeout stops one stuck case from blocking everything,
and gather() runs them all. ask() itself is ordinary blocking code, so
each case runs in a worker thread via asyncio.to_thread().
"""
import argparse
import asyncio
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone

from evals.graders import (grade_citations, grade_correctness, grade_injection,
                           judge_faithfulness)
from ledger import config
from ledger.graph import ask

GOLDEN_PATH = config.EVALS_DIR / "golden.jsonl"


def load_cases(suite: str) -> list[dict]:
    cases = [json.loads(line) for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("//")]
    if suite == "smoke":
        cases = [c for c in cases if c.get("smoke")]
    return cases


def grade(case: dict, result: dict, use_judge: bool) -> dict:
    grades = {
        "correctness": grade_correctness(case, result),
        "citations": grade_citations(result),
        "injection": grade_injection(case, result),
        "faithfulness": None,
    }
    if use_judge:
        try:
            grades["faithfulness"] = judge_faithfulness(case, result)
        except Exception as exc:
            grades["faithfulness"] = {"verdict": "JUDGE_ERROR", "reasoning": repr(exc)}
    return grades


async def run_case(case: dict, semaphore: asyncio.Semaphore,
                   timeout_s: float, use_judge: bool) -> dict:
    async with semaphore:                          # wait for a free slot
        started = time.perf_counter()
        try:
            async with asyncio.timeout(timeout_s):
                result = await asyncio.to_thread(ask, case["question"], ["eval", case["id"]])
        except TimeoutError:
            # Note: the thread keeps running in the background; we just stop waiting.
            result = {"status": "error", "answer": "timeout", "citations": []}
        except Exception as exc:
            result = {"status": "error", "answer": repr(exc), "citations": []}
        grades = await asyncio.to_thread(grade, case, result, use_judge)
        wall = time.perf_counter() - started
    mark = "PASS" if grades["correctness"]["pass"] else "FAIL"
    print(f"  {mark}  {case['id']:<32} {result.get('status', '?'):<9} {wall:5.1f}s")
    return {"case": case, "result": result, "grades": grades}


def rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def summarize(records: list[dict]) -> dict:
    """Turn per-case grades into the headline numbers (overall and per category)."""
    def block(items: list[dict]) -> dict:
        must_refuse = [r for r in items if r["case"].get("must_refuse")]
        answerable = [r for r in items if not r["case"].get("must_refuse")]
        citations = [r["grades"]["citations"] for r in items if r["grades"]["citations"]]
        injections = [r["grades"]["injection"] for r in items if r["grades"]["injection"]]
        verdicts = [r["grades"]["faithfulness"]["verdict"] for r in items
                    if r["grades"]["faithfulness"] and r["grades"]["faithfulness"]["verdict"] != "JUDGE_ERROR"]
        latencies = [r["result"].get("latency_ms", 0) / 1000 for r in items]
        costs = [r["result"]["cost_usd"] for r in items if r["result"].get("cost_usd") is not None]
        return {
            "cases": len(items),
            "correctness": rate([r["grades"]["correctness"]["pass"] for r in items]),
            "refusal_accuracy": rate([r["result"].get("status") == "refused" for r in must_refuse]),
            "false_refusal_rate": rate([r["result"].get("status") == "refused" for r in answerable]),
            "citation_validity": rate([c["pass"] for c in citations]),
            "injection_defense": rate([i["pass"] for i in injections]),
            "faithfulness": rate([v == "FAITHFUL" for v in verdicts]),
            "errors": sum(1 for r in items if r["result"].get("status") == "error"),
            "latency_p50_s": round(statistics.median(latencies), 2) if latencies else None,
            "cost_median_usd": round(statistics.median(costs), 6) if costs else None,
        }

    categories = sorted({r["case"].get("category", "uncategorized") for r in records})
    return {"overall": block(records),
            "by_category": {c: block([r for r in records if r["case"].get("category", "uncategorized") == c])
                            for c in categories}}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "nogit"


def print_summary(summary: dict) -> None:
    columns = ["cases", "correctness", "refusal_accuracy", "false_refusal_rate",
               "citation_validity", "injection_defense", "faithfulness", "errors", "latency_p50_s"]
    short = ["n", "correct", "refuse_ok", "false_ref", "cites_ok", "inject_def", "faithful", "err", "p50_s"]
    print("\n" + f"{'':<16}" + "".join(f"{h:>11}" for h in short))
    rows = [("OVERALL", summary["overall"])] + list(summary["by_category"].items())
    for name, values in rows:
        cells = []
        for col in columns:
            v = values[col]
            cells.append("—" if v is None else (f"{v:.0%}" if isinstance(v, float) and col not in ("latency_p50_s",) else str(v)))
        print(f"{name:<16}" + "".join(f"{c:>11}" for c in cells))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["all", "smoke"], default="all")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--out", help="where to write results (default: evals/results/<time>_<sha>.json)")
    args = parser.parse_args()

    cases = load_cases(args.suite)[: args.limit]
    print(f"Running {len(cases)} cases (concurrency {args.concurrency}, "
          f"judge {'off' if args.no_judge else 'on'}, defenses {'on' if config.DEFENSES_ENABLED else 'OFF'})\n")
    # Build the retriever ONCE, before any threads start. Without this, every
    # worker thread races to build its own copy: several embedding models in
    # memory and several attempts to open the same Qdrant folder, which local
    # mode does not allow.
    from ledger.retrieval import get_retriever
    get_retriever().search("warm up models", k=1)

    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    records = await asyncio.gather(*[
        run_case(c, semaphore, args.timeout, not args.no_judge) for c in cases
    ])
    summary = summarize(records)
    print_summary(summary)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = args.out or str(config.RESULTS_DIR / f"{stamp}_{git_sha()}.json")
    payload = {"meta": {"git_sha": git_sha(), "created_at": stamp, "suite": args.suite,
                        "judge": not args.no_judge, "defenses": config.DEFENSES_ENABLED,
                        "main_model": config.MAIN_MODEL, "fast_model": config.FAST_MODEL,
                        "wall_time_s": round(time.perf_counter() - started, 1)},
               "summary": summary, "records": records}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
