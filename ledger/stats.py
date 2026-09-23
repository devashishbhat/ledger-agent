"""
ledger/stats.py — a small terminal "dashboard" built from data/traces/.

    uv run python -m ledger.stats              # every run ever recorded
    uv run python -m ledger.stats --tag eval   # only runs with this tag
"""
import argparse
import json
import statistics

from ledger import config


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="only include runs that have this tag")
    args = parser.parse_args()

    runs = []
    for path in sorted(config.TRACES_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            run = json.loads(line)
            if args.tag is None or args.tag in run.get("tags", []):
                runs.append(run)
    if not runs:
        raise SystemExit("No traces yet. Ask Ledger some questions first.")

    latencies = [r["latency_ms"] / 1000 for r in runs]
    costs = [r["cost_usd"] for r in runs if r.get("cost_usd") is not None]
    print(f"runs:                {len(runs)}")
    print(f"errors:              {sum(1 for r in runs if r.get('error'))}")
    print(f"latency p50 / p95:   {percentile(latencies, 50):.1f}s / {percentile(latencies, 95):.1f}s")
    print(f"LLM calls per run:   {statistics.mean(r['llm_calls'] for r in runs):.1f} (avg)")
    print(f"tokens per run:      {statistics.mean(r['input_tokens'] for r in runs):,.0f} in / "
          f"{statistics.mean(r['output_tokens'] for r in runs):,.0f} out (avg)")
    if costs:
        print(f"cost per run:        ${statistics.median(costs):.4f} median, "
              f"${max(costs):.4f} max, ${sum(costs):.2f} total")
    else:
        print("cost per run:        unknown (fill in PRICES_PER_MTOK in config.py)")

    # Where does the time go? Average duration of each kind of step.
    by_step: dict[str, list[float]] = {}
    for run in runs:
        for s in run["spans"]:
            by_step.setdefault(f"{s['kind']}:{s['name']}", []).append(s["duration_ms"])
    print("\naverage time per step:")
    for step, durations in sorted(by_step.items(), key=lambda kv: -statistics.mean(kv[1])):
        print(f"  {step:<22} {statistics.mean(durations) / 1000:6.2f}s  (x{len(durations)})")

    # Which runs needed the most searches? (A loop detector.)
    attempts = [sum(1 for s in r["spans"] if s["name"] == "retrieve" and s["kind"] == "node")
                for r in runs]
    print("\nsearch attempts per run:",
          {n: attempts.count(n) for n in sorted(set(attempts))})


if __name__ == "__main__":
    main()
