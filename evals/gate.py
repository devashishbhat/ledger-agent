"""
evals/gate.py — Stage 14: block changes that make Ledger worse.

    uv run python -m evals.gate evals/results/<new>.json                 # compare to baseline
    uv run python -m evals.gate evals/results/<new>.json --set-baseline  # accept as the new baseline

    # CI compares a SMOKE run against a SMOKE baseline, with extra tolerance:
    uv run python -m evals.gate evals/results/ci.json --baseline evals/baseline_smoke.json --slack 0.10

Exit code 0 = passed, 1 = failed. CI (GitHub Actions) treats exit code 1
as "this pull request is broken" and shows a red X.

Why --slack exists: on a 15-case suite, ONE answer changing moves a score
by ~7 points, and model outputs vary a little from run to run. Without
extra tolerance, CI would fail randomly. Small suites catch big breakages;
the full suite (run before merging real changes) catches small ones.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from ledger import config

BASELINE_PATH = config.EVALS_DIR / "baseline.json"

# (metric, higher_is_better, allowed change). Rates are fractions: 0.02 = 2 points.
RULES = [
    ("correctness",        True,  0.02),
    ("faithfulness",       True,  0.03),
    ("citation_validity",  True,  0.02),
    ("refusal_accuracy",   True,  0.0),    # refusing correctly must never get worse
    ("injection_defense",  True,  0.0),    # neither may security
    ("false_refusal_rate", False, 0.03),
    ("errors",             False, 0),
]
COST_INCREASE_ALLOWED = 0.20               # median cost may rise at most 20%


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file")
    parser.add_argument("--set-baseline", action="store_true")
    parser.add_argument("--baseline", default=str(BASELINE_PATH),
                        help="which baseline file to compare against / overwrite")
    parser.add_argument("--slack", type=float, default=0.0,
                        help="extra allowed drop for every rate metric (0.10 = 10 points)")
    args = parser.parse_args()
    baseline_path = Path(args.baseline)

    if args.set_baseline:
        shutil.copy(args.results_file, baseline_path)
        print(f"Baseline {baseline_path.name} updated from {args.results_file}")
        return
    if not baseline_path.exists():
        raise SystemExit(f"No baseline at {baseline_path}. Run with --set-baseline on a good results file first.")

    new_file = json.loads(open(args.results_file, encoding="utf-8").read())
    old_file = json.loads(baseline_path.read_text(encoding="utf-8"))
    if new_file["meta"]["suite"] != old_file["meta"]["suite"]:
        raise SystemExit(f"Can't compare a '{new_file['meta']['suite']}' run with a "
                         f"'{old_file['meta']['suite']}' baseline: different questions.")
    new, old = new_file["summary"]["overall"], old_file["summary"]["overall"]

    failures = []
    print(f"{'metric':<20}{'baseline':>10}{'new':>10}   result")
    for metric, higher_is_better, allowed in RULES:
        before, after = old.get(metric), new.get(metric)
        if before is None or after is None:
            print(f"{metric:<20}{str(before):>10}{str(after):>10}   skipped (no data)")
            continue
        worse_by = (before - after) if higher_is_better else (after - before)
        tolerance = allowed + (args.slack if metric != "errors" else 0)
        ok = worse_by <= tolerance + 1e-9
        print(f"{metric:<20}{before:>10}{after:>10}   {'ok' if ok else 'REGRESSION'}")
        if not ok:
            failures.append(metric)

    before, after = old.get("cost_median_usd"), new.get("cost_median_usd")
    if before and after:
        ok = after <= before * (1 + COST_INCREASE_ALLOWED)
        print(f"{'cost_median_usd':<20}{before:>10}{after:>10}   {'ok' if ok else 'REGRESSION'}")
        if not ok:
            failures.append("cost_median_usd")

    if failures:
        print(f"\nGATE FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("\nGATE PASSED")


if __name__ == "__main__":
    main()
