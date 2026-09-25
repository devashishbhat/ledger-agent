"""
evals/judge_calibration.py — Stage 11: check whether the LLM judge can be trusted.

Step 1 — YOU label answers by hand:
    uv run python -m evals.judge_calibration label evals/results/<file>.json --n 50
Step 2 — compare your labels to the judge's:
    uv run python -m evals.judge_calibration score
Step 3 — after editing JUDGE_SYSTEM in ledger/prompts.py, re-run the judge on
         the SAME answers you labelled, and compare again:
    uv run python -m evals.judge_calibration score --rejudge

Labels are saved to evals/labels.jsonl. Each record keeps the question,
answer and passages, so the comparison is always on identical material.
"""
import argparse
import json
import random

from evals.graders import cohens_kappa, judge_faithfulness
from ledger import config

LABELS_PATH = config.EVALS_DIR / "labels.jsonl"
CHOICES = {"f": "FAITHFUL", "p": "PARTIAL", "u": "UNFAITHFUL"}


def load_labels() -> list[dict]:
    if not LABELS_PATH.exists():
        return []
    return [json.loads(line) for line in LABELS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def label(results_file: str, n: int, seed: int) -> None:
    data = json.loads(open(results_file, encoding="utf-8").read())
    already = {(l["results_file"], l["case_id"]) for l in load_labels()}
    candidates = [r for r in data["records"]
                  if r["grades"]["faithfulness"]
                  and r["grades"]["faithfulness"]["verdict"] in CHOICES.values()
                  and (results_file, r["case"]["id"]) not in already]
    random.Random(seed).shuffle(candidates)       # a fixed seed = the same sample every time
    print(f"{len(candidates)} unlabelled answered cases; labelling up to {n}.")
    print("For each: is EVERY claim in the answer supported by the passages?")
    print("  f = faithful   p = partial   u = unfaithful   s = skip   q = quit\n")

    for record in candidates[:n]:
        case, result = record["case"], record["result"]
        print("=" * 80)
        print(f"QUESTION: {case['question']}\n\nANSWER: {result['answer']}\n\nPASSAGES:")
        for p in result.get("cited_passages", []):
            print(f"  [{p['source']}]\n  {p['text'][:1200]}\n")
        choice = ""
        while choice not in ("f", "p", "u", "s", "q"):
            choice = input("your label [f/p/u/s/q]: ").strip().lower()
        if choice == "q":
            break
        if choice == "s":
            continue
        entry = {"results_file": results_file, "case_id": case["id"],
                 "question": case["question"], "answer": result["answer"],
                 "cited_passages": result.get("cited_passages", []),
                 "human": CHOICES[choice],
                 "judge": record["grades"]["faithfulness"]["verdict"]}
        with open(LABELS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nTotal labels so far: {len(load_labels())}")


def score(rejudge: bool) -> None:
    labels = load_labels()
    if not labels:
        raise SystemExit("No labels yet. Run the 'label' step first.")
    if rejudge:
        print(f"Re-running the current judge prompt on {len(labels)} labelled answers...")
        for entry in labels:
            fake_result = {"status": "answered", "answer": entry["answer"],
                           "cited_passages": entry["cited_passages"]}
            entry["judge"] = judge_faithfulness({"question": entry["question"]}, fake_result)["verdict"]

    human = [e["human"] for e in labels]
    judge = [e["judge"] for e in labels]
    agreement = sum(h == j for h, j in zip(human, judge)) / len(labels)
    print(f"\nlabelled cases:  {len(labels)}")
    print(f"agreement:       {agreement:.0%}")
    print(f"Cohen's kappa:   {cohens_kappa(human, judge):.2f}   (aim for 0.70 or higher)")

    # Confusion matrix: rows = what you said, columns = what the judge said.
    names = list(CHOICES.values())
    print("\n" + " " * 14 + "judge ->  " + "".join(f"{n[:6]:>9}" for n in names))
    for h in names:
        counts = [sum(1 for a, b in zip(human, judge) if a == h and b == j) for j in names]
        print(f"  you: {h:<12}      " + "".join(f"{c:>9}" for c in counts))

    disagreements = [e for e in labels if e["human"] != e["judge"]]
    if disagreements:
        print(f"\n{len(disagreements)} disagreements (read these, then fix the rubric or your label):")
        for e in disagreements:
            print(f"  - {e['case_id']}: you={e['human']} judge={e['judge']} | {e['question'][:70]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_label = sub.add_parser("label")
    p_label.add_argument("results_file")
    p_label.add_argument("--n", type=int, default=50)
    p_label.add_argument("--seed", type=int, default=7)
    p_score = sub.add_parser("score")
    p_score.add_argument("--rejudge", action="store_true")
    args = parser.parse_args()
    if args.command == "label":
        label(args.results_file, args.n, args.seed)
    else:
        score(args.rejudge)


if __name__ == "__main__":
    main()
