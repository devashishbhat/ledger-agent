"""
evals/retrieval_eval.py — Stage 6: measure search quality BEFORE touching any AI model.

    uv run python -m evals.retrieval_eval

For each labelled query in evals/retrieval_queries.jsonl we ask: did the
passage containing the answer show up in the top k results?

We label with "the answer text" (e.g. "391,035") instead of chunk ids.
Chunk ids change every time you re-chunk; the answer text doesn't, so
your labels survive changes to chunking.

Prints the ablation table for your README: each search method with and
without metadata filters.
"""
import json
import re

from ledger import config
from ledger.retrieval import get_retriever

QUERIES_PATH = config.EVALS_DIR / "retrieval_queries.jsonl"
CONFIGS = [
    ("vector",        "vector",        False),
    ("bm25",          "bm25",          False),
    ("hybrid (RRF)",  "hybrid",        False),
    ("hybrid+rerank", "hybrid_rerank", False),
    ("hybrid+rerank+filters", "hybrid_rerank", True),
]


def normalize(text: str) -> str:
    """Make matching ignore formatting: "$391,035" and "391035" both become "391035".
    (An identical function lives in evals/graders.py, which you write in Stage 11.)"""
    text = text.lower().replace("\u2019", "'").replace("$", "")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def is_hit(chunk: dict, query: dict) -> bool:
    if chunk["ticker"] != query["ticker"]:
        return False
    text = normalize(chunk["text"])
    return any(normalize(expected) in text for expected in query["expected_any"])


def main() -> None:
    queries = [json.loads(line) for line in QUERIES_PATH.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    retriever = get_retriever()
    print(f"{len(queries)} labelled queries\n")
    print(f"{'configuration':<26}{'recall@5':>10}{'recall@10':>11}{'MRR':>8}")

    for label, mode, use_filters in CONFIGS:
        hits5 = hits10 = 0
        reciprocal_ranks = []
        misses = []
        for q in queries:
            results = retriever.search(q["query"], tickers=[q["ticker"]], years=[q["fiscal_year"]],
                                       k=10, mode=mode, use_filters=use_filters)
            rank = next((i for i, c in enumerate(results) if is_hit(c, q)), None)
            hits5 += rank is not None and rank < 5
            hits10 += rank is not None
            reciprocal_ranks.append(0 if rank is None else 1 / (rank + 1))
            if rank is None:
                misses.append(q["query"])
        n = len(queries)
        print(f"{label:<26}{hits5 / n:>10.2f}{hits10 / n:>11.2f}{sum(reciprocal_ranks) / n:>8.2f}")
        if use_filters and misses:
            print("\n  still missed with the best configuration:")
            for m in misses:
                print(f"   - {m}")


if __name__ == "__main__":
    main()
