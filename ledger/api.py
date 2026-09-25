"""
ledger/api.py — Stage 13: Ledger as a web service.

Run locally:
    uv run uvicorn ledger.api:app --reload
Then open http://127.0.0.1:8000/docs for an interactive page where you
can type a question and press "Execute".

Protections, because a public URL that spends YOUR API credits is a risk:
  - LEDGER_API_KEY: if set, every request must send header  X-API-Key: <that value>
  - LEDGER_MAX_RUNS_PER_DAY: hard cap on questions per day (default 100)
"""
import os
import threading
from datetime import date

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ledger.graph import ask
from ledger.retrieval import get_retriever

app = FastAPI(title="Ledger", description="Evidence-grounded answers from SEC 10-K filings.")

MAX_RUNS_PER_DAY = int(os.environ.get("LEDGER_MAX_RUNS_PER_DAY", "100"))
_usage = {"day": date.today(), "runs": 0}
_usage_lock = threading.Lock()


class AskRequest(BaseModel):
    # Field(...) adds validation: FastAPI rejects empty or huge questions
    # with a 422 error before our code even runs.
    question: str = Field(min_length=3, max_length=500,
                          examples=["What were Apple's total net sales in fiscal 2024?"])


@app.get("/healthz")
def healthz() -> dict:
    """Liveness: is the process running at all?"""
    return {"ok": True}


@app.get("/readyz")
def readyz() -> dict:
    """Readiness: is the search index loaded and usable?"""
    retriever = get_retriever()
    return {"ready": True, "chunks": len(retriever.chunks),
            "filings": retriever.available_filings()}


@app.post("/v1/ask")
def ask_endpoint(request: AskRequest, x_api_key: str | None = Header(default=None)) -> dict:
    # A plain `def` (not `async def`) is deliberate: ask() is slow and
    # blocking, and FastAPI runs plain-def endpoints in a thread pool so
    # one slow question doesn't freeze the whole server.
    expected = os.environ.get("LEDGER_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key header.")

    with _usage_lock:
        if _usage["day"] != date.today():                  # a new day: reset the counter
            _usage.update(day=date.today(), runs=0)
        if _usage["runs"] >= MAX_RUNS_PER_DAY:
            raise HTTPException(status_code=429, detail="Daily question limit reached.")
        _usage["runs"] += 1

    return ask(request.question, tags=["api"])
