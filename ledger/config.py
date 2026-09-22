"""
ledger/config.py — every setting for the project, in one place.
Other files do `from ledger import config` and then read `config.SOMETHING`.
If you ever want to change a model, a folder, or a limit, change it HERE,
not scattered across the code.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
# Read the .env file in the project root and copy its values into
# os.environ. This is how secrets (API keys) reach the program without
# ever being written into the code itself.
load_dotenv()
# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
# __file__ is this file's own path: .../ledger-agent/ledger/config.py
# .parent is the ledger/ folder, .parent again is the project root.
# Building paths from here means the code works no matter which folder
# you happen to run it from.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" # downloaded 10-K HTML files
TEXT_DIR = DATA_DIR / "text" # cleaned plain text of each filing
CHUNKS_PATH = DATA_DIR / "chunks.jsonl" # every chunk, one JSON object per line
QDRANT_PATH = DATA_DIR / "qdrant" # the vector database's files
TRACES_DIR = DATA_DIR / "traces" # one line per agent run (cost, timing)
REPORTS_DIR = DATA_DIR / "reports" # reports saved after human approval
EVALS_DIR = ROOT / "evals"
RESULTS_DIR = EVALS_DIR / "results" # output of every eval run
# ---------------------------------------------------------------------------
# SEC EDGAR (where the filings come from)
# ---------------------------------------------------------------------------
# The SEC requires every automated request to identify who is making it,
# in the form "Your Name your@email.com". Requests without it get blocked.
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "")
# Which companies and fiscal years to download. 5 companies x 3 years
# = 15 filings. Start small; you can add more later.
COMPANIES = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]
FISCAL_YEARS = [2022, 2023, 2024]
# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_MAX_WORDS = 250 # a chunk is closed once it would exceed this
CHUNK_MIN_CHARS = 200 # chunks shorter than this are thrown away (noise)
# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
# Embedding model: turns text into 384 numbers. Runs locally, free.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# This particular model was trained to expect this prefix on QUERIES
# (not on documents). Leaving it off makes search measurably worse.
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Reranker: reads (question, passage) pairs together and scores them.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QDRANT_COLLECTION = "filings"
# If you run Qdrant as a separate server (Docker), put its URL in .env,
# e.g. QDRANT_URL=http://localhost:6333 . Otherwise a local folder is used.
QDRANT_URL = os.environ.get("QDRANT_URL")
RETRIEVE_CANDIDATES = 30 # vector search and BM25 each return this many
EVIDENCE_PER_QUERY = 6 # after fusion + reranking, keep this many
MAX_EVIDENCE = 12 # never show the model more than this many passages
# ---------------------------------------------------------------------------
# Language models (Anthropic)
# ---------------------------------------------------------------------------
# Model names change over time. Check https://docs.claude.com for the
# current names and override them in .env if these stop working.
MAIN_MODEL = os.environ.get("LEDGER_MAIN_MODEL", "claude-sonnet-5")
FAST_MODEL = os.environ.get("LEDGER_FAST_MODEL", "claude-haiku-4-5-20251001")
# The eval judge should differ from the model that WRITES answers (MAIN),
# so it isn't grading its own homework.
JUDGE_MODEL = os.environ.get("LEDGER_JUDGE_MODEL", FAST_MODEL)
# Prices in US dollars per MILLION tokens: (input price, output price).
# FILL THESE IN from Anthropic's pricing page. While they are None, cost
# is reported as "unknown" instead of a made-up number.
PRICES_PER_MTOK: dict[str, tuple[float | None, float | None]] = {
MAIN_MODEL: (None, None),
FAST_MODEL: (None, None),
}
# How many times the agent may search before giving up (first try + retries).
MAX_ATTEMPTS = 2
# Prompt-injection defenses. Set LEDGER_DEFENSES=off in .env to measure
# how the agent behaves WITHOUT them (you will do this in Stage 12).
DEFENSES_ENABLED = os.environ.get("LEDGER_DEFENSES", "on").lower() != "off"