"""
ledger/chunk.py — Stage 4: split a filing's clean text into chunks.

A "chunk" is a piece of a document small enough to search and to show
the model. Every chunk remembers:
  - which filing it came from (ticker, fiscal year)
  - which 10-K section it's in ("Item 7. MD&A", ...)
  - exactly where it sits in the full text (char_start, char_end)

That last part is what lets us cite an exact span of the original
document later, instead of just "somewhere in Apple's 10-K".
"""
import re
from dataclasses import asdict, dataclass

from ledger import config

# Standard 10-K sections. Every 10-K uses these item numbers.
ITEM_NAMES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity", "2": "Properties", "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures", "5": "Market for Common Equity",
    "6": "Reserved", "7": "Management's Discussion and Analysis (MD&A)",
    "7A": "Market Risk Disclosures", "8": "Financial Statements",
    "9": "Changes in and Disagreements with Accountants", "9A": "Controls and Procedures",
    "9B": "Other Information", "9C": "Foreign Jurisdiction Disclosure",
    "10": "Directors, Executive Officers and Governance", "11": "Executive Compensation",
    "12": "Security Ownership", "13": "Relationships and Related Transactions",
    "14": "Principal Accountant Fees", "15": "Exhibits", "16": "Form 10-K Summary",
}

# Matches heading lines like "Item 7. Management's Discussion..." or
# "PART II Item 5." — case-insensitive. Group 1 captures "7", "1A", etc.
HEADING = re.compile(
    r"^(?:part\s+[iv]+\s*[.,\-–—]?\s*)?item\s+(\d{1,2}[abc]?)\b",
    re.IGNORECASE,
)

# A paragraph = one or more non-empty lines, separated from the next
# paragraph by a blank line. (Table blocks have single newlines inside,
# so a whole table counts as one paragraph.)
PARAGRAPH = re.compile(r"[^\n]+(?:\n[^\n]+)*")


@dataclass
class Chunk:
    chunk_id: str      # e.g. "AAPL-2024-0042"
    ticker: str
    company: str
    fiscal_year: int
    form: str
    section: str
    text: str
    char_start: int    # position of the first character in the filing's full text
    char_end: int      # position just after the last character
    source_url: str

    def to_dict(self) -> dict:
        return asdict(self)


def context_header(chunk: dict) -> str:
    """
    A one-line label we glue onto the front of a chunk before embedding it
    or keyword-indexing it. A chunk that just says "Net sales increased 2%"
    is ambiguous on its own; with this header the search engine also
    knows WHICH company and WHICH year it's about.
    """
    return (f"{chunk['company']} ({chunk['ticker']}) Form {chunk['form']}, "
            f"fiscal year {chunk['fiscal_year']}. Section: {chunk['section']}.")


def detect_heading(paragraph: str) -> str | None:
    """Return a section label if this paragraph is an 'Item X' heading, else None."""
    # Real headings are short single lines. Tables of contents are [TABLE]
    # blocks (they contain newlines), so they're skipped here on purpose.
    if len(paragraph) > 150 or "\n" in paragraph:
        return None
    match = HEADING.match(paragraph)
    if not match:
        return None
    number = match.group(1).upper()
    if number not in ITEM_NAMES:
        return None
    return f"Item {number}. {ITEM_NAMES[number]}"


def chunk_document(text: str, meta: dict,
                   max_words: int = config.CHUNK_MAX_WORDS,
                   min_chars: int = config.CHUNK_MIN_CHARS) -> list[Chunk]:
    """
    Walk through the paragraphs in order, packing them into chunks.
    A chunk is closed ("flushed") when:
      - the next paragraph would push it over max_words, or
      - we hit a new section heading (chunks never span two sections).
    """
    chunks: list[Chunk] = []
    section = "Cover page"
    current: list[tuple[int, int]] = []   # (start, end) of each paragraph in this chunk
    word_count = 0

    def flush() -> None:
        nonlocal current, word_count
        if current:
            start, end = current[0][0], current[-1][1]
            body = text[start:end]       # slice the ORIGINAL text, so offsets are exact
            if len(body) >= min_chars:
                chunks.append(Chunk(
                    chunk_id=f"{meta['ticker']}-{meta['fiscal_year']}-{len(chunks):04d}",
                    ticker=meta["ticker"],
                    company=meta["company"],
                    fiscal_year=int(meta["fiscal_year"]),
                    form=meta.get("form", "10-K"),
                    section=section,
                    text=body,
                    char_start=start,
                    char_end=end,
                    source_url=meta.get("source_url", ""),
                ))
        current, word_count = [], 0

    for match in PARAGRAPH.finditer(text):
        paragraph = match.group()
        heading = detect_heading(paragraph)
        if heading:
            flush()              # close the old section's chunk FIRST...
            section = heading    # ...then switch to the new section
        words = len(paragraph.split())
        if current and word_count + words > max_words:
            flush()
        current.append((match.start(), match.end()))
        word_count += words

    flush()   # don't forget the last chunk
    return chunks
