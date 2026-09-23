"""
ledger/security.py — Stage 12: prompt-injection defenses.

Two cheap, deterministic layers (they don't call any AI model):

1. screen:   before evidence reaches the model, flag passages that look
             like instructions aimed at an AI, and drop them.
2. sanitize: after the model answers, remove images and links to any
             site other than sec.gov (a classic way to leak data is to
             make the AI output an image URL that carries the data).

Neither is perfect. That's why the prompt also tells the model to treat
evidence as data, and why we MEASURE how often attacks still get through.
"""
import re

# Phrases that almost never appear in a real 10-K but often appear in
# injection attempts. Deliberately narrow: a pattern like "AI systems"
# would flag half of NVIDIA's legitimate filing.
INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|above|earlier|preceding)\s+instructions",
    r"disregard\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\b",
    r"\bsystem\s+prompt\b",
    r"\bnote\s+to\s+(?:ai|llm|language\s+model|assistant)s?\b",
    r"\b(?:ai|llm|language\s+model)\s+(?:assistants?|systems?|models?)\s+(?:must|should|are\s+instructed)\b",
    r"\byou\s+are\s+now\b",
    r"\bnew\s+instructions?\s*:",
    r"!\[[^\]]*\]\(\s*https?://",          # a markdown image with a URL
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


# Markdown images: ![alt](url)
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# Any http(s) URL
_URL = re.compile(r"https?://[^\s)\]]+")


def sanitize_answer(answer: str) -> str:
    answer = _MARKDOWN_IMAGE.sub("[image removed]", answer)

    def keep_only_sec(match: re.Match) -> str:
        url = match.group(0)
        return url if re.match(r"https?://(?:www\.)?sec\.gov/", url) else "[link removed]"

    return _URL.sub(keep_only_sec, answer)
