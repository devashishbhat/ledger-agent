"""
ledger/parse.py — Stage 3: turn a 10-K's messy HTML into clean plain text.

The output is a single string where:
  - paragraphs are separated by a blank line ("\n\n")
  - each table becomes one block:
        [TABLE]
        Total net sales | 391,035 | 383,285 | 394,328
        ...
        [/TABLE]

Why we care so much: the model can only answer from text we give it.
Garbled tables = wrong numbers = wrong answers, no matter how good the AI is.
"""
import re
import warnings

from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning

# Ignore XMLParsedAsHTMLWarning from lxml
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Tags that start a new line/paragraph when a browser displays them.
BLOCK_TAGS = ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6",
              "section", "article", "blockquote", "pre", "tr"]

# CSS that makes text invisible to a human reader. Hidden text is a classic
# place to hide instructions meant for an AI (prompt injection), and real
# 10-Ks also contain hidden machine-readable data we don't want.
HIDDEN_STYLE = re.compile(
    r"display\s*:\s*none"
    r"|visibility\s*:\s*hidden"
    r"|font-size\s*:\s*0(?:\.0+)?(?:px|pt|em)?\s*(?:;|$)"
    # The lookbehind stops this from matching "background-color: #ffffff",
    # which financial tables use for alternating row shading. Matching it
    # would silently delete real table rows.
    r"|(?<![-\w])color\s*:\s*(?:#fff(?:fff)?|white)\b",
    re.IGNORECASE,
)

# Lines that are only a page number, e.g. "23" or "Page 23".
PAGE_NUMBER = re.compile(r"^(?:page\s+)?\d{1,3}$", re.IGNORECASE)
# Other repeated junk lines found at the top/bottom of every page.
BOILERPLATE = {"table of contents"}

# Table cells that are just formatting (a lone "$" or ")") — drop them so a
# row reads "Total net sales | 391,035" instead of "Total net sales | $ | 391,035".
FORMATTING_CELLS = {"$", "%", ")", "(", "—", "-"}


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")   # lxml = a fast, forgiving HTML parser

    # 1. Remove things that are never content.
    for tag in soup(["script", "style", "head", "noscript"]):
        tag.decompose()                  # decompose() = delete the tag and everything inside

    # 2. Remove the hidden inline-XBRL header. Modern 10-Ks embed a block of
    #    machine-readable financial tags inside <ix:header>; it's invisible
    #    in a browser and useless (and confusing) to us.
    for tag in soup.find_all(lambda t: t.name is not None and t.name.lower() == "ix:header"):
        tag.decompose()

    # 3. Remove anything styled to be invisible (see HIDDEN_STYLE above).
    for tag in soup.find_all(style=HIDDEN_STYLE):
        tag.decompose()

    # 4. Flatten every table into rows of "cell | cell | cell".
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c and c not in FORMATTING_CELLS]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            flat = "\n\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]\n\n"
            table.replace_with(NavigableString(flat))
        else:
            table.decompose()             # empty layout tables: just remove

    # 5. Make line breaks where a browser would. <br> becomes a newline;
    #    block tags get a newline before and after. Inline tags (<span>,
    #    <b>, <a>...) get nothing, so "revenue was <b>$391</b> billion"
    #    stays one sentence.
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(BLOCK_TAGS):
        if tag.parent is not None:
            tag.insert_before("\n")
            tag.insert_after("\n")

    text = soup.get_text()
    # \xa0 is a "non-breaking space"; \u200b is an invisible "zero-width space".
    text = text.replace("\xa0", " ").replace("\u200b", "")

    # 6. Clean line by line and rebuild paragraphs.
    paragraphs: list[str] = []
    table_lines: list[str] = []
    in_table = False
    for raw_line in text.split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()   # squash runs of spaces
        if not line:
            continue
        if line == "[TABLE]":
            in_table, table_lines = True, []
            continue
        if line == "[/TABLE]":
            in_table = False
            if table_lines:
                paragraphs.append("[TABLE]\n" + "\n".join(table_lines) + "\n[/TABLE]")
            continue
        if in_table:
            table_lines.append(line)      # keep table rows together, one per line
            continue
        if PAGE_NUMBER.match(line) or line.lower() in BOILERPLATE:
            continue
        paragraphs.append(line)

    return "\n\n".join(paragraphs)
