"""
ledger/edgar.py — Stage 2: download 10-K filings from SEC EDGAR.

Run it with:   uv run python -m ledger.edgar

What happens:
  1. Look up each ticker's CIK (the SEC's ID number for a company).
  2. Ask the SEC for that company's list of filings.
  3. Keep only form "10-K" (annual reports) for the years in config.
  4. Download each report's HTML into data/raw/, plus a small .json
     "sidecar" file describing it (company, year, URL, ...).
"""
import json
import time

import requests

from ledger import config

# The three SEC addresses we use. {cik:010d} means "the number, padded
# with zeros to 10 digits" — the SEC's required format (320193 -> 0000320193).
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"


def make_session() -> requests.Session:
    """A Session reuses one connection for many requests and carries our headers."""
    if "@" not in config.SEC_USER_AGENT:
        raise SystemExit(
            "SEC_USER_AGENT is missing. Add a line like this to your .env file:\n"
            '  SEC_USER_AGENT="Your Name your.email@example.com"'
        )
    session = requests.Session()
    session.headers.update({
        "User-Agent": config.SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    })
    return session


def polite_get(session: requests.Session, url: str, attempts: int = 4) -> requests.Response:
    """
    GET a URL without upsetting the SEC.

    - Sleeps 0.2s before every request (5 per second; the SEC's limit is 10).
    - Retries temporary failures (429 = "slow down", 5xx = server trouble)
      with exponential backoff: waits 1s, 2s, 4s...
    - Gives up immediately on permanent failures like 403 or 404,
      because retrying those can never succeed.
    """
    error: Exception | None = None
    for attempt in range(attempts):
        time.sleep(0.2)
        try:
            response = session.get(url, timeout=30)
        except requests.RequestException as exc:   # network problem
            error = exc
        else:                                       # we got an answer
            if response.status_code == 200:
                return response
            error = RuntimeError(f"HTTP {response.status_code} for {url}")
            if response.status_code == 403:
                raise RuntimeError(
                    "SEC returned 403 Forbidden. This almost always means the "
                    "SEC_USER_AGENT in .env is missing or doesn't look like "
                    "'Name email@domain.com'."
                )
            if response.status_code not in (429, 500, 502, 503, 504):
                raise error                          # permanent: don't retry
        wait = 2 ** attempt
        print(f"    retry {attempt + 1}/{attempts} in {wait}s ({error})")
        time.sleep(wait)
    raise error  # every attempt failed


def load_ticker_map(session: requests.Session) -> dict[str, dict]:
    """
    Download the SEC's ticker list and reshape it into
        {"AAPL": {"cik": 320193, "company": "Apple Inc."}, ...}
    The raw file looks like
        {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    """
    raw = polite_get(session, TICKERS_URL).json()
    return {
        row["ticker"].upper(): {"cik": int(row["cik_str"]), "company": row["title"]}
        for row in raw.values()
    }


def find_10k_filings(session: requests.Session, cik: int) -> list[dict]:
    """
    Return this company's 10-K filings, newest first.

    The SEC stores the list "column-wise": one list per field, where
    position i in every list belongs to filing i. So form[i], reportDate[i]
    and accessionNumber[i] all describe the same filing.
    """
    data = polite_get(session, SUBMISSIONS_URL.format(cik=cik)).json()
    recent = data["filings"]["recent"]
    filings = []
    for i, form in enumerate(recent["form"]):
        if form != "10-K":          # skip quarterly reports, amendments, etc.
            continue
        report_date = recent["reportDate"][i]   # end of the fiscal year, e.g. "2024-09-28"
        filings.append({
            "form": form,
            "accession": recent["accessionNumber"][i],   # the filing's unique ID
            "filing_date": recent["filingDate"][i],
            "report_date": report_date,
            # Companies name a fiscal year after the calendar year it ENDS in.
            # Apple's year ending 2024-09-28 is "fiscal 2024". NVIDIA's year
            # ending 2024-01-28 is also "fiscal 2024". So the year part of
            # report_date is the fiscal year.
            "fiscal_year": int(report_date[:4]),
            "primary_document": recent["primaryDocument"][i],  # the main HTML file
        })
    return filings


def download_all() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()
    print("Loading the SEC ticker list...")
    tickers = load_ticker_map(session)

    for ticker in config.COMPANIES:
        if ticker not in tickers:
            print(f"!! {ticker} not found in the SEC ticker list, skipping")
            continue
        info = tickers[ticker]
        print(f"\n{ticker} — {info['company']} (CIK {info['cik']})")
        found_years = set()

        for filing in find_10k_filings(session, info["cik"]):
            year = filing["fiscal_year"]
            if year not in config.FISCAL_YEARS or year in found_years:
                continue
            found_years.add(year)

            stem = f"{ticker}_{year}_10-K"
            html_path = config.RAW_DIR / f"{stem}.html"
            meta_path = config.RAW_DIR / f"{stem}.json"
            if html_path.exists():
                print(f"  fiscal {year}: already downloaded")
                continue

            url = ARCHIVE_URL.format(
                cik=info["cik"],
                accession=filing["accession"].replace("-", ""),  # folder name has no dashes
                document=filing["primary_document"],
            )
            print(f"  fiscal {year}: downloading {url}")
            html = polite_get(session, url).text
            html_path.write_text(html, encoding="utf-8")
            meta = {"ticker": ticker, "company": info["company"], "cik": info["cik"],
                    "source_url": url, **filing}
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"    saved {html_path.name} ({len(html) / 1e6:.1f} MB)")

        missing = set(config.FISCAL_YEARS) - found_years
        if missing:
            print(f"  note: no 10-K found for fiscal {sorted(missing)}")


if __name__ == "__main__":
    download_all()
