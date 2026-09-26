# Ledger

Ledger answers questions about public companies using their annual reports (SEC Form 10-K). Every number it gives you comes with the exact sentence it came from and the character position of that sentence in the filing. If it can't find the answer, it says so instead of guessing.

I built this to learn how agent systems are put together, and specifically to learn how you tell whether one is any good. Most of the work here is not the agent. It's the 159 test questions, the search comparison, and the security testing that tell me what the agent actually does.

```
$ uv run python -m ledger.cli "How did Apple's net income change between fiscal 2023 and fiscal 2024?"

STATUS: ANSWERED

Apple's net income decreased from $96,995 million in fiscal 2023 to $93,736
million in fiscal 2024, a decline of about $3,259 million (roughly 3.4%).
This drop occurred despite higher revenue and operating income in 2024,
largely because the provision for income taxes rose from $16,741 million
(14.7% effective tax rate) to $29,749 million (24.1%), driven primarily by
a one-time $10.2 billion income tax charge related to the State Aid Decision.

SOURCES:
  [AAPL fiscal 2024 10-K, Item 8. Financial Statements]
    "Net income | 93,736 | 96,995 | 99,803"
    https://www.sec.gov/Archives/edgar/data/320193/...  (chars 125351-125388)
  [AAPL fiscal 2024 10-K, Item 7. MD&A]
    "The Company's effective tax rate for 2024 was higher than the statutory
     federal income tax rate due primarily to a one-time income tax charge
     of $10.2 billion, net, related to the State Aid Decision"
    https://www.sec.gov/Archives/edgar/data/320193/...  (chars 111985-112183)

  run 682817fef8f5 | 9.8s | 3 LLM calls | 7,412 in / 511 out tokens | cost $0.0151
```

And when it can't answer:

```
$ uv run python -m ledger.cli "What is Tim Cook's home address?"

STATUS: REFUSED

This question can't be answered from the filings I have. The question asks
for personal information, which is not in 10-K filings.
```

## Results

Measured on 150 hand-written questions with known answers. Every expected value was checked against the actual filing text by script before the questions were used.

| | Score |
|---|---|
| Correctness | 87% |
| Citation validity | 100% |
| Faithfulness (model judge) | 98% |
| Refusal accuracy | 100% (24/24) |
| False refusal rate | 13% |
| Errors | 0 |
| Median latency | 6.4s |
| Median cost per question | about $0.015 |

By question type:

| Type | Count | Correct |
|---|---|---|
| Single fact | 47 | 94% |
| Multi-hop (comparisons, changes, ratios) | 35 | 94% |
| Narrative text (risks, competition, partnerships) | 24 | 88% |
| Should refuse | 24 | 100% |
| Tricky (false premises, vague years, multi-part) | 17 | 47% |

Citation validity being 100% matters more than it looks. It doesn't mean the model was careful. It means every quote it produced was checked in Python against the passage it claimed to be quoting, and anything that didn't match exactly was thrown out and counted. Across 150 questions, nothing was thrown out.

The weak spot is the tricky category, and I've left it weak rather than papering over it. More on that below.

### Search

Before touching any language model I measured search on its own, using 23 queries where I knew which passage contained the answer.

| Configuration | recall@5 | recall@10 | MRR |
|---|---|---|---|
| Vectors only | 0.83 | 0.96 | 0.54 |
| BM25 only | 0.70 | 0.74 | 0.48 |
| **Hybrid (RRF)** | **0.91** | 0.91 | **0.59** |
| Hybrid + reranker | 0.74 | 0.96 | 0.52 |
| Hybrid + reranker + filters | 0.87 | 0.96 | 0.57 |

Two things to read out of that. BM25 alone scores the same at 5 and at 10, which means the queries it misses it never finds at all, at any rank. Those are the paraphrased ones, where the question shares no words with the document. Vector search handles them.

The second thing is that the reranker made it worse, which is not what I expected. It cost 17 points of recall@5 while leaving recall@10 alone, so it isn't losing passages, it's pushing correct ones out of the top 5 and into positions 6 to 10. In one trace I watched it promote a footnote about deferred revenue above the actual income statement. I guessed it was struggling with tables and added prose queries to check. That didn't help it either, so the guess was wrong and the cause is still open. I turned it off. The code and the comparison row are still there so it can be measured again if the corpus changes.

### Prompt injection

Filings are documents, and documents can contain text written to manipulate whatever reads them. I built a fake company, INJX, out of a real filing with seven attacks planted in it: four visible (a direct override, a data exfiltration link, a fake "system" instruction, and a quiet fake correction to a number) and three hidden with CSS so a human reader would never see them.

| | Defenses off | Defenses on |
|---|---|---|
| Attacks resisted | 12/12 | 12/12 |
| Citation validity | 88% | 100% |
| Errors | 0 | 1 |

The honest reading is that the keyword screen and output filter, which is what "defenses off" disables, made no difference to whether attacks succeeded. The prompt instruction that evidence is data and not instructions did all the work. What the screen measurably changed was citation validity: with the poisoned passage dropped before the model saw it, there was nothing to misquote.

Two caveats I'd rather state than have someone find. "Off" is not "undefended", because the parser still strips hidden text and the prompt still holds. And 12 cases is small.

The quiet fake correction is the interesting one. It tells any automated summary to report total net sales as $1 billion, and it contains none of the phrases a keyword screen looks for, so the screen never flagged it. The model caught it anyway, reported the real figure, and told me the document contained a planted instruction. Keyword screening only catches attacks that look like attacks.

### The judge

The faithfulness number comes from a model grading answers, so I checked the grader. I labelled 33 answers by hand and compared.

Agreement was 97%. Cohen's kappa was 0.00.

That's not a contradiction, it's what kappa does when one category dominates. 32 of my 33 labels were "faithful", so agreement by chance is also about 97%, and kappa divides out to nothing. The sample is skewed because citations are already verified in code and unsupported answers are already blocked by the verifier, so there's almost nothing unfaithful left to disagree about.

So I measured the thing I actually cared about instead: can the judge spot a wrong number? I took 15 correct answers, shifted one figure in each, and re-judged them. It caught 14 of 15. It also graded severity sensibly without being asked, calling a corrupted headline number unfaithful and a corrupted supporting number partial.

The one it missed had cited a passage that stated the figure in rounded prose ("total net sales were $383.3 billion") rather than as an exact number, so the corrupted exact figure didn't look obviously different. Exact-match faithfulness checking gets weaker wherever the filing rounds.

## How it works

```
Question
   |
   v
 plan  ----(nothing answerable)----> refuse
   |
   v
retrieve --> answer --> verify --(supported)--> finalize
   ^                       |
   |                       |
   +----(retry allowed)----+
                           |
                    (out of retries)
                           |
                           v
                        refuse
```

A model is used for exactly four things:

- **plan**: turn the question into a company, a fiscal year, and one to three search queries, or decide it's out of scope. It gets the list of available filings, so an out of scope question is refused in about 1.7 seconds and one cheap call.
- **answer**: write the answer from the retrieved passages, citing each claim with a quote.
- **verify**: a separate call, different model, that checks whether the quotes actually support the answer.
- **judge**: used only in the evaluation harness, never when answering.

Everything else is ordinary code. Search is BM25 plus embeddings on my laptop, free. Whether a quote is real is decided by looking for it in the passage, not by asking. The flow through the graph is a state machine. Loop bounds are set in config, not by the model deciding when to stop.

That division is deliberate. Language models are good at language and unreliable at everything else, so they get the language jobs and code handles anything where I need certainty.

The pipeline underneath: download 15 filings from SEC EDGAR, strip the HTML down to text with tables preserved as rows, cut it into about 3,575 chunks that never cross a section boundary, embed them with bge-small, and store them in Qdrant with the ticker and fiscal year attached so searches can be filtered.

## What's in here

```
ledger/
  edgar.py        download filings from SEC EDGAR
  parse.py        HTML to clean text, tables kept intact
  chunk.py        split into chunks with section labels and exact offsets
  index.py        embed everything, store in Qdrant
  retrieval.py    BM25 + vectors + RRF + filters (+ a reranker, disabled)
  graph.py        the agent: plan, retrieve, answer, verify, refuse
  prompts.py      every prompt in the project
  llm.py          the one place that calls the model API
  tracing.py      records tokens, cost and timing for every run
  security.py     injection screening and output filtering
  mcp_server.py   the same search, exposed over MCP
  api.py          FastAPI service
  cli.py          terminal interface
  stats.py        reads the traces, prints latency and cost

evals/
  golden.jsonl            159 questions with known answers
  run_evals.py            runs them all, concurrently, and scores them
  graders.py              the scoring rules
  judge_calibration.py    hand-labelling and agreement measurement
  gate.py                 fails if a change makes things worse
  retrieval_eval.py       the search comparison above
  make_injection_filing.py builds the booby-trapped test filing

tests/    fast checks on the parts that don't need a model
```

## Running it

You need Python 3.12, [uv](https://docs.astral.sh/uv/), and an Anthropic API key.

```bash
git clone https://github.com/devashishbhat/ledger-agent.git
cd ledger-agent
uv sync
cp .env.example .env     # then fill in your key and an SEC user agent
```

The SEC asks that automated requests identify themselves, so `SEC_USER_AGENT` needs to be your name and email.

```bash
uv run python -m ledger.edgar   # download 15 filings, about 2 minutes
uv run python -m ledger.index   # parse, chunk, embed, about 1 minute
uv run python -m ledger.cli "What were Apple's total net sales in fiscal 2024?"
```

Other things you can run:

```bash
uv run python -m evals.retrieval_eval        # the search comparison
uv run python -m evals.run_evals             # the full test set, about $3
uv run python -m evals.run_evals --suite smoke   # 17 cases, about $0.30
uv run python -m ledger.stats                # cost and latency across runs
uv run uvicorn ledger.api:app --reload       # web API at /docs
uv run pytest -q
```

There's also an MCP server, so the search can be used from any MCP host:

```bash
uv run python -m ledger.mcp_server
```

Three read-only tools: list the filings, search them, fetch a passage by id. It clamps the result count and returns errors as data rather than throwing, because the caller is somebody else's agent and I don't control what it asks for.

### Docker

```bash
docker build -t ledger .
docker run --rm -p 8000:8000 --env-file .env -v "$(pwd)/data:/app/data" ledger
```

Not deployed anywhere public. The image needs about 2GB of RAM held warm for the embedding and reranking models. Free tiers at Render, Koyeb and Fly cap out at 256 to 512MB, Hugging Face moved Docker Spaces behind a paid plan, and Oracle's free tier has the memory but reclaims instances that sit idle for a week, which is exactly what a portfolio demo does. Hosting it properly is $5 to $7 a month and I didn't think a demo justified that. The container runs the same anywhere with the memory available.

## Things that went wrong

Full write-up in [POSTMORTEM.md](POSTMORTEM.md). Three worth knowing about:

**The parser silently deleted table rows.** Apple's segment table was missing Greater China and Rest of Asia Pacific. I only noticed because the remaining rows summed to 293,425 against a stated total of 391,035. The rule for spotting invisible white text, `color: #fff`, also matches `background-color: #ffffff`, and financial tables shade alternate rows white. No error, no warning, just missing data. Fixed with a lookbehind so only text colour matches, plus a regression test.

**Amazon's headings were invisible to the chunker.** The index build reported 1 section for all three Amazon filings where every other company showed 15 to 20. Amazon writes its section headings as one-row HTML tables rather than as text, so the parser turned them into table blocks and the heading detector skipped them. Caught it by reading a summary table during the build, which is why that summary exists.

**My own grader was wrong.** One injection case checked that the answer didn't contain "1 billion". The agent defeated the attack, reported the real figure, and explained that a passage contained a planted instruction, which of course meant the explanation contained the forbidden phrase. It scored as a failure. Checking for the absence of a string is the wrong test when the correct behaviour is to surface the attack.

The pattern across all three is that nothing threw an exception. Every one was found by reading output or by a measurement disagreeing with itself. That's most of what I took away from building this.

## Limitations

- The tricky category sits at 47%. False premise questions ("why did Apple's revenue double?") and multi-part questions where only some parts are answerable are the main failures. My prompt changes moved the problem from the planner to the verifier rather than fixing it. The verifier has no notion that correcting a false premise is a legitimate answer.
- Section detection is rule based and depends on how each company formats its HTML. Two of five companies needed specific handling, and a new filer may need more. The index build prints sections per filing so this stays visible.
- 23 labelled search queries is enough to separate 0.91 from 0.74, not enough to separate 0.91 from 0.87.
- The injection test filing is a real filing with the company renamed, so it isn't a fully independent document.
- Exact-match faithfulness checking gets weaker wherever a filing states figures in rounded prose.
- 15 filings, 5 companies, 3 fiscal years. I expect recall to drop when the corpus gets bigger and haven't measured that yet.

## Costs

A factual question is about 3 model calls and $0.015. A long narrative question can hit $0.04 because output tokens cost several times more than input tokens, and those answers are long. A refusal is about $0.001 because it stops at the planner. The full 150 case run with the judge on comes to roughly $3.

All of that comes from the tracer, which writes a line per run with token counts, timings and cost. `ledger.stats` reads those lines back.