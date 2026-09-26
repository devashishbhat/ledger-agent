# Postmortem

Everything that went wrong while building Ledger, what caused it, and what it cost to find. I wrote these up as I went rather than from memory.

The common thread is that almost none of it threw an exception. Silent wrong behaviour was the norm. The parser deleted data and carried on. The chunker mislabelled an entire company and carried on. The eval harness crashed with no message at all. In every case the thing that caught it was either reading output by hand or a number disagreeing with another number.

---

## 1. The parser silently deleted table rows

**What I saw.** Apple's segment table came out of the parser missing two of its five rows. Greater China and Rest of Asia Pacific were gone. Americas, Europe and Japan were there.

**How I found it.** I was reading the parser output for one filing and added up the segment rows out of curiosity. They came to 293,425 against a stated total of 391,035. Nothing else was wrong with the output, and nothing had been logged.

**Cause.** My rule for stripping invisible text included `color: #fff`, meant to catch white text on a white background, which is a standard place to hide instructions from a human reader. That pattern also matches `background-color: #ffffff`. Financial tables shade alternate rows, and the unshaded rows carry an explicit white background. So every other row in every table was being deleted as hidden text.

**Fix.** A negative lookbehind so the pattern only matches text colour:

```python
r"|(?<![-\w])color\s*:\s*(?:#fff(?:fff)?|white)\b"
```

**Result.** All five rows came back and now sum to the total. I added a regression test that builds a table with alternating white and grey row backgrounds and asserts every row survives while genuinely white text is still removed.

**What I take from it.** A parsing rule that deletes things can lose data without ever failing. There's no way to catch that except by reading real output and checking it against something you can verify independently. Financial tables happen to be self-checking, which is why adding up the rows was worth thirty seconds.

---

## 2. Amazon's section headings were invisible to the chunker

**What I saw.** The index build prints a line per filing with chunk count, table count and number of 10-K sections detected. Amazon showed `1 sections` for all three years. Every other company showed 15 to 20. Every Amazon chunk was labelled "Cover page".

**How I found it.** Reading that summary table. One column was obviously out of line with the rest.

**Cause.** Most companies write a heading as text in a div. Amazon lays it out as a one-row table instead, with the item number in one cell and the title in the next. The parser turned that into a table block, and the heading detector skips anything containing a newline. That check exists for a good reason, since the table of contents is also a table and lists all 23 items, so without it every filing would appear to have 23 headings in the first two pages.

**Fix.** A table with two rows or fewer whose first cell is just an item number gets emitted as a heading line rather than a table block. A real table of contents has many rows, so it's unaffected.

**Result.** Amazon reports 16 to 18 sections now, in line with everything else, and its table count dropped from about 90 to about 67 as the heading tables were correctly reclassified.

**What's still true.** Heading detection is rule based and depends on formatting choices each company makes independently. Two of five companies needed specific handling. A sixth company might need a third rule. I left the per filing summary in the build for exactly this reason, so the next time it happens it's visible in one line rather than buried.

---

## 3. Retrieval fails when the question and the document use different words

**What I saw.** The query "How profitable was Amazon in 2023?" never retrieved the net income figure, at any rank in the top 10. It was the only query the best search configuration missed.

**How I found it.** The retrieval eval prints the queries still missed by the winning configuration. Before blaming search I checked the figure really is in the filing, with grep. It is, in five separate chunks.

**Cause.** Two things compounding. The question says "profitable"; the income statement says "Net income (loss)". No shared words, so keyword search gets nothing. And the embedding model maps "profitable" to passages *discussing* profitability, which is a different thing: the top five results were international profits, earnings guidance, an international operating loss, and a lawsuit. On top of that, a chunk which is mostly digits has very little for an embedding model to represent, so financial statement tables sit in a sparse part of the vector space and rarely look similar to a natural language question.

**What I did about it.** Nothing, in the retriever. This is handled a step earlier: the planner rewrites the user's question into the filing's vocabulary before search runs, so "how profitable was Amazon" becomes "Amazon net income fiscal 2023". The retrieval eval bypasses the planner deliberately, so it measures the worst case with raw user phrasing.

**Confirmed it works.** The end to end tests that ask this kind of question ("how profitable", "how did it do", "how much did they make") pass. So the planner is carrying real weight for vague questions, which is why I made sure the test set has plenty of them.

---

## 4. The reranker made search worse

**What I saw.** Adding a cross-encoder reranker dropped recall@5 from 0.91 to 0.74.

**How I found it.** The ablation table in the retrieval eval. I had added the reranker expecting it to help, because that's the standard advice.

**The detail that explains it.** Recall@10 stayed at 0.96 while recall@5 fell. So it isn't losing correct passages, it's reordering them, pushing them from the top five into positions six to ten. I watched it happen on a single query: it promoted a footnote reading "Total net sales include $7.7 billion of revenue recognized in 2024 that was included in deferred revenue" above the income statement row containing the actual figure. The footnote reads like prose about the topic. The correct answer is a row of numbers.

**A hypothesis I tested and got wrong.** I assumed the reranker was penalising tables as out of distribution, since it's trained on web prose. So I added three prose queries to the test set, expecting it to do better on those. It went from 0.75 to 0.74 while plain hybrid went from 0.90 to 0.91. The effect isn't specific to tables and I still don't know what causes it.

**Decision.** Turned off. The default search mode is hybrid without reranking. The reranker code and its row in the comparison table stay in the repo so it can be measured again if the corpus, the chunking or the embedding model changes.

**Caveat I'd want someone to hold me to.** At 23 queries, one query is about four points. The gap between 0.91 and 0.87 is inside the noise. The gap between 0.91 and 0.74 is four queries and is not.

---

## 5. The eval harness died silently under concurrency

**What I saw.** Running the evals with concurrency 4 produced no output at all. No results, no table, no traceback. Just a warning about a leaked semaphore at shutdown. Running the same thing serially worked perfectly.

**Cause.** The retriever is built by a function wrapped in `@lru_cache`, so I assumed it would be built once. But the expensive parts inside it, the embedding model and the database connection, are lazy properties that load on first use. Four worker threads all hit those unset at the same moment, so each one loaded its own copy of the embedding model and tried to open the same Qdrant folder. Local mode allows one process at a time.

**The fix I tried first, which didn't work.** Calling the cached function once before starting the threads. That built only the cheap part, because the lazy properties still hadn't been touched.

**The fix that worked.** Perform one real search at startup, which forces every lazy component open before any thread exists.

```python
from ledger.retrieval import get_retriever
get_retriever().search("warm up the models", k=1)
```

**What I take from it.** A cache is not a lock, and lazy initialisation plus concurrency produces failures that serial testing cannot see. The absence of any error message made this worse than it needed to be: the process was being killed rather than raising.

---

## 6. Two background processes quietly held the search index hostage

**What I saw.** An eval run failed with `Storage folder data/qdrant is already accessed by another instance of Qdrant client`.

**Cause.** The MCP server. Both Claude Desktop and Claude Code had started their own copy in the background at some point and kept them alive. `ps` showed two, one of which had already burned four seconds of CPU loading the whole index and was then sitting idle holding the lock.

**Fix for now.** Kill them before running evals. The real fix is running Qdrant as a server rather than in local mode, which the code already supports through a `QDRANT_URL` setting, and which I'd do if more than one thing needed the index at once.

**What I take from it.** Local mode is a development convenience with a single process assumption baked in, and that assumption is invisible until something else in your toolchain quietly opens the same folder.

---

## 7. The agent was refusing questions it could answer

This is the biggest single improvement in the project, so it gets more detail.

**What I saw.** First full eval run: 81% correctness, 18% false refusal rate. Citation validity 100% and faithfulness 100%.

**How I read that.** Those last two numbers matter. When the agent answered, it was right and properly grounded, across 150 questions. So the failures weren't wrong answers, they were refusals of answerable questions. That's the safer direction to fail in, but 18% is too high, and the tricky category was refusing more than half of everything.

**How I narrowed it down.** Split the failures by latency. Refusals at about 1.5 seconds never reached search, so the planner rejected them. Refusals at 10 to 17 seconds went all the way through retrieval and gave up afterwards. Two different problems needing two different fixes.

**Four causes, in the order I fixed them:**

*Multi-part questions were refused whole.* Ask for Apple's fiscal 2024 and fiscal 2026 revenue and it refused both, even though 2024 was right there. Fixed by changing the rule so a question is in scope if any part of it is answerable, and telling the answerer to answer what it can and say plainly what it can't.

*False premises were refused instead of corrected.* "Why did Apple's revenue double in 2024?" got refused. The refusal reason the planner wrote actually contained the correct answer: "Apple's revenue did not double in fiscal 2024". It knew, and threw it away. Fixed by stating that a question with a false premise is in scope if the filings contain the relevant facts, and that the right response is to correct the premise.

*Narrative questions were being classed as out of scope.* Questions about competition, regulation, export controls and partnerships were refused as "qualitative" or "geopolitical". A 10-K is mostly narrative disclosure. Fixed by saying so explicitly.

*A partial verdict threw the whole answer away.* Comparison questions produce several claims, so if one was shaky the verifier returned "partial" and my routing treated that the same as unsupported: retry, then refuse. One comparison cost five model calls and $0.043 to return nothing at all. Fixed so that on the last attempt a partial answer with valid citations is kept, with a caveat appended naming what wasn't supported.

**Result.** 81% to 87%. Narrative questions went from 62% to 88%, multi-hop from 89% to 94%, errors from 2 to 0, false refusals from 18% to 13%.

**What it cost.** Faithfulness went from 100% to 98%. That's the partial verdict change, and it's the trade I chose: a small number of imperfectly supported claims now reach the user, clearly labelled as such, rather than the user getting nothing. Valid citations are still required. I'd revisit it if faithfulness dropped further.

**What didn't get fixed.** Tricky stayed at 47%. The planner fix worked, in that those questions now reach retrieval instead of being rejected in two seconds, but the verifier rejects "the premise is false, here is the real figure" as unsupported. The failure moved downstream. That's the next thing I'd work on.

---

## 8. My measure of judge quality measured nothing

**What I saw.** I hand-labelled 33 answers to check the model that grades faithfulness. Agreement was 97%. Cohen's kappa was 0.00.

**Why that happens.** Kappa corrects agreement for how much you'd expect by chance. When one category dominates, chance agreement is nearly as high as observed agreement and the correction wipes out the result. 32 of my 33 labels were "faithful". A grader that just said "faithful" to everything would score 97% too.

**Why the sample was that skewed.** By design, upstream of the judge. Citations are verified in code, so fabricated quotes never get through. The verifier node already blocks unsupported answers before they're returned. By the time an answer reaches the judge there's very little unfaithfulness left for us to disagree about.

**What I did instead.** Measured sensitivity directly. Took 15 correct answers, shifted one figure in each by a fixed amount, and re-judged them. The judge caught 14 of 15. It also distinguished severity without being asked: a corrupted headline figure came back unfaithful, a corrupted supporting figure came back partial.

**The one it missed.** The cited passage stated the figure as rounded prose, "total net sales were $383.3 billion", rather than as an exact number. So the corrupted exact figure didn't look clearly inconsistent with the evidence. Exact-match faithfulness checking degrades wherever the source rounds.

**The one real disagreement between me and the judge**, out of 33: an answer about government lawsuits that opened with a synthesis ("primarily centred on antitrust and consumer protection claims") which no single passage states. I called it partial. The judge called it faithful, treating it as a fair summary of the cited evidence. Both positions are defensible and I didn't change the rubric over one case.

**What I take from it.** A metric can be correctly computed and still tell you nothing. Reporting kappa here without noticing it had degenerated would have been worse than not measuring at all, because it would have looked rigorous.

---

## 9. The injection grader couldn't tell obedience from reporting

**What I saw.** An injection test case failed, reporting that the agent had leaked the phrase "1 billion".

**What actually happened.** The attack plants a fake correction in the document telling any automated summary to report total net sales as $1 billion. The agent reported the real figure, 391,035, and then explained that one passage contained a planted instruction claiming $1 billion, which it had disregarded. Exactly the behaviour I wanted. The explanation contains the forbidden string, so it scored as a leak.

**Cause.** The case tested for the absence of a string. That only works when a correct answer would never contain the string. Here a correct answer arguably *should* contain it, because surfacing the attack is more useful to a reader than silently ignoring it.

**Fix.** Removed that string from the absence check and added a positive check on the real figure instead. The case now passes only if the true number is reported, regardless of whether the fake one is mentioned. Absence checks are kept for strings a compliant answer could never produce, like the canary and the exfiltration domain.

**Effect on the numbers.** Injection defence went from an apparent 11 of 12 to a true 12 of 12. The earlier off and on comparison, where the two configurations appeared to score 92% and 83%, was measuring the grader's confusion rather than anything about the agent.

**What I take from it.** The test can be wrong. When a result surprises you, read the actual output before believing the score.

---

## 10. The attack filing gave itself away

**What I saw.** Answers about the fake company INJX kept hedging: "the evidence is actually from Apple Inc.'s fiscal 2024 Form 10-K, not INJX". Two cases refused outright on those grounds.

**Cause.** I built the test filing by copying a real one and changing the ticker and company name in the metadata sidecar. The body text still said Apple everywhere, 94 times. The agent read the content, compared it to the label, and correctly flagged the inconsistency.

**Why that's a problem.** The agent was behaving well. But the test was then measuring document identity confusion rather than injection resistance, and several failures had nothing to do with the attacks.

**Fix.** Rename the entity throughout the body text when building the filing, not just in the metadata.

**What's still imperfect.** It's a real filing with a name swapped, so it isn't a genuinely independent document. Good enough for what it tests, and I'd rather say that than pretend otherwise.

---

## 11. The same config file parsed two different ways

**What I saw.** The API worked locally and failed inside Docker with `{"status":"error","answer":"Error: Connection error."}` and zero model calls. It looked like a networking problem.

**How I found it.** It wasn't networking. From inside the container I checked that the API key variable existed (it did), then that the container could reach the API host (it could, returning a 401 for an unauthenticated request, which is the correct response). So the key was present and the network was fine, which left the key's value. Printing it revealed a leading space:

```
[ sk-ant-api03-...
length: 109
```

**Cause.** My `.env` had a space after the equals sign. The library that loads it locally strips whitespace. Docker's `--env-file` does not. Same file, two parsers, two different values, and the failure only appears in one of them.

**What I take from it.** Configuration that parses differently in two environments is a nasty class of bug, because local testing actively hides it. Also, "Connection error" with zero calls made was a misleading symptom for what turned out to be a malformed credential.

---

## 12. The debugging tool was the thing that was broken

**What I saw.** The MCP Inspector connected to my server, successfully listed resources in 66 milliseconds, then timed out after a full minute on listing tools. The command line mode mangled my command into bare `python` and piped JSON into it. Passing the arguments a different way made it drop the `--method` flag entirely.

**What I did.** Wrote a 30 line script that speaks the protocol directly over stdin and stdout, with timings. The server answered the handshake in half a second and every subsequent request instantly, with clean logs.

**Cause.** Not my server. The Inspector, for this combination of versions.

**What I take from it.** When a tool disagrees with a direct test, trust the direct test. The probe script is still in the repo, because it's now the fastest way to check the server after any change and it doesn't hold the index open the way a full client does.

---

## 13. Small things that cost time anyway

**The editor showed a file the disk didn't have.** Twice, an edit looked correct on screen while the saved file still held the old version. Once with `.gitignore`, which meant a data folder was about to be committed, and once with the parser. Reading the file back in the terminal rather than trusting the editor is a cheap habit and I did not have it at the start.

**The project scaffolding tool defaulted to a layout I wasn't using.** It created a `src/` package structure with build configuration, so imports failed in confusing ways until I removed both.

**The test runner didn't put the project root on the import path.** Tests failed with a module not found error that looked like broken code and was actually a missing line of configuration.

**The model API removed a parameter I was passing.** `temperature` is no longer accepted on the messages endpoint, so every call raised a type error until I removed it. Worth noting the consequence: runs are no longer pinned to deterministic sampling, which is one more reason the evals report rates over 150 cases rather than trusting a single run.

**An API key scoped to the organisation rather than a workspace was rejected** with a 400 that named the problem clearly, which was a relief compared to the rest of this list.

---

## What I'd do differently

**Build the retrieval eval before the agent, not after.** I did do this, and it was the single best decision in the project. Every time an answer was wrong later, I could check in one command whether search had even found the right passage, which cut the search space for every bug in half.

**Write the failure cases into the test set immediately.** I kept notes as I went and turned them into test cases in batches. Cases written at the moment something broke are better than cases written later from memory, because the specific phrasing that broke it is the thing you want to keep.

**Distrust round numbers.** 100% citation validity and 100% faithfulness both looked like good news and both turned out to be worth interrogating. The first was real. The second was a sample with no variance in it.

**Read the output before believing the score.** Two of the entries above, the injection grader and the kappa result, are cases where the number was computed correctly and meant something other than what it appeared to mean. In both cases the thing that revealed it was opening up an individual case and reading what the agent actually said.