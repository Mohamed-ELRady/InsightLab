# AI capability roadmap

A review of where intelligence can be added to InsightLab, what it would be
worth, and in what order to build it.

## 1. What the current architecture makes easy, and what it makes hard

Three decisions in the existing design determine which features are cheap and
which are expensive.

**The model never calculates.** `analysis/` is deterministic pandas; the model
only explains results and interprets free text. Any feature that follows the
same shape — *model plans, pandas executes* — drops straight into the existing
structure. Any feature that needs the model to produce a number is fighting the
architecture and needs a verification layer first.

**Agents are generators driven by a supervisor.** Adding a stage is cheap: write
a `run()` that yields decisions. Adding a *conversation* is not, because a
generator pipeline runs forwards only. Conversation needs a second entry point
into the same state, not another stage.

**`BusinessMemory` is a flat, append-only list of sentences.** It is read into
prompts and never questioned. Making it structured, contradictable and
comparable across runs is the single highest-leverage change in the codebase,
because every other memory feature depends on it.

## 2. What is missing as a whole subsystem

Four gaps are not features but absent layers. Each is called out again in the
proposals below.

| Missing layer | Consequence today |
|---|---|
| **Conversation** | The user cannot ask a question. The collaboration the product is named for runs in one direction only. |
| **Verification** | Nothing checks that a number in a generated sentence exists in the data. One wrong figure destroys trust permanently. |
| **Semantic / entity layer** | "Customer" is a column name, not an entity. Facts cannot be attached to things, only to runs. |
| **Temporal layer** | Every run is an island. The product cannot say "down 8% since your last analysis" — the one sentence an owner most wants. |

## 3. Proposals

Grouped by family. Difficulty is engineering effort against the current
codebase. Priority is against the product thesis, not against general appeal.

---

### Family A — Conversation

#### A1. Ask Anything (conversational follow-up)

- **Problem.** The user sees "November peaked at 2.8M" and has a follow-up
  question with nowhere to put it. Every real analysis conversation is
  question-driven, and this product has no questions from the user in it.
- **How it works.** A chat panel opens on the results screen. The user types a
  question. A planner agent turns it into a **structured query plan** — target
  columns, filter, grouping, aggregation, sort, limit — as JSON. The plan is
  validated against the real schema and executed by pandas. The result is
  rendered as a small table or chart, and a second call writes one sentence of
  interpretation over the *executed* result. Any fact the user states while
  chatting goes into business memory like any other custom answer.
- **Technology.** LLM with structured output for the plan; existing pandas layer
  for execution; the existing chart builders for rendering. No text-to-SQL, no
  code execution — the plan is a closed vocabulary, which removes the entire
  class of arbitrary-code risk.
- **Difficulty.** Medium. The planner and validator are the work; execution and
  rendering already exist.
- **Impact.** Transformative. It closes the collaboration loop the product is
  built around.
- **Priority.** Must Have.
- **Requirements.** None beyond the current model provider.

#### A2. Suggested next questions

- **Problem.** A non-analyst does not know what to ask. An empty chat box is as
  intimidating as an empty spreadsheet.
- **How it works.** After the insights stage, generate four questions this
  dataset can actually answer, each tied to a plan that is validated before it
  is offered — so a suggested question is never one the data cannot answer.
  Rendered as buttons that run A1 directly.
- **Technology.** LLM over the schema and the computed findings; the A1 validator
  as the gate.
- **Difficulty.** Easy, once A1 exists.
- **Impact.** High. It is what makes A1 usable by the intended audience.
- **Priority.** Must Have (bundled with A1).
- **Requirements.** A1.

#### A3. Voice conversation

- **Problem.** The intended user is a shop owner or a clinic manager, often on
  their feet. Typing an analytical question is a barrier that has nothing to do
  with their expertise.
- **How it works.** A microphone button records the question; speech-to-text
  feeds A1; the answer sentence is spoken back while the chart renders. Arabic
  and English both supported.
- **Technology.** Whisper or a provider speech endpoint for STT; any TTS.
- **Difficulty.** Medium — mostly browser audio plumbing in Streamlit.
- **Impact.** High for the actual target user, low for an analyst.
- **Priority.** Nice to Have.
- **Requirements.** Speech API access; microphone permission.

---

### Family B — Trust and correctness

#### B1. Numeric grounding (claim verification)

- **Problem.** The insight agent asks a model to write conclusions containing
  figures. Nothing checks that those figures appear anywhere in the data. A
  single sentence saying "revenue grew 45%" when it grew 4.5% ends the user's
  trust in every other number on the page — and they have no way to tell which
  one was wrong.
- **How it works.** Before any generated text is shown, build an **allow-list of
  facts**: every value in every chart table, every KPI, every profile statistic.
  Extract each number from the generated text, normalise it (2.8M, 2,800,000,
  2.8 million), and match it against the allow-list within a tolerance. A
  sentence containing an unmatched number is either regenerated once or dropped,
  and the drop is written to the activity log.
- **Technology.** Deterministic — regex, number normalisation, set membership.
  No model needed for the check itself.
- **Difficulty.** Easy to medium. Number normalisation is the fiddly part.
- **Impact.** Invisible when it works, which is the point. It is the difference
  between a demo and something a business acts on.
- **Priority.** Must Have.
- **Requirements.** None.

#### B2. Adversarial insight critic

- **Problem.** A finding can be arithmetically correct and still wrong as a
  conclusion — based on eleven rows, or explained by a third variable nobody
  mentioned.
- **How it works.** Each insight goes to a second agent whose only instruction is
  to refute it, with the supporting rows available. It returns a verdict and a
  reason. A refuted insight is demoted to low confidence with the objection
  attached, not deleted — the user should see the objection.
- **Technology.** LLM with a deliberately adversarial persona; the existing
  five-part `Insight` structure.
- **Difficulty.** Easy. It is one more agent in an existing pipeline.
- **Impact.** High on credibility, and it makes the confidence levels mean
  something instead of being a label.
- **Priority.** Must Have.
- **Requirements.** None.

#### B3. Confounder and Simpson's paradox detector

- **Problem.** "Channel A converts better than channel B" reverses once you split
  by region more often than anyone expects. This is the most dangerous class of
  wrong conclusion because it survives every sanity check a non-analyst can run.
- **How it works.** For every group comparison that becomes an insight, re-run it
  conditioned on each other categorical column. If the sign of the difference
  reverses in most subgroups, flag it, and rewrite the insight to lead with the
  conditioning variable instead.
- **Technology.** Fully deterministic — pandas group-bys. The model only writes
  the resulting sentence.
- **Difficulty.** Medium. The statistics are simple; deciding which comparisons
  to test without a combinatorial explosion is the design work.
- **Impact.** High, and genuinely differentiating. Almost no self-service tool
  does this.
- **Priority.** Must Have.
- **Requirements.** None.

#### B4. Significance gate

- **Problem.** With 40 rows per region, a 6% difference between two regions is
  noise. The product currently presents it in the same voice as a 60% difference
  across 4,000 rows.
- **How it works.** Before a comparison becomes an insight, run the appropriate
  test (bootstrap difference in means, chi-square for counts). Below the
  threshold it is either dropped or stated as "too few records to tell apart",
  which is itself a useful finding.
- **Technology.** Deterministic. Bootstrap avoids adding a stats dependency.
- **Difficulty.** Easy.
- **Impact.** Medium-high. Mostly it removes noise, which raises the value of
  what remains.
- **Priority.** Must Have.
- **Requirements.** None.

#### B5. Data trust score

- **Problem.** The user has no calibrated sense of how far to trust the whole
  analysis. Completeness, duplication and time coverage are all in the profile
  but never combined into a judgement.
- **How it works.** A 0–100 score from completeness, duplication, time coverage,
  cardinality sanity and outlier share, shown on the first screen with its
  drivers and the one action that would raise it most.
- **Technology.** Deterministic; model writes the explanation.
- **Difficulty.** Easy.
- **Impact.** Medium. Sets expectations before the user reads a single finding.
- **Priority.** Nice to Have.
- **Requirements.** None.

---

### Family C — Memory that compounds

#### C1. Contradiction detection (memory that argues back)

- **Problem.** Memory is currently a passive list. If the user said "peak season
  starts in November" and the new file peaks in March, the system injects both
  into the prompt and says nothing. That moment — where the data disagrees with
  the owner — is the most valuable conversation the product can have.
- **How it works.** After profiling, every stored fact that can be expressed as a
  testable claim is checked against the new data. A contradiction raises a
  decision through the existing framework: *"You told us peak season starts in
  November. This file peaks in March. Which is right?"* with options to update
  the fact, keep it and treat this file as unusual, or record that both are true
  for different parts of the business.
- **Technology.** LLM to turn a stored sentence into a testable claim (a column,
  a comparison, a threshold); pandas to test it; the existing decision framework
  to raise it.
- **Difficulty.** Medium. Depends on C2 for reliability.
- **Impact.** Very high. This is the product thesis made real, and it is the
  thing a general-purpose chat assistant structurally cannot do.
- **Priority.** Must Have.
- **Requirements.** C2 ideally; workable without it.

#### C2. Structured memory (facts as data, not sentences)

- **Problem.** `Fact.statement` is free text. It cannot be tested, compared,
  versioned or applied automatically. Every memory feature is capped by this.
- **How it works.** Extend `Fact` with an optional machine-readable form:
  subject entity, predicate, column, operator, value, scope, confidence, valid
  period. The sentence stays for display; the structure drives behaviour. When a
  fact carries structure, it is applied automatically — a VIP threshold becomes a
  column on every future run without asking again.
- **Technology.** LLM extraction into a schema at the point the fact is recorded,
  with the user shown the interpretation and able to correct it.
- **Difficulty.** Medium. The migration is trivial; the extraction schema needs
  care.
- **Impact.** High, mostly as an enabler. It is the foundation under C1, C3, C4
  and E2.
- **Priority.** Must Have.
- **Requirements.** None.

#### C3. Longitudinal comparison across runs

- **Problem.** Each run is an island. "Revenue is 11.8M" is far less useful than
  "revenue is 11.8M, down 8% on the period you analysed in March."
- **How it works.** Fingerprint each run's schema. On a new run with a matching
  fingerprint, load the previous summary, align the periods, and compute deltas
  for every KPI. Changes beyond a threshold become their own insight class, with
  the previous figure and the gap.
- **Technology.** Deterministic comparison over the existing `summary.json`;
  model writes the narrative.
- **Difficulty.** Medium. Period alignment across differently-shaped exports is
  the real work.
- **Impact.** Very high. It converts a one-shot tool into something with a reason
  to be opened every month.
- **Priority.** Must Have.
- **Requirements.** At least two runs. Schema fingerprinting.

#### C4. Learned decision preferences

- **Problem.** A user who has capped outliers on three consecutive runs is asked
  the same question a fourth time, in the same words.
- **How it works.** The answer history is already stored. When the same decision
  topic has been answered consistently, the recommended option becomes that
  answer, the reason states why ("you have chosen this each time"), and the user
  can still change it. After enough consistency, offer to stop asking.
- **Technology.** Deterministic frequency over stored answers; no model needed.
- **Difficulty.** Easy.
- **Impact.** Medium-high on repeat use. It is what makes the product feel like
  it knows the user.
- **Priority.** Nice to Have.
- **Requirements.** Persisted answer history across runs.

---

### Family D — Data reach

#### D1. Multi-file joining agent

- **Problem.** Real businesses export sales, products and customers separately.
  The product handles exactly one file, which is a hard ceiling on how useful an
  analysis can be — margin analysis is impossible if cost lives in a second file.
- **How it works.** Accept several files. Profile each. Propose join keys from
  name similarity, type compatibility and value overlap, ranked by measured
  overlap rather than by guess. Show the user the proposed relationships in plain
  language, with the row counts each join would produce and a warning where a
  join would multiply rows. The user confirms through the existing framework.
- **Technology.** Deterministic candidate scoring; LLM to name the relationship
  in business terms and to explain the consequence.
- **Difficulty.** Hard. Fan-out and many-to-many joins are where analyses go
  silently wrong.
- **Impact.** Very high. It removes the ceiling.
- **Priority.** Must Have.
- **Requirements.** Multi-file upload; a redesign of the load stage.

#### D2. Document and image ingestion

- **Problem.** Many small businesses do not have an export. They have a stack of
  invoices, a PDF bank statement, or photographs of a handwritten ledger.
- **How it works.** A vision model extracts line items into a table. Extraction
  confidence is shown per field, low-confidence cells are highlighted, and the
  user corrects them in a grid before the table enters the normal pipeline.
- **Technology.** Vision LLM for extraction; the existing pipeline downstream.
- **Difficulty.** Hard. Accuracy on handwriting and correction UX are both real
  work.
- **Impact.** High, and it opens a market segment that has no data file at all.
- **Priority.** Future Feature.
- **Requirements.** Vision model access; correction interface.

#### D3. External context enrichment

- **Problem.** "November peaked" is an observation. "November peaked, which was
  Black Friday and the two weeks before Ramadan" is an explanation.
- **How it works.** For each notable time point, retrieve context for the
  relevant country and sector — public holidays, religious calendar, major retail
  events, FX moves, weather where it matters — and offer it as a *candidate*
  explanation the user confirms or rejects. A confirmed one becomes a business
  memory fact.
- **Technology.** RAG over a curated calendar and macro dataset; a holiday
  library covers most of the value with no model at all.
- **Difficulty.** Medium.
- **Impact.** High. It is the most obviously "intelligent" moment in the product
  and it is grounded rather than invented.
- **Priority.** Nice to Have.
- **Requirements.** Country and sector from the user; holiday data; optionally an
  FX or weather API.

#### D4. Live connectors

- **Problem.** Manual export is friction, and it is what stops the product being
  used monthly.
- **How it works.** OAuth connections to Google Sheets, Shopify, QuickBooks and
  similar. A saved connection turns a re-run into one click, which is what C3 and
  E3 need to be worth anything.
- **Technology.** Standard API integration. No AI.
- **Difficulty.** Medium per connector, and it never ends.
- **Impact.** High on retention, zero on analysis quality.
- **Priority.** Future Feature.
- **Requirements.** OAuth apps per provider; token storage; a security review.

---

### Family E — Prediction and decision support

#### E1. Root cause explorer

- **Problem.** "Why did November spike?" is the first question every owner asks
  and the product cannot answer it.
- **How it works.** Given a point of interest, decompose the change across every
  available dimension — which region, channel, product and customer segment
  contributed how much of the total movement — and rank contributions. Present it
  as "November was 1.6M above average; 62% of that came from Electronics in
  Cairo, and 24% from three wholesale customers."
- **Technology.** Deterministic contribution analysis, which is exactly the
  shape the current architecture wants. The model writes the sentence.
- **Difficulty.** Medium.
- **Impact.** Very high. It is the single most requested capability in this
  category of product.
- **Priority.** Must Have.
- **Requirements.** A date column and at least one categorical column.

#### E2. Forecasting with honest uncertainty

- **Problem.** Owners plan forward; the product only looks back.
- **How it works.** Seasonal decomposition plus a simple model, presented as a
  range rather than a line, with the assumptions stated and a refusal to forecast
  when there is too little history — which is more valuable than a confident
  wrong number.
- **Technology.** statsmodels or a seasonal-naive baseline. Deliberately not a
  neural forecaster: explicability matters more than accuracy here, and the
  history is short.
- **Difficulty.** Medium.
- **Impact.** High, with a real risk of overtrust if the uncertainty is not
  presented well.
- **Priority.** Nice to Have.
- **Requirements.** At least 18 months of history for anything seasonal.

#### E3. What-if simulator

- **Problem.** Every finding ends in an action, and the user has no way to test
  the action before taking it.
- **How it works.** The user states a change in plain language — "raise
  Electronics prices 10%" — which is parsed into a structured scenario, applied
  to a copy of the data with an elasticity assumption the user can adjust, and
  the KPI deltas are shown side by side. The assumption is always visible,
  because it is doing most of the work.
- **Technology.** LLM to parse the scenario into structured parameters; pandas to
  apply it.
- **Difficulty.** Medium. The honesty about assumptions is the hard part.
- **Impact.** High. It moves the product from reporting to decision support.
- **Priority.** Nice to Have.
- **Requirements.** Elasticity assumptions, ideally from the user's own history.

#### E4. At-risk customer scoring

- **Problem.** Retention is where the money is, and "customers who stopped
  buying" is invisible in a transaction file unless someone looks for it.
- **How it works.** Recency, frequency and value per customer; flag those whose
  gap since last purchase far exceeds their own historical rhythm. Output a
  ranked list with the revenue at stake, exportable as a call list.
- **Technology.** Deterministic RFM. No ML needed, and ML would be worse here
  because the output has to be explainable to whoever makes the calls.
- **Difficulty.** Easy.
- **Impact.** High, and immediately actionable — the output is a list of names.
- **Priority.** Nice to Have.
- **Requirements.** Customer identifier and date columns.

---

### Family F — Reach and operation

#### F1. Arabic-native bilingual experience

- **Problem.** The product is aimed at business owners who are not data
  specialists. In this region most of them work in Arabic. The entire interface,
  every explanation and every report is English-only, so the intended user
  cannot use the product at all.
- **How it works.** Translate the interface; have every agent write in the user's
  language; support Arabic free-text answers into business memory; RTL layout;
  Arabic-capable fonts in the PDF and PowerPoint output. Code, logs and column
  names stay English — only what the user reads changes.
- **Technology.** The existing LLM handles the prose natively. The work is
  interface strings, RTL layout and font embedding in ReportLab.
- **Difficulty.** Medium. The report fonts are the awkward part.
- **Impact.** Very high — it decides whether the target market can use the
  product at all.
- **Priority.** Must Have.
- **Requirements.** Arabic-capable embedded fonts; a translated string catalogue.

#### F2. Continuous monitoring agent

- **Problem.** The product is a one-shot analysis. Business problems appear
  between analyses.
- **How it works.** A scheduled re-run against a saved connection compares
  against the established baseline and alerts only on a genuine break — not on
  normal variation. The alert says what changed, by how much, and what it is
  worth.
- **Technology.** Scheduler; the C3 comparison machinery; anomaly thresholds from
  the data's own history rather than fixed numbers.
- **Difficulty.** Medium, given C3 and D4.
- **Impact.** Very high on retention. It is what makes this a service rather than
  a tool.
- **Priority.** Future Feature.
- **Requirements.** C3, D4, scheduling, a delivery channel.

#### F3. Spoken executive briefing

- **Problem.** A twelve-page PDF does not get read before the meeting it was
  written for.
- **How it works.** Generate a three-minute spoken summary of the findings and
  the recommended actions, downloadable as audio.
- **Technology.** LLM for the script; TTS for the audio.
- **Difficulty.** Easy.
- **Impact.** Medium. Genuinely useful for a commute; not a reason to choose the
  product.
- **Priority.** Nice to Have.
- **Requirements.** TTS access.

#### F4. Industry benchmarking

- **Problem.** "Your margin is 29%" means nothing without knowing that the sector
  runs at 22%.
- **How it works.** The user states their sector and country; retrieve published
  benchmark ranges; present the comparison with the source and the year, and be
  explicit when no reliable benchmark exists.
- **Technology.** RAG over a curated benchmark corpus. **The corpus is the whole
  problem** — this is a data licensing exercise, not an engineering one, and an
  invented benchmark is worse than none.
- **Difficulty.** Hard, for that reason.
- **Impact.** High if the data is real, actively harmful if it is not.
- **Priority.** Future Feature.
- **Requirements.** A licensed benchmark dataset.

#### F5. Reproducible notebook export

- **Problem.** An accountant or an analyst downstream will want to verify the
  numbers, and cannot.
- **How it works.** Export the run as a standalone Python notebook that
  reproduces every cleaning step, chart and KPI from the original file. The
  activity log already holds everything needed to generate it.
- **Technology.** Template rendering from the existing log. No AI.
- **Difficulty.** Medium.
- **Impact.** Medium overall, high for the one sceptical person in the room whose
  approval decides adoption.
- **Priority.** Nice to Have.
- **Requirements.** Operations recorded as parameters rather than only as prose.

---

## 4. Full ranking

Scored on value to the product thesis, practicality against the current
codebase, and defensibility against a general-purpose assistant.

| # | Feature | Difficulty | Priority |
|---|---|---|---|
| 1 | A1 Ask Anything | Medium | Must Have |
| 2 | B1 Numeric grounding | Easy–Medium | Must Have |
| 3 | F1 Arabic-native experience | Medium | Must Have |
| 4 | C1 Contradiction detection | Medium | Must Have |
| 5 | E1 Root cause explorer | Medium | Must Have |
| 6 | C3 Longitudinal comparison | Medium | Must Have |
| 7 | C2 Structured memory | Medium | Must Have |
| 8 | B2 Adversarial critic | Easy | Must Have |
| 9 | D1 Multi-file joining | Hard | Must Have |
| 10 | B3 Confounder detector | Medium | Must Have |
| 11 | A2 Suggested questions | Easy | Must Have |
| 12 | B4 Significance gate | Easy | Must Have |
| 13 | E4 At-risk customers | Easy | Nice to Have |
| 14 | C4 Learned preferences | Easy | Nice to Have |
| 15 | E3 What-if simulator | Medium | Nice to Have |
| 16 | D3 External context | Medium | Nice to Have |
| 17 | B5 Trust score | Easy | Nice to Have |
| 18 | E2 Forecasting | Medium | Nice to Have |
| 19 | F2 Continuous monitoring | Medium | Future |
| 20 | A3 Voice conversation | Medium | Nice to Have |
| 21 | F5 Notebook export | Medium | Nice to Have |
| 22 | F3 Spoken briefing | Easy | Nice to Have |
| 23 | D4 Live connectors | Medium | Future |
| 24 | D2 Document ingestion | Hard | Future |
| 25 | F4 Industry benchmarking | Hard | Future |

## 5. Deliberately rejected

Worth recording so they are not proposed again.

- **AutoML / automatic model training.** The users cannot evaluate a model and
  the datasets are too small to train one honestly. It would produce confident
  numbers nobody can check — the opposite of what this product is for.
- **Free-form code execution from chat.** The obvious way to build A1, and the
  wrong one. A closed query vocabulary gives the same coverage for real questions
  with none of the risk.
- **A neural forecaster.** Explicability matters more than accuracy at this
  history length, and a wrong forecast presented confidently costs more than no
  forecast.
- **Automatic natural-language "storytelling" over raw data.** Without B1 this is
  a hallucination generator pointed at someone's business.

## 6. Roadmap

**v0.2 — Make it trustworthy.** B1, B2, B4, B5. Nothing new on screen; every
existing number becomes defensible. This has to come first, because every later
feature multiplies whatever error rate exists underneath it.

**v0.3 — Make it a conversation.** A1, A2, E1. The user can finally ask, and the
question they always ask has an answer.

**v0.4 — Make the memory compound.** C2, C1, C3, C4. Where the product stops
being replaceable by a general assistant with a file attached.

**v0.5 — Reach the actual market.** F1, D1. Arabic and multi-file — the two
things blocking real businesses from real use.

**v0.6 — Look forward.** E4, E3, E2, D3.

**v1.0 — Become a service.** D4, F2, F5. Connected, scheduled, verifiable.

**Beyond.** D2, F4, and multi-user collaboration on findings.

## 7. The one structural change worth making early

A1 and E1 both need something the pipeline cannot currently express: **a
question against finished state, outside the forward-only stage sequence.**

The cheapest way in is a second entry point on `Supervisor` — call it
`ask(question)` — that runs a short planner/executor loop against the completed
`PipelineState` without advancing any stage, and appends to the activity log like
anything else. It reuses the state, the memory, the chart builders and the log,
and it does not disturb the generator pipeline at all.

Building that entry point first makes A1, A2, E1 and E3 all incremental. Not
building it means each of them invents its own way around the architecture.
