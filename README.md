# InsightLab

Data analysis for people who run a business rather than analyse one.

You upload your files. At every point where the right answer depends on how
*your* business works rather than on what the numbers say, the analysis stops,
explains the choice in plain language, and asks you. Everything you tell it is
remembered, tested against future data, and applied without asking twice.

The premise is a division of labour: the system is the expert in analysis, you
are the expert in your own business, and neither one alone produces a useful
answer. Which is why the conversation runs both ways — the analysis asks you
questions, and you can ask it questions back.

Available in English and Arabic.

## Why it stops and asks

Automated analysis tools go wrong in a predictable way. They see 24 identical
rows and delete them — but in your business two identical sales in the same
minute might be perfectly normal. They see an order ten times larger than any
other and drop it as an error — but it was your biggest customer of the year.
They rank customers into quartiles — but you already have a VIP tier and it
means something different.

None of those are analysis problems. They are questions only the owner can
answer, so InsightLab asks them, and remembers the answers.

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your API key
streamlit run insightlab/app/main.py
```

There is a sample dataset built in — click **Try it with sample data** to see the
whole flow without uploading anything.

To run an analysis end to end with no questions asked:

```bash
python main.py data/samples/retail_sales.csv
```

## Running without an API key

Every stage works without credentials. The statistics, the cleaning, the charts
and the KPIs are all deterministic pandas code, so the numbers are identical
either way. What the language model adds is the writing: tailored explanations
instead of templates, an interpretation of anything you type in your own words,
and conclusions phrased for your specific situation.

Without a key you get a competent statistical tool. With one you get a
consultant. Nothing breaks in between.

## Nothing reaches you unchecked

Four checks stand between a conclusion and the person reading it.

**Grounding.** Every figure in generated text is matched against an allow-list
built from the data itself. A sentence carrying a number that is not in your
data is removed, and the removal is written to the run log. One wrong figure
ends your trust in every other number on the page, and you would have no way to
tell which one was wrong.

**Significance.** A difference is bootstrapped before it is stated. Too few
records, or a gap inside the noise, and the conclusion is dropped — the chart
stays, because the chart was accurate; it is the conclusion drawn from it that
the data will not carry.

**Confounding.** Every group comparison is re-tested against each other column.
A finding that reverses once you split by region is Simpson's paradox, and it is
reported as such rather than presented as a fact.

**Refutation.** A sceptical reviewer is given the row counts and asked to knock
each finding down. What it finds is shown beside the finding, not used to delete
it — a reader who can see the objection can judge it.

## Ask it questions

The analysis stops and asks you things. You can also ask it:

> Which region grew fastest this year?
> Why was November so high?

A question becomes a **query plan** — a closed vocabulary of measure,
aggregation, grouping, filter — validated against your real columns and then
executed by pandas. No code is generated and no number comes from the model. A
question your data cannot answer gets a plain refusal naming what is missing,
rather than a number that looks like an answer.

**Why did that happen** decomposes any movement across every dimension you have:
*"November was 1.36M above an average month. Most of it is sales channel: Store
alone added 914,155 of it, 67% of the total."* The parts sum to the whole
exactly.

## It remembers, and it argues back

Anything you state about your business is stored as a testable rule, not just a
sentence. On the next file those rules are checked:

> You told us peak season starts in November. This file peaks in March. Which is
> right?

You decide whether the rule changed, the file is unusual, or the rule never
applied here. A rule with a label — *"VIP customers spend over 5,000"* — becomes
a column on every future run without being asked again.

Runs are not islands either. A second analysis of the same kind of file reports
what moved: *"Average order value is 10,808, against 11,540 in your last
analysis — 6% down."*

## Several files at once

Sales, products and customers usually come out of a system as separate exports.
Upload them together and the relationships are proposed from **measured value
overlap**, not from column names, with the effect on your row count stated
before anything is joined. Joining on a repeating key silently multiplies rows
and inflates every total afterwards, so that number is part of the question.

## The two modes

**Interactive** — the pipeline pauses at every decision point and waits for you.
This is the mode the product is built around.

**Autonomous** — the pipeline runs end to end on its own judgement, taking the
recommended option every time. Every decision it took is still logged and marked
as automatic, so you can review afterwards what nobody approved.

## The four options

Every decision, at every stage, offers the same four things:

1. **Use the suggestion** — with the reason it is being recommended.
2. **Choose an alternative** — each with what it would do to your data.
3. **Write your own instruction** — free text, which is interpreted and saved.
4. **Skip** — with a plain statement of what stays unchanged as a result.

This is built once in [`insightlab/core/decision.py`](insightlab/core/decision.py)
and rendered once in [`insightlab/app/decision_panel.py`](insightlab/app/decision_panel.py).
No agent implements its own.

## Business memory

Anything you state about your business is written to one shared store that every
agent reads for the rest of the project:

> "VIP customers are the ones whose purchases exceed 5,000."
> "Peak season starts in November."
> "Ignore cancelled invoices."

It is saved with the run as `business_memory.json`. Point a later analysis at it
and none of those questions get asked again.

## The agents

| Agent | What it does |
|---|---|
| **Supervisor** | Owns the pipeline state, routes between agents, applies the decision framework, knows which mode the run is in |
| **Data Loader** | Reads CSV and Excel: sniffs encoding and delimiter, handles multi-sheet workbooks, title rows above the headings, duplicate column names |
| **Data Understanding** | Profiles the data and gives the first plain-language summary, then asks what a row represents and which rows to leave out |
| **Memory** | Tests what you have already told us against the new file and raises anything that disagrees, before the data is touched |
| **Data Cleaning** | Duplicates, outliers and columns — each sub-flow shows the affected rows before asking |
| **Feature Engineering** | Calendar breakdown, profit and margin, order size bands, customer value; asks about your own classifications and builds them into real columns |
| **Exploratory Analysis** | Asks which business question to answer, offering only the ones your columns can support, then builds the charts |
| **KPI** | Selects the measures worth watching, calculates them with their formulas, and asks about your own measures and targets |
| **Insight** | Turns findings into conclusions with evidence, interpretation, confidence and a specific action; lets you overrule what the numbers suggest |
| **Analyst** | Not a pipeline stage — answers your questions against the finished analysis, on demand |
| **Dashboard** | Asks who will actually open it, then lays out for that reader, plus one dashboard per area explored |
| **Report** | PDF, PowerPoint and Word off one shared content model |

## What you end up with

- A cleaned copy of your data
- Every change made to it, listed in order
- The conclusions, each with the figure behind it and how far it can be trusted
- Your headline numbers, each with its formula
- Dashboards laid out for whoever will open them
- Downloadable reports
- The full log of the run, including every decision and who took it
- Everything you told us about your business, saved for next time

## Design decisions worth knowing

**The model never calculates anything.** Column roles, cleaning operations,
statistics, chart data and KPIs are all deterministic pandas code. The model is
asked to explain those results and to interpret what you type — nothing else.
That is what makes the numbers reproducible and keeps a run cheap.

**Agents are generators.** Each one yields a decision and receives an answer, so
the supervisor can hold the whole pipeline open across a browser round-trip
without any agent knowing that happened, and without replaying earlier work.

**Charts follow one visual system.** A fixed eight-hue categorical order that is
never cycled, a single-hue ramp for magnitude, a diverging ramp with a neutral
midpoint for correlation, and reserved status colours that never become a data
series. Every chart carries the numbers as a table alongside it. Dark mode is a
selected set of steps from the same ramps, not an inversion.

**Failure is contained.** A stage that breaks is marked failed and the run
continues. A report format that fails to render does not cost the others. A
missing API key degrades the writing, not the analysis.

**Circular explanations are excluded by construction.** A column derived from
the measure being explained, or one of our own cleaning markers, will always
appear to explain a difference in it. Those are filtered out everywhere a
"cause" is proposed, because a caveat that is an artefact of our own pipeline is
worse than no caveat.

**Arabic is a first-class language, not a translation layer.** The interface,
every explanation and all three report formats. The PDF needs contextual letter
shaping and bidirectional reordering, which ReportLab does not do, plus a font
carrying both Arabic letters and Western digits — macOS ships one with Arabic
and no digits, which would render every figure as an empty box, so fonts are
validated for both before use.

## Layout

```
insightlab/
├── core/           decision framework, business memory, testable claims,
│                   numeric grounding, activity log, pipeline state,
│                   run persistence, reasoning layer, languages
├── analysis/       profiling, cleaning, features, charts, KPIs, palette,
│                   significance, confounding, attribution, query plans,
│                   cross-run comparison, multi-file joining
├── agents/         one module per agent, the supervisor, the verifier,
│                   and the analyst that answers questions
├── reports/        shared content model, PDF, PowerPoint, Word, Arabic
└── app/            Streamlit interface, decision panel, conversation
data/samples/       the demo dataset
scripts/            sample data generator
tests/              core, analysis, pipeline and interface tests
main.py             command-line entry point
```

## Tests

```bash
pytest
```

The suite runs entirely offline, which is deliberate: it is the configuration
that exercises the fallback paths, and those are what keep the product working
for someone with no credentials.

## Configuration

All settings live in `.env` — see [`.env.example`](.env.example). The provider is
switchable between OpenAI, Anthropic and Google without touching code.

For Arabic PDFs on a machine with no suitable font, point `INSIGHTLAB_ARABIC_FONT`
at a `.ttf` containing both Arabic letters and Western digits. Without one the
Word and PowerPoint versions still work, since those applications supply their
own fonts, and the run says so rather than producing an unreadable PDF.

## Where this is going

[`docs/ai-roadmap.md`](docs/ai-roadmap.md) reviews what the architecture makes
cheap, what it makes expensive, and the features still to come — with what was
considered and deliberately rejected, so it does not get proposed again.
