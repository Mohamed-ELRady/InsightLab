# InsightLab: Master Build Prompt

You are building **InsightLab**, a multi-agent AI platform that lets non-technical business users analyze their own business data through a guided, collaborative process. The AI is the data-analysis expert, the user is the business expert, and every step is designed as a conversation between the two rather than a black box.

## Hard rules (non-negotiable, apply to every file and every commit)

1. **No trace of AI assistance anywhere.** No `Co-Authored-By` trailers, no "Generated with Claude" or similar signatures, no AI mentions in commit messages, code comments, README, or any other file. Check git config and any commit template before the first commit and strip default attribution if present.
2. **English only in code.** All code, comments, variable and function names, log messages, and printed output are written entirely in English, regardless of what language this brief or later conversations use.
3. **Isolated environment first.** Create and activate a Python virtual environment (`python -m venv .venv`) before installing a single package. Never install anything system-wide.
4. **Git from commit one, hosted on GitHub, professional workflow.** See the Git & GitHub section; this applies from the very first file created.

## Core concept

Two operating modes:
- **Interactive Mode**: the pipeline pauses at every decision point, shows the options below, and waits for the user before moving on.
- **Autonomous Mode**: the pipeline runs end to end using the AI's best judgment at each decision point, but every decision and its reasoning are still logged so the user can review them afterward.

## The decision framework

At every stage, in Interactive Mode, present exactly these four options:
1. Use the AI's suggestion.
2. Choose from a set of ready-made alternative suggestions.
3. Write a custom instruction in free text.
4. Skip this stage.

This is shared logic, not something each agent reimplements. Build it once (a `DecisionPoint` component or equivalent) and have every agent call into it.

## Business memory

Any business fact the user states must be saved to a shared store that every agent can read for the rest of the project. Examples:
- "VIP customers are the ones whose purchases exceed $5,000."
- "Peak season starts in November."
- "Ignore cancelled invoices."
- "We have our own internal customer classification."

One shared store, read by all agents, persisted with the project. Not per-agent memory.

## Dialogue style (applies to every agent)

Agents talk like a professional business consultant, not a technical tool:
- Plain language, no jargon.
- Explain why before asking the user to decide, and what effect the decision will have.
- Don't ask more questions than necessary.
- The user should feel like they're working with a business expert, not filling out a form.

## Agent architecture

Build these agents with CrewAI, coordinated by a Supervisor.

**Supervisor Agent**: owns the pipeline state, routes work between agents in order, applies the decision framework at each checkpoint, and knows whether the run is in Interactive or Autonomous mode.

**Data Loader Agent**: ingests the user's data (CSV/Excel at minimum), validates it can be parsed, and reports what was loaded (rows, columns, file type) before handing off.

**Data Understanding Agent**: profiles the loaded data (shape, column types, missing-value rates, basic summary statistics) and produces the first plain-language summary of what the dataset looks like.

**Data Cleaning Agent**: three sub-flows, each ending in the four-option decision framework.
- *Duplicates*: show the duplicate rows, then offer delete / keep / merge / custom instruction.
- *Outliers*: show all outlier values, ask "Do you consider these values normal for your business?", with options data-entry error, naturally high-priced items, seasonal effect, not sure, other (free text).
- *Columns*: for each column, suggest keep / ignore / rename / change data type, with a custom-instruction option.

**Feature Engineering Agent**: suggests new features such as age categories, profit margin, month, quarter, customer lifetime value, revenue classification. Asks "Does your company use any internal classifications?" with a free-text box.

**Exploratory Data Analysis (EDA) Agent**: generates distributions, correlations, trends, heatmaps, boxplots, histograms, time-based analysis, and category analysis. Asks which business axis to focus on (sales, customers, products, marketing, profits, regions, operations), plus free text.

**Insight Agent**: turns analysis into business-language conclusions. Every insight includes the result, the evidence behind it, the business interpretation, a confidence level, and a suggested action.

**KPI Agent**: suggests relevant KPIs (revenue, profit, profit margin, growth rate, customer retention, average order value). Asks "Does your company rely on any specific KPIs?" with free text.

**Dashboard Agent**: asks who the dashboard is for (CEO, sales manager, marketing manager, CFO, operations manager), then builds a general overview dashboard plus specialized ones per department, with interactive filters, drill-down, responsive layout, and export.

**Report Agent**: produces PDF, PowerPoint, and Word versions of a report containing executive summary, data description, cleaning summary, top insights, KPIs, charts, recommendations, and next steps.

## Final deliverables

By the end of a run, the user should have:
- A cleaned copy of their data.
- A full log of every cleaning operation performed.
- Top business insights.
- A KPI summary.
- An executive dashboard.
- Detailed dashboards per axis.
- Actionable recommendations.
- Downloadable reports.
- A full log of every project step.
- The accumulated business memory, saved for future analyses.

## Suggested tech stack (confirm and adjust during planning)

- Python, CrewAI for agent orchestration.
- pandas / numpy for data handling, scikit-learn where needed, matplotlib / seaborn / plotly for charts.
- A web UI framework for the interactive parts and dashboards. Streamlit is a reasonable default for speed; use FastAPI plus a JS frontend if more control over drill-down and export is needed.
- python-docx / python-pptx / a PDF library for the Report Agent's output formats.
- An LLM provider for the CrewAI agents, configured through environment variables in a git-ignored `.env` file. Ask which provider and model to use if this isn't already decided.

## Suggested project structure (adjust as needed)

```
insightlab/
├── agents/              # one module per agent
├── core/                # shared state, business memory, decision framework
├── data/                # sample/test data only, real user data never committed
├── dashboard/           # dashboard app
├── reports/             # report generation
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── main.py
```

## Environment setup

1. `python -m venv .venv && source .venv/bin/activate`
2. Only then install dependencies, tracked in `requirements.txt`.
3. `.gitignore` excludes `.venv/`, `__pycache__/`, `.env`, and any real user data files.

## Git & GitHub workflow

1. `git init` before writing any project file.
2. Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`), one logical change per commit, commit often.
3. Keep `main` stable; do feature work on branches per module (`feature/data-cleaning-agent`, etc.) and merge back cleanly.
4. Create a GitHub repository named `InsightLab` and push the history. Use `gh repo create InsightLab --private --source=. --push` if the GitHub CLI is authenticated; otherwise create it manually on github.com and add it as the `origin` remote.
5. No AI attribution anywhere in the history. Double-check commit messages, the README, and code comments before every push.

## How to start

1. Set up the virtual environment and git repository first, per the sections above.
2. Confirm the open questions in the tech-stack section (UI framework, LLM provider, sample dataset for testing) before writing agent code.
3. Build and commit one agent at a time, starting with Supervisor and Data Loader, testing each before moving to the next.
4. Treat this document as the source of truth for scope. If a later instruction conflicts with it, ask before assuming.
