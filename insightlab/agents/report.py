"""Report Agent: write everything up and save the run.

The deliverables are what the owner is left holding after the conversation ends,
so this stage writes the documents, saves the cleaned data next to them, and
persists the business memory in a form a later run can pick up.

A failure in one format never costs the others: if PowerPoint export breaks, the
PDF is still written and the failure is reported rather than swallowed.
"""

from __future__ import annotations

from pathlib import Path

from ..core.decision import Option
from ..core.reasoning import AgentPersona
from ..core.state import PipelineState
from ..core.storage import save_run
from ..reports import build_content, write_docx, write_pdf, write_pptx
from .base import Agent, Flow

#: Renderer for each format, with the extension it produces.
FORMATS = {
    "pdf": ("PDF document", "pdf", write_pdf),
    "pptx": ("PowerPoint deck", "pptx", write_pptx),
    "docx": ("Word document", "docx", write_docx),
}


class ReportAgent(Agent):
    stage = "report"
    key = "report"
    title = "Writing the report"

    persona = AgentPersona(
        role="Report writer",
        goal=(
            "Leave the owner with something they can read on their own, act on, "
            "and hand to someone else without having to explain it first."
        ),
        backstory=(
            "You write the document that outlives the meeting. You lead with the "
            "conclusion, you show every figure with the calculation behind it, "
            "and you never present a finding without saying how far it can be "
            "trusted."
        ),
    )

    def run(self, state: PipelineState) -> Flow:
        state.begin_stage(self.stage)

        decision = self._format_decision(state)
        answer = yield decision

        if answer.is_skip:
            chosen: list[str] = []
            self.note(state, "No documents were produced, at your request.")
        elif answer.is_custom:
            self.capture_custom(state, decision, answer, category="context")
            chosen = self._formats_from_text(answer.text)
        else:
            chosen = list(answer.payload.get("formats", []))

        workspace = save_run(state)
        state.add_artefact("Business memory", workspace.memory_path, stage=self.stage)
        state.add_artefact("Run summary", workspace.summary_path, stage=self.stage)

        if not chosen:
            state.finish_stage(
                self.stage,
                "Saved the cleaned data and the run log. No documents were "
                "generated.",
            )
            return

        if state.language.rtl:
            from ..reports import arabic

            if not arabic.available():
                # A PDF without an Arabic font renders every letter as an empty
                # box. Saying so is far better than handing over a file the
                # owner cannot read and cannot diagnose.
                self.warn(
                    state,
                    "No Arabic-capable font was found on this machine, so the "
                    "PDF would come out as empty boxes. Set "
                    "INSIGHTLAB_ARABIC_FONT to a .ttf file to fix it. The Word "
                    "and PowerPoint versions are unaffected, since those "
                    "applications supply their own fonts.",
                )

        # Built once and shared: chart images are the expensive part, and three
        # renderers of the same material must not disagree with each other.
        content = build_content(state)
        if state.charts and not content.chart_images:
            self.warn(
                state,
                "Charts could not be turned into pictures, so the documents carry "
                "the findings as text. The charts are still available on screen.",
            )

        stem = self._filename_stem(state)
        produced = 0
        for key in chosen:
            if key not in FORMATS:
                continue
            label, extension, renderer = FORMATS[key]
            target = workspace.report_path(f"{stem}.{extension}")
            try:
                renderer(state, target, content)
            except Exception as error:  # noqa: BLE001 - one format must not lose the rest
                self.warn(state, f"The {label} could not be written: {error}")
                continue
            state.add_artefact(label, target, stage=self.stage)
            produced += 1

        if produced:
            state.finish_stage(
                self.stage,
                f"Wrote {produced} document(s), the cleaned data and the full run "
                f"log to {workspace.root}.",
            )
        else:
            state.fail_stage(
                self.stage,
                "No document could be written. The cleaned data and the run log "
                "were still saved.",
            )

    # -- decision ----------------------------------------------------------

    def _format_decision(self, state: PipelineState):
        return self.decide(
            topic="What to produce",
            question="Which documents would you like?",
            context=(
                "Whatever you choose, you also get the cleaned copy of your data, "
                "the full log of every change made to it, and everything you told "
                "us about your business saved for next time.\n\n"
                "The PDF is the one to read and file. The PowerPoint is the one "
                "to present. The Word version is the one to edit before sending "
                "it on."
            ),
            suggestion=Option(
                label="PDF and PowerPoint",
                rationale=(
                    "One document to read and file, one to present. This covers "
                    "what most people need."
                ),
                payload={"formats": ["pdf", "pptx"]},
            ),
            alternatives=[
                Option(
                    label="All three formats",
                    rationale="PDF, PowerPoint and Word.",
                    payload={"formats": ["pdf", "pptx", "docx"]},
                ),
                Option(
                    label="PDF only",
                    rationale="The quickest option, and the one that prints correctly.",
                    payload={"formats": ["pdf"]},
                ),
                Option(
                    label="Word only",
                    rationale="Choose this if you intend to edit the report before sending it.",
                    payload={"formats": ["docx"]},
                ),
            ],
            custom_prompt="Name the formats you want, for example: pdf and word.",
            skip_effect=(
                "No documents are written. The cleaned data and the run log are "
                "still saved."
            ),
        )

    @staticmethod
    def _formats_from_text(text: str) -> list[str]:
        lowered = text.casefold()
        wanted = []
        if "pdf" in lowered:
            wanted.append("pdf")
        if any(word in lowered for word in ("powerpoint", "pptx", "slide", "deck", "present")):
            wanted.append("pptx")
        if any(word in lowered for word in ("word", "docx", "doc")):
            wanted.append("docx")
        return wanted or ["pdf"]

    @staticmethod
    def _filename_stem(state: PipelineState) -> str:
        base = Path(state.source_name or "analysis").stem or "analysis"
        safe = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in base
        )
        return f"{safe}_analysis"[:60]
