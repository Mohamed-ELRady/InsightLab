"""Report Agent: write everything up and save the run.

The deliverables are what the owner is left holding after the conversation ends,
so this stage writes the documents, saves the cleaned data next to them, and
persists project memory in a form a later run can pick up.

A failure in one format never costs the others: if PowerPoint export breaks, the
PDF is still written and the failure is reported rather than swallowed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..core.decision import Option
from ..core.reasoning import AgentPersona
from ..core.state import PipelineState
from ..core.storage import save_run, save_run_summary
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
            "Leave the user with something they can read on their own, use, "
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

        state.report_progress(self.stage, 0.08, "progress.report.save")
        workspace = save_run(state)
        # Keep the historical key for saved-run and API compatibility; the UI
        # localises it to "Project memory"/"ذاكرة المشروع".
        state.add_artefact("Business memory", workspace.memory_path, stage=self.stage)
        state.add_artefact("Run summary", workspace.summary_path, stage=self.stage)

        if not chosen:
            state.finish_stage(
                self.stage,
                "Saved the cleaned data and the run log. No documents were "
                "generated.",
            )
            save_run_summary(state, workspace)
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
        jobs = []
        for key in chosen:
            if key not in FORMATS:
                continue
            label, extension, renderer = FORMATS[key]
            target = workspace.report_path(f"{stem}.{extension}")
            jobs.append((label, target, renderer))

        produced = 0
        state.report_progress(self.stage, 0.66, "progress.report.documents")
        # PDF, PowerPoint and Word read the same immutable content and write
        # separate files.  Producing them concurrently shortens the common
        # PDF+PowerPoint path without changing either document.
        with ThreadPoolExecutor(max_workers=min(3, max(1, len(jobs)))) as pool:
            futures = {
                pool.submit(renderer, state, target, content): (label, target)
                for label, target, renderer in jobs
            }
            for index, future in enumerate(as_completed(futures)):
                label, target = futures[future]
                try:
                    future.result()
                except Exception as error:  # noqa: BLE001 - one format must not lose the rest
                    self.warn(state, f"The {label} could not be written: {error}")
                else:
                    state.add_artefact(label, target, stage=self.stage)
                    produced += 1
                state.report_progress(
                    self.stage,
                    0.66 + (0.28 * (index + 1) / max(len(jobs), 1)),
                    "progress.report.documents",
                )

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
        # ``save_run`` happens near the start so renderers can use the workspace.
        # Refresh the audit snapshot after this stage reaches done/failed;
        # otherwise summary.json permanently says the report is still running.
        save_run_summary(state, workspace)

    # -- decision ----------------------------------------------------------

    def _format_decision(self, state: PipelineState):
        ar = state.language.code == "ar"
        return self.decide(
            topic="ملفات التقرير" if ar else "What to produce",
            question="أي ملفات تريد إنشاءها؟" if ar else "Which documents would you like?",
            context=(
                ("أيًا كان اختيارك، ستحصل أيضًا على نسخة البيانات المنظّفة وسجل كامل بكل التعديلات والمعلومات المحفوظة عن مشروعك وبياناتك.\n\n"
                 "ملف PDF للقراءة والحفظ، وPowerPoint للعرض، وWord للتعديل قبل الإرسال.")
                if ar else
                ("Whatever you choose, you also get the cleaned copy of your data, "
                "the full log of every change made to it, and everything you told "
                "us about this project and its data saved for next time.\n\n"
                "The PDF is the one to read and file. The PowerPoint is the one "
                "to present. The Word version is the one to edit before sending "
                "it on.")
            ),
            suggestion=Option(
                label="PDF وPowerPoint" if ar else "PDF and PowerPoint",
                rationale=(
                    "ملف للقراءة والحفظ وملف للعرض، وده يغطي الاستخدام الأكثر شيوعًا."
                    if ar else "One document to read and file, one to present. This covers what most people need."
                ),
                payload={"formats": ["pdf", "pptx"]},
            ),
            alternatives=[
                Option(
                    label="الصيغ الثلاث كلها" if ar else "All three formats",
                    rationale="PDF وPowerPoint وWord." if ar else "PDF, PowerPoint and Word.",
                    payload={"formats": ["pdf", "pptx", "docx"]},
                ),
                Option(
                    label="PDF فقط" if ar else "PDF only",
                    rationale="الخيار الأسرع والأنسب للطباعة." if ar else "The quickest option, and the one that prints correctly.",
                    payload={"formats": ["pdf"]},
                ),
                Option(
                    label="Word فقط" if ar else "Word only",
                    rationale="اختاره لو تريد تعديل التقرير قبل إرساله." if ar else "Choose this if you intend to edit the report before sending it.",
                    payload={"formats": ["docx"]},
                ),
            ],
            custom_prompt=("اكتب الصيغ المطلوبة، مثل: PDF وWord." if ar else "Name the formats you want, for example: pdf and word."),
            skip_effect=(
                "لن يتم إنشاء تقارير، لكن ستظل البيانات المنظّفة وسجل التشغيل محفوظين."
                if ar else "No documents are written. The cleaned data and the run log are still saved."
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
