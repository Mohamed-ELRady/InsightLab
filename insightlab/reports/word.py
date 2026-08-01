"""Word rendering with python-docx.

The Word version exists because it is the one people edit before forwarding, so
it keeps real heading styles and real tables rather than a picture of them.
"""

from __future__ import annotations

import io
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

from ..core.state import PipelineState
from .content import ReportContent, build_content

INK = RGBColor(0x0B, 0x0B, 0x0B)
SECONDARY = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x89, 0x87, 0x81)

#: Page width available for an image, in inches.
CONTENT_WIDTH = 6.3


def write_docx(state: PipelineState, path: Path, content: ReportContent | None = None) -> Path:
    content = content or build_content(state)
    path.parent.mkdir(parents=True, exist_ok=True)

    document = Document()
    _set_base_font(document)

    heading = document.add_heading(content.title, level=0)
    for run in heading.runs:
        run.font.color.rgb = INK

    subtitle = document.add_paragraph(content.subtitle)
    subtitle.runs[0].font.color.rgb = MUTED
    subtitle.runs[0].font.size = Pt(10)

    for section in content.sections:
        document.add_heading(section.title, level=1)

        for paragraph in section.paragraphs:
            document.add_paragraph(paragraph)

        for bullet in section.bullets:
            lines = [line for line in str(bullet).split("\n") if line.strip()]
            if not lines:
                continue
            document.add_paragraph(lines[0], style="List Bullet")
            for line in lines[1:]:
                nested = document.add_paragraph(line)
                nested.paragraph_format.left_indent = Inches(0.5)
                nested.paragraph_format.space_after = Pt(2)
                for run in nested.runs:
                    run.font.size = Pt(9)
                    run.font.color.rgb = SECONDARY

        if section.table:
            _add_table(document, *section.table)

        for chart_id in section.chart_ids:
            _add_chart(document, content, state, chart_id)

    document.save(str(path))
    return path


def _set_base_font(document: Document) -> None:
    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)
    style.font.color.rgb = SECONDARY
    style.paragraph_format.space_after = Pt(8)


def _add_table(document: Document, headers: list[str], rows: list[list[str]]) -> None:
    if not rows:
        return
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.autofit = True

    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.text = str(header)
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
                run.font.size = Pt(9)
                run.font.color.rgb = INK

    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            cells[index].text = str(value)
            for paragraph in cells[index].paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(8.5)

    document.add_paragraph()


def _add_chart(document: Document, content: ReportContent, state, chart_id: str) -> None:
    chart = state.chart(chart_id)
    if chart is None:
        return

    title = document.add_paragraph()
    run = title.add_run(chart.title)
    run.font.bold = True
    run.font.color.rgb = INK

    image = content.chart_images.get(chart_id)
    if image:
        document.add_picture(io.BytesIO(image), width=Inches(CONTENT_WIDTH))
        document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    caption = document.add_paragraph(chart.description)
    for run in caption.runs:
        run.font.size = Pt(9)
        run.font.italic = True
        run.font.color.rgb = MUTED
