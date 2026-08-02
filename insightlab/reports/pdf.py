"""PDF rendering with ReportLab."""

from __future__ import annotations

import io
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..core.state import PipelineState
from . import arabic
from .content import ReportContent, build_content

INK = colors.HexColor("#0b0b0b")
SECONDARY = colors.HexColor("#52514e")
MUTED = colors.HexColor("#898781")
RULE = colors.HexColor("#e1e0d9")
ACCENT = colors.HexColor("#2a78d6")

#: Table columns wider than this get their text wrapped into a paragraph.
WRAP_THRESHOLD = 45


def _styles(rtl: bool = False) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body_font = arabic.FONT_NAME if rtl else "Helvetica"
    bold_font = arabic.FONT_NAME if rtl else "Helvetica-Bold"
    italic_font = arabic.FONT_NAME if rtl else "Helvetica-Oblique"
    align = TA_RIGHT if rtl else TA_LEFT
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=bold_font,
            fontSize=24,
            leading=29,
            textColor=INK,
            alignment=align,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName=body_font,
            fontSize=11,
            textColor=MUTED,
            spaceAfter=18,
            alignment=align,
        ),
        "heading": ParagraphStyle(
            "heading",
            parent=base["Heading1"],
            fontName=bold_font,
            fontSize=15,
            leading=19,
            textColor=INK,
            spaceBefore=18,
            spaceAfter=8,
            alignment=align,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName=body_font,
            fontSize=10,
            leading=15,
            textColor=SECONDARY,
            spaceAfter=8,
            alignment=align,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["Normal"],
            fontName=body_font,
            fontSize=10,
            leading=15,
            textColor=SECONDARY,
            leftIndent=0 if rtl else 10,
            rightIndent=10 if rtl else 0,
            spaceAfter=7,
            alignment=align,
        ),
        "cell": ParagraphStyle(
            "cell",
            parent=base["Normal"],
            fontName=body_font,
            fontSize=8,
            leading=11,
            textColor=SECONDARY,
            alignment=align,
        ),
        "cellhead": ParagraphStyle(
            "cellhead",
            parent=base["Normal"],
            fontName=bold_font,
            fontSize=8,
            leading=11,
            textColor=INK,
            alignment=align,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["Normal"],
            fontName=italic_font,
            fontSize=9,
            leading=13,
            textColor=MUTED,
            spaceAfter=14,
            alignment=align,
        ),
    }


def _escape(text: str) -> str:
    # Shaping happens here, at the single point every string passes through on
    # its way into the document, so no caller can forget it.
    prepared = arabic.prepare(str(text))
    return (
        prepared
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _table(headers: list[str], rows: list[list[str]], styles, available: float,
           rtl: bool = False) -> Table:
    """Build a table, wrapping any column whose text is too long for a cell."""
    # A right-to-left table reads with its first column on the right, and
    # ReportLab has no notion of that, so the columns are physically reversed.
    if rtl:
        headers = list(reversed(headers))
        rows = [list(reversed(row)) for row in rows]

    wrap = [
        any(len(str(row[index])) > WRAP_THRESHOLD for row in rows)
        for index in range(len(headers))
    ]

    data = [[Paragraph(_escape(head), styles["cellhead"]) for head in headers]]
    for row in rows:
        data.append(
            [
                Paragraph(_escape(value), styles["cell"]) if wrap[index] else _escape(value)
                for index, value in enumerate(row)
            ]
        )

    # Wrapped columns take the space the narrow ones do not need.
    narrow = available * 0.16
    wide_count = sum(wrap) or 1
    remaining = available - narrow * (len(headers) - sum(wrap))
    widths = [
        max(remaining / wide_count, 40) if wrap[index] else narrow
        for index in range(len(headers))
    ]

    table = Table(
        data, colWidths=widths, repeatRows=1, hAlign="RIGHT" if rtl else "LEFT"
    )
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), styles["cell"].fontName),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TEXTCOLOR", (0, 1), (-1, -1), SECONDARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), INK),
                ("FONTNAME", (0, 0), (-1, 0), styles["cellhead"].fontName),
                ("LINEBELOW", (0, 0), (-1, 0), 0.75, ACCENT),
                ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                # Plain-string cells are not Paragraphs and so do not pick up
                # the paragraph alignment; without this the headers and the
                # values under them sit on opposite sides of each column.
                ("ALIGN", (0, 0), (-1, -1), "RIGHT" if rtl else "LEFT"),
            ]
        )
    )
    return table


def _chart_flowables(content: ReportContent, chart_id: str, state, styles, width):
    """A chart image plus its explanation, or just the explanation if export failed."""
    flowables = []
    chart = state.chart(chart_id)
    if chart is None:
        return flowables

    flowables.append(Paragraph(f"<b>{_escape(chart.title)}</b>", styles["body"]))
    image = content.chart_images.get(chart_id)
    if image:
        picture = Image(io.BytesIO(image))
        ratio = picture.imageHeight / picture.imageWidth
        picture.drawWidth = width
        picture.drawHeight = width * ratio
        flowables.append(picture)
        flowables.append(Spacer(1, 4))
    flowables.append(Paragraph(_escape(chart.description), styles["caption"]))
    return flowables


def write_pdf(state: PipelineState, path: Path, content: ReportContent | None = None) -> Path:
    """Render the full report to ``path``."""
    content = content or build_content(state)
    rtl = state.language.rtl and arabic.available()
    styles = _styles(rtl)

    path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=content.title,
    )
    available = document.width

    story = [
        Paragraph(_escape(content.title), styles["title"]),
        Paragraph(_escape(content.subtitle), styles["subtitle"]),
    ]

    for section in content.sections:
        story.append(Paragraph(_escape(section.title), styles["heading"]))

        for paragraph in section.paragraphs:
            story.append(Paragraph(_escape(paragraph), styles["body"]))

        for bullet in section.bullets:
            lines = [line for line in str(bullet).split("\n") if line.strip()]
            if not lines:
                continue
            story.append(
                Paragraph(f"&bull;&nbsp;{_escape(lines[0])}", styles["bullet"])
            )
            for line in lines[1:]:
                story.append(
                    Paragraph(f"&nbsp;&nbsp;&nbsp;{_escape(line)}", styles["bullet"])
                )

        if section.table:
            headers, rows = section.table
            if rows:
                story.append(Spacer(1, 4))
                story.append(_table(headers, rows, styles, available, rtl))
                story.append(Spacer(1, 6))

        if section.chart_ids:
            for index, chart_id in enumerate(section.chart_ids):
                if index and index % 2 == 0:
                    story.append(PageBreak())
                story.extend(
                    _chart_flowables(content, chart_id, state, styles, available)
                )

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return path


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(
        document.pagesize[0] - 18 * mm, 12 * mm, f"Page {canvas.getPageNumber()}"
    )
    canvas.drawString(18 * mm, 12 * mm, "InsightLab")
    canvas.restoreState()
