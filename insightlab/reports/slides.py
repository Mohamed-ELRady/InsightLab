"""PowerPoint rendering with python-pptx.

The deck is the version that gets presented, so it carries one idea per slide
and the speaker notes hold the detail that would not fit on screen.
"""

from __future__ import annotations

import io
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

from ..core.state import PipelineState
from .content import ReportContent, build_content

INK = RGBColor(0x0B, 0x0B, 0x0B)
SECONDARY = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x89, 0x87, 0x81)
ACCENT = RGBColor(0x2A, 0x78, 0xD6)

SLIDE_WIDTH = Inches(13.333)
SLIDE_HEIGHT = Inches(7.5)

#: Bullets past this many are pushed onto a continuation slide.
BULLETS_PER_SLIDE = 6

#: KPI tiles across the top of the headline slide.
TILES_PER_ROW = 4


def write_pptx(state: PipelineState, path: Path, content: ReportContent | None = None) -> Path:
    content = content or build_content(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    _RTL["on"] = state.language.rtl

    presentation = Presentation()
    presentation.slide_width = SLIDE_WIDTH
    presentation.slide_height = SLIDE_HEIGHT
    blank = presentation.slide_layouts[6]

    _title_slide(presentation, blank, content)

    if state.kpis:
        _kpi_slide(presentation, blank, state)

    for section in content.sections:
        if section.title in {"Headline figures", "About the data", "Charts"}:
            continue
        _text_slides(presentation, blank, section)

    for chart_id in (section_ids := _chart_ids(content)):
        _chart_slide(presentation, blank, content, state, chart_id)
    if not section_ids and state.charts:
        for chart in state.charts[:6]:
            _chart_slide(presentation, blank, content, state, chart.id)

    presentation.save(str(path))
    return path


def _chart_ids(content: ReportContent) -> list[str]:
    for section in content.sections:
        if section.chart_ids:
            return section.chart_ids
    return []


# ---------------------------------------------------------------------------
# Slide builders
# ---------------------------------------------------------------------------


def _textbox(slide, left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    return frame


#: Set on the deck when the run is in a right-to-left language.
_RTL = {"on": False}


def _style(paragraph, *, size: int, bold: bool = False, colour=SECONDARY) -> None:
    for run in paragraph.runs:
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = colour
        run.font.name = "Arial" if _RTL["on"] else "Calibri"
    if _RTL["on"]:
        # PowerPoint shapes and reorders Arabic itself; it needs the direction
        # flag and the alignment, nothing more.
        properties = paragraph._p.get_or_add_pPr()
        properties.set("rtl", "1")
        paragraph.alignment = PP_ALIGN.RIGHT


def _title_slide(presentation, layout, content: ReportContent) -> None:
    slide = presentation.slides.add_slide(layout)
    frame = _textbox(slide, Inches(0.9), Inches(2.4), Inches(11.5), Inches(2.4))

    heading = frame.paragraphs[0]
    heading.text = content.title
    _style(heading, size=40, bold=True, colour=INK)

    subtitle = frame.add_paragraph()
    subtitle.text = content.subtitle
    _style(subtitle, size=16, colour=MUTED)


def _kpi_slide(presentation, layout, state: PipelineState) -> None:
    slide = presentation.slides.add_slide(layout)
    _slide_title(slide, "What the data shows")

    tiles = state.kpis[:8]
    tile_width = Inches(2.9)
    tile_height = Inches(1.9)
    gap = Inches(0.24)
    left_start = Inches(0.9)
    top_start = Inches(1.6)

    for index, kpi in enumerate(tiles):
        row, column = divmod(index, TILES_PER_ROW)
        left = Emu(int(left_start + column * (tile_width + gap)))
        top = Emu(int(top_start + row * (tile_height + gap)))
        frame = _textbox(slide, left, top, tile_width, tile_height)

        label = frame.paragraphs[0]
        label.text = kpi.name
        _style(label, size=12, colour=MUTED)

        value = frame.add_paragraph()
        value.text = kpi.display_value
        _style(value, size=30, bold=True, colour=ACCENT)

        note = frame.add_paragraph()
        note.text = kpi.formula
        _style(note, size=9, colour=MUTED)


def _slide_title(slide, text: str) -> None:
    frame = _textbox(slide, Inches(0.9), Inches(0.5), Inches(11.5), Inches(0.9))
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    _style(paragraph, size=26, bold=True, colour=INK)


def _text_slides(presentation, layout, section) -> None:
    """One slide per section, continued onto more slides if it is long."""
    blocks: list[str] = list(section.paragraphs) + [
        str(bullet).replace("\n", " - ") for bullet in section.bullets
    ]
    if not blocks:
        return

    chunks = [
        blocks[index : index + BULLETS_PER_SLIDE]
        for index in range(0, len(blocks), BULLETS_PER_SLIDE)
    ]

    for number, chunk in enumerate(chunks):
        slide = presentation.slides.add_slide(layout)
        title = section.title if number == 0 else f"{section.title} (continued)"
        _slide_title(slide, title)

        frame = _textbox(slide, Inches(0.9), Inches(1.6), Inches(11.5), Inches(5.2))
        for index, block in enumerate(chunk):
            paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
            paragraph.text = f"•  {block}"
            paragraph.space_after = Pt(12)
            _style(paragraph, size=14)


def _chart_slide(presentation, layout, content: ReportContent, state, chart_id: str) -> None:
    chart = state.chart(chart_id)
    if chart is None:
        return

    slide = presentation.slides.add_slide(layout)
    _slide_title(slide, chart.title)

    image = content.chart_images.get(chart_id)
    if image:
        slide.shapes.add_picture(
            io.BytesIO(image), Inches(1.4), Inches(1.5), width=Inches(10.5)
        )

    frame = _textbox(slide, Inches(0.9), Inches(6.4), Inches(11.5), Inches(0.9))
    paragraph = frame.paragraphs[0]
    paragraph.text = chart.description
    _style(paragraph, size=13, colour=SECONDARY)
