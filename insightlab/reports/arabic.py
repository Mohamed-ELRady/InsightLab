"""Making a PDF that can actually be read in Arabic.

Word and PowerPoint shape and reorder Arabic themselves, so those two formats
need only an RTL paragraph flag. A PDF has no layout engine: ReportLab draws the
glyphs it is given, in the order it is given them. Arabic handed over untouched
comes out as disconnected letters running the wrong way - technically present,
completely unreadable.

Two things are therefore required, and neither is optional:

**Shaping.** Arabic letters change form depending on their neighbours. The
reshaper substitutes the correct contextual form for each one.

**Bidirectional reordering.** Arabic runs right to left while its numbers run
left to right. The Unicode bidi algorithm works out the visual order.

And a font that contains Arabic glyphs, which none of ReportLab's built-in ones
do. If no such font can be found, the report is written in English with the
reason stated, rather than producing a document full of empty boxes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = logging.getLogger(__name__)

#: Name the font is registered under once found.
FONT_NAME = "InsightLabArabic"

#: Places a font covering both Arabic and Latin is usually found. Font
#: collections (.ttc) are excluded: ReportLab cannot read them. Arabic-only
#: fonts are excluded too - see :func:`covers_both`.
CANDIDATE_FONTS = (
    # Linux, and anywhere Noto or DejaVu is installed
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansArabic-Regular.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    # Windows
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
)

#: Characters the font must contain. Arabic alone is not enough: this product
#: is about numbers, and a font with no digits renders every figure as an empty
#: box. macOS ships SFArabic, which is exactly that trap - full Arabic
#: coverage, no Latin digits.
REQUIRED_CHARACTERS = "0123456789.,%" + "\u0628\u0645"

_registered: bool | None = None


def covers_both(path: Path) -> bool:
    """Whether a font can render Arabic letters *and* Western digits."""
    try:
        from fontTools.ttLib import TTFont as _Inspect

        with _Inspect(str(path), fontNumber=0, lazy=True) as font:
            available = font.getBestCmap()
    except Exception:  # noqa: BLE001 - an unreadable font is an unusable one
        return False
    return all(ord(character) in available for character in REQUIRED_CHARACTERS)


def find_font() -> Path | None:
    """A font on this machine that covers Arabic and digits, if there is one."""
    override = os.getenv("INSIGHTLAB_ARABIC_FONT", "").strip()
    if override:
        path = Path(override)
        if path.exists() and path.suffix.lower() in (".ttf", ".otf"):
            if covers_both(path):
                return path
            logger.warning(
                "%s does not contain both Arabic letters and digits, so figures "
                "would render as empty boxes",
                override,
            )
        else:
            logger.warning(
                "INSIGHTLAB_ARABIC_FONT points at %s, which is not usable", override
            )

    for candidate in CANDIDATE_FONTS:
        path = Path(candidate)
        if path.exists() and covers_both(path):
            return path
    return None


def register_font() -> bool:
    """Make the Arabic font available to ReportLab. True if it worked."""
    global _registered
    if _registered is not None:
        return _registered

    path = find_font()
    if path is None:
        logger.warning(
            "No Arabic-capable font found. Set INSIGHTLAB_ARABIC_FONT to a .ttf "
            "to produce Arabic PDFs."
        )
        _registered = False
        return False

    try:
        pdfmetrics.registerFont(TTFont(FONT_NAME, str(path)))
    except Exception as error:  # noqa: BLE001 - any font problem is the same problem
        logger.warning("Could not load the Arabic font at %s: %s", path, error)
        _registered = False
        return False

    _registered = True
    return True


def available() -> bool:
    """Whether an Arabic PDF can be produced at all."""
    return register_font()


def shape(text: str) -> str:
    """Prepare Arabic text for a PDF: contextual forms, then visual order.

    Left alone if the reshaping libraries are missing, which produces
    unreadable output rather than a crash - so callers should check
    :func:`available` first and fall back to English instead.
    """
    if not text:
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError:  # pragma: no cover - dependencies are pinned
        return text

    try:
        return get_display(arabic_reshaper.reshape(text))
    except Exception:  # noqa: BLE001 - never lose a report to a shaping edge case
        return text


def contains_arabic(text: str) -> bool:
    """Whether a string has any Arabic in it and therefore needs shaping."""
    return any("\u0600" <= character <= "\u06ff" for character in str(text))


def prepare(text: str) -> str:
    """Shape a string only if it needs it, leaving English untouched."""
    return shape(text) if contains_arabic(text) else text
