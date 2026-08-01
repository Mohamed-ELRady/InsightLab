"""Report generation in PDF, PowerPoint and Word."""

from .content import ReportContent, build_content
from .pdf import write_pdf
from .slides import write_pptx
from .word import write_docx

__all__ = [
    "ReportContent",
    "build_content",
    "write_pdf",
    "write_pptx",
    "write_docx",
]
