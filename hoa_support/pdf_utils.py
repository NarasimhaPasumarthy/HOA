"""PDF reading, image detection and OCR.

Replaces Spring AI's PagePdfDocumentReader, PdfOcrDetector (PDFBox) and the
Tika/Tesseract OCR code in HoaSupportService.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .domain import Document

logger = logging.getLogger(__name__)


def read_pdf_pages(path: Path, top_lines_to_delete: int = 1, bottom_lines_to_delete: int = 1) -> list[Document]:
    """One Document per page, dropping the first/last line (headers/footers),
    like ExtractedTextFormatter.withNumberOfTop/BottomTextLinesToDelete(1)."""
    from pypdf import PdfReader  # type: ignore[reportMissingImports]

    reader = PdfReader(str(path))
    pages: list[Document] = []
    for index, page in enumerate(reader.pages, start=1):
        lines = (page.extract_text() or "").splitlines()
        end = len(lines) - bottom_lines_to_delete if bottom_lines_to_delete else len(lines)
        text = "\n".join(lines[top_lines_to_delete:end]).strip()
        pages.append(Document(text=text, metadata={"page_number": index, "file_name": path.name}))
    return pages


class PdfOcrDetector:
    def has_ocr_content(self, pdf_path: Path) -> bool:
        """True if any page contains an embedded image (same rule as the Java detector)."""
        logger.info("Checking if PDF has OCR content: %s", pdf_path)
        if pdf_path is None or not Path(pdf_path).is_file():
            return False
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf_path))
            has_text = any((p.extract_text() or "").strip() for p in reader.pages)
            has_images = any(self._page_has_image(p) for p in reader.pages)
            logger.info("PDF %s hasSelectableText=%s, hasImageContent=%s", Path(pdf_path).name, has_text, has_images)
            return has_images
        except Exception as exc:  # noqa: BLE001 - mirror Java: log and return false
            logger.warning("Failed to check OCR content for %s: %s", Path(pdf_path).name, exc)
            return False

    @staticmethod
    def has_ocr_metadata(document: Document) -> bool:
        if document is None:
            return False
        if document.metadata.get("ocr") is True:
            return True
        return str(document.metadata.get("type", "")).lower() == "pdf-ocr"

    @staticmethod
    def _page_has_image(page) -> bool:
        resources = page.get("/Resources")
        if resources is None:
            return False
        xobjects = resources.get_object().get("/XObject")
        if xobjects is None:
            return False
        for ref in xobjects.get_object().values():
            if ref.get_object().get("/Subtype") == "/Image":
                return True
        return False


class OcrExtractor:
    def __init__(self, language: str = "eng", tesseract_path: str = "", tessdata_path: str = "") -> None:
        self._language = language
        self._tesseract_path = tesseract_path
        self._tessdata_path = tessdata_path

    def extract_text(self, pdf_path: Path, dpi: int = 300, timeout_seconds: int = 300) -> str:
        """Render every page at 300 DPI and OCR it with Tesseract (Tika OCR_ONLY equivalent)."""
        if pdf_path is None:
            return ""
        try:
            import pytesseract
            from pdf2image import convert_from_path

            if self._tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = self._tesseract_path
            config = f'--tessdata-dir "{self._tessdata_path}"' if self._tessdata_path else ""

            texts = []
            for image in convert_from_path(str(pdf_path), dpi=dpi):
                texts.append(
                    pytesseract.image_to_string(image, lang=self._language, config=config, timeout=timeout_seconds)
                )
            return "\n".join(texts)
        except Exception as exc:  # noqa: BLE001
            logger.error("OCR extraction failed for %s: %s", Path(pdf_path).name, exc)
            return ""
