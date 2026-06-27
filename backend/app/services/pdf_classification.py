from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pymupdf

from app.services.pdf_extraction import (
    clean_extracted_text,
    resolve_pdf_path,
)


DEFAULT_MIN_TEXT_CHARACTERS_PER_PAGE = 20

DEFAULT_LARGE_IMAGE_COVERAGE_THRESHOLD = 0.50


class PDFPageClassification(str, Enum):
    """
    Classification assigned to one PDF page.
    """

    TEXT = "text"
    LOW_TEXT = "low_text"
    IMAGE_ONLY = "image_only"
    EMPTY = "empty"


class PDFDocumentClassification(str, Enum):
    """
    Overall classification assigned to one PDF document.
    """

    TEXT_BASED = "text_based"
    PARTIALLY_EXTRACTABLE = "partially_extractable"
    SCANNED_OR_IMAGE_ONLY = "scanned_or_image_only"
    EMPTY = "empty"
    ENCRYPTED = "encrypted"
    MALFORMED = "malformed"


@dataclass(
    frozen=True,
    slots=True,
)
class PDFPageAnalysis:
    """
    Classification information for one PDF page.
    """

    page_number: int
    classification: PDFPageClassification
    text_character_count: int
    word_count: int
    image_count: int
    image_coverage_ratio: float


@dataclass(
    frozen=True,
    slots=True,
)
class PDFDocumentAnalysis:
    """
    Overall content analysis and classification for one PDF.
    """

    source_path: Path
    classification: PDFDocumentClassification
    page_count: int
    pages: tuple[PDFPageAnalysis, ...]
    extractable_page_count: int
    image_only_page_count: int
    empty_page_count: int
    total_text_characters: int
    total_words: int
    was_repaired: bool
    reason: str
    error_message: str | None = None

    @property
    def has_usable_text(self) -> bool:
        """
        Return True when at least one page contains text.
        """

        return self.extractable_page_count > 0

    @property
    def requires_ocr(self) -> bool:
        """
        Return True when at least one page appears image-only.
        """

        return self.image_only_page_count > 0

    @property
    def requires_password(self) -> bool:
        """
        Return True when the PDF requires authentication.
        """

        return (
            self.classification
            == PDFDocumentClassification.ENCRYPTED
        )

    @property
    def can_continue_to_chunking(self) -> bool:
        """
        Return True when usable text is available for chunking.
        """

        return (
            self.classification
            in {
                PDFDocumentClassification.TEXT_BASED,
                PDFDocumentClassification.PARTIALLY_EXTRACTABLE,
            }
            and self.has_usable_text
        )


def calculate_image_coverage(
    page: pymupdf.Page,
) -> tuple[int, float]:
    """
    Estimate how much of a page is covered by raster images.

    The returned ratio is limited to 1.0 because overlapping images
    could otherwise cause the summed area to exceed the page area.
    """

    image_information = page.get_image_info()

    image_count = len(
        image_information
    )

    page_area = float(
        page.rect.get_area()
    )

    if page_area <= 0:
        return image_count, 0.0

    total_image_area = 0.0

    for image in image_information:
        bounding_box = image.get(
            "bbox"
        )

        if bounding_box is None:
            continue

        image_rectangle = pymupdf.Rect(
            bounding_box
        )

        visible_rectangle = (
            image_rectangle & page.rect
        )

        if visible_rectangle.is_empty:
            continue

        total_image_area += float(
            visible_rectangle.get_area()
        )

    coverage_ratio = min(
        total_image_area / page_area,
        1.0,
    )

    return image_count, coverage_ratio


def classify_page(
    page: pymupdf.Page,
    *,
    minimum_text_characters: int,
    large_image_coverage_threshold: float,
) -> PDFPageAnalysis:
    """
    Extract enough information to classify one page.
    """

    raw_text = page.get_text(
        "text",
        sort=True,
    )

    cleaned_text = clean_extracted_text(
        raw_text
    )

    text_character_count = len(
        cleaned_text
    )

    word_count = (
        len(cleaned_text.split())
        if cleaned_text
        else 0
    )

    (
        image_count,
        image_coverage_ratio,
    ) = calculate_image_coverage(
        page
    )

    if (
        text_character_count
        >= minimum_text_characters
    ):
        classification = (
            PDFPageClassification.TEXT
        )

    elif text_character_count > 0:
        classification = (
            PDFPageClassification.LOW_TEXT
        )

    elif (
        image_count > 0
        and image_coverage_ratio
        >= large_image_coverage_threshold
    ):
        classification = (
            PDFPageClassification.IMAGE_ONLY
        )

    else:
        classification = (
            PDFPageClassification.EMPTY
        )

    return PDFPageAnalysis(
        page_number=page.number + 1,
        classification=classification,
        text_character_count=text_character_count,
        word_count=word_count,
        image_count=image_count,
        image_coverage_ratio=round(
            image_coverage_ratio,
            4,
        ),
    )


def create_malformed_result(
    source_path: Path,
    error: Exception,
) -> PDFDocumentAnalysis:
    """
    Return a controlled result for an unreadable PDF.
    """

    return PDFDocumentAnalysis(
        source_path=source_path,
        classification=(
            PDFDocumentClassification.MALFORMED
        ),
        page_count=0,
        pages=(),
        extractable_page_count=0,
        image_only_page_count=0,
        empty_page_count=0,
        total_text_characters=0,
        total_words=0,
        was_repaired=False,
        reason=(
            "The PDF could not be opened or parsed."
        ),
        error_message=str(error),
    )


def classify_pdf_document(
    pdf_path: str | Path,
    *,
    minimum_text_characters: int = (
        DEFAULT_MIN_TEXT_CHARACTERS_PER_PAGE
    ),
    large_image_coverage_threshold: float = (
        DEFAULT_LARGE_IMAGE_COVERAGE_THRESHOLD
    ),
) -> PDFDocumentAnalysis:
    """
    Inspect and classify one PDF without modifying it.
    """

    if minimum_text_characters < 1:
        raise ValueError(
            "minimum_text_characters must be "
            "greater than or equal to 1."
        )

    if not (
        0.0
        <= large_image_coverage_threshold
        <= 1.0
    ):
        raise ValueError(
            "large_image_coverage_threshold must "
            "be between 0.0 and 1.0."
        )

    resolved_path = resolve_pdf_path(
        pdf_path
    )

    try:
        with pymupdf.open(
            resolved_path
        ) as document:
            page_count = document.page_count

            was_repaired = bool(
                document.is_repaired
            )

            if document.needs_pass:
                return PDFDocumentAnalysis(
                    source_path=resolved_path,
                    classification=(
                        PDFDocumentClassification.ENCRYPTED
                    ),
                    page_count=page_count,
                    pages=(),
                    extractable_page_count=0,
                    image_only_page_count=0,
                    empty_page_count=0,
                    total_text_characters=0,
                    total_words=0,
                    was_repaired=was_repaired,
                    reason=(
                        "The PDF requires a password "
                        "before content can be accessed."
                    ),
                )

            page_analyses: list[
                PDFPageAnalysis
            ] = []

            for page_index in range(
                page_count
            ):
                page = document.load_page(
                    page_index
                )

                page_analysis = classify_page(
                    page,
                    minimum_text_characters=(
                        minimum_text_characters
                    ),
                    large_image_coverage_threshold=(
                        large_image_coverage_threshold
                    ),
                )

                page_analyses.append(
                    page_analysis
                )

            extractable_page_count = sum(
                1
                for page in page_analyses
                if page.text_character_count > 0
            )

            image_only_page_count = sum(
                1
                for page in page_analyses
                if (
                    page.classification
                    == PDFPageClassification.IMAGE_ONLY
                )
            )

            empty_page_count = sum(
                1
                for page in page_analyses
                if (
                    page.classification
                    == PDFPageClassification.EMPTY
                )
            )

            total_text_characters = sum(
                page.text_character_count
                for page in page_analyses
            )

            total_words = sum(
                page.word_count
                for page in page_analyses
            )

            if page_count == 0:
                classification = (
                    PDFDocumentClassification.EMPTY
                )

                reason = (
                    "The PDF contains no pages."
                )

            elif (
                extractable_page_count
                == page_count
            ):
                classification = (
                    PDFDocumentClassification.TEXT_BASED
                )

                reason = (
                    "Every page contains extractable text."
                )

            elif extractable_page_count > 0:
                classification = (
                    PDFDocumentClassification
                    .PARTIALLY_EXTRACTABLE
                )

                reason = (
                    "Some pages contain extractable text "
                    "while others do not."
                )

            elif image_only_page_count > 0:
                classification = (
                    PDFDocumentClassification
                    .SCANNED_OR_IMAGE_ONLY
                )

                reason = (
                    "No text was extracted, but one or more "
                    "pages contain large raster images."
                )

            else:
                classification = (
                    PDFDocumentClassification.EMPTY
                )

                reason = (
                    "No extractable text or significant "
                    "page images were detected."
                )

            return PDFDocumentAnalysis(
                source_path=resolved_path,
                classification=classification,
                page_count=page_count,
                pages=tuple(
                    page_analyses
                ),
                extractable_page_count=(
                    extractable_page_count
                ),
                image_only_page_count=(
                    image_only_page_count
                ),
                empty_page_count=empty_page_count,
                total_text_characters=(
                    total_text_characters
                ),
                total_words=total_words,
                was_repaired=was_repaired,
                reason=reason,
            )

    except (RuntimeError, ValueError) as exc:
        return create_malformed_result(
            source_path=resolved_path,
            error=exc,
        )