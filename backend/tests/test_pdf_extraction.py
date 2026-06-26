from pathlib import Path

import pymupdf
import pytest

from app.core.exceptions import (
    PDFExtractionError,
    PDFPathError,
)
from app.services.pdf_extraction import (
    clean_extracted_text,
    extract_pdf_pages,
)


def create_test_pdf(
    pdf_path: Path,
    page_texts: list[str],
) -> None:
    """
    Create a small real PDF for testing.

    Each item in page_texts becomes one PDF page.
    An empty string creates a page with no text.
    """

    with pymupdf.open() as document:
        for page_text in page_texts:
            page = document.new_page()

            if page_text:
                page.insert_text(
                    (72, 72),
                    page_text,
                )

        document.save(
            pdf_path
        )


def test_clean_extracted_text_normalizes_noise() -> None:
    raw_text = (
        "  Evidence\u00a0Vault\tAI  "
        "\r\n\r\n\r\n"
        "  ﬁle\x00   upload  "
    )

    cleaned_text = clean_extracted_text(
        raw_text
    )

    assert cleaned_text == (
        "Evidence Vault AI\n\n"
        "file upload"
    )


def test_clean_extracted_text_handles_empty_input() -> None:
    assert clean_extracted_text("") == ""


def test_extract_pdf_pages_returns_page_metadata(
    tmp_path: Path,
) -> None:
    pdf_path = (
        tmp_path / "three-pages.pdf"
    )

    create_test_pdf(
        pdf_path,
        [
            "First page text",
            "",
            "Third page text",
        ],
    )

    result = extract_pdf_pages(
        pdf_path
    )

    assert result.source_path == (
        pdf_path.resolve()
    )

    assert result.page_count == 3
    assert result.text_page_count == 2
    assert result.empty_page_numbers == (2,)
    assert result.has_extractable_text is True

    assert len(result.pages) == 3

    first_page = result.pages[0]
    second_page = result.pages[1]
    third_page = result.pages[2]

    assert first_page.page_number == 1
    assert first_page.text == "First page text"
    assert first_page.character_count == 15
    assert first_page.word_count == 3
    assert first_page.is_empty is False

    assert second_page.page_number == 2
    assert second_page.text == ""
    assert second_page.character_count == 0
    assert second_page.word_count == 0
    assert second_page.is_empty is True

    assert third_page.page_number == 3
    assert third_page.text == "Third page text"
    assert third_page.is_empty is False

    assert result.total_characters == 30
    assert result.total_words == 6


def test_document_with_only_empty_page_has_no_text(
    tmp_path: Path,
) -> None:
    pdf_path = (
        tmp_path / "empty-page.pdf"
    )

    create_test_pdf(
        pdf_path,
        [""],
    )

    result = extract_pdf_pages(
        pdf_path
    )

    assert result.page_count == 1
    assert result.text_page_count == 0
    assert result.empty_page_numbers == (1,)
    assert result.total_characters == 0
    assert result.total_words == 0
    assert result.has_extractable_text is False


def test_missing_pdf_raises_path_error(
    tmp_path: Path,
) -> None:
    missing_path = (
        tmp_path / "missing.pdf"
    )

    with pytest.raises(
        PDFPathError,
        match="PDF file does not exist",
    ):
        extract_pdf_pages(
            missing_path
        )


def test_non_pdf_extension_raises_path_error(
    tmp_path: Path,
) -> None:
    text_path = (
        tmp_path / "document.txt"
    )

    text_path.write_text(
        "Not a PDF",
        encoding="utf-8",
    )

    with pytest.raises(
        PDFPathError,
        match=r"must have a \.pdf extension",
    ):
        extract_pdf_pages(
            text_path
        )


def test_malformed_pdf_raises_extraction_error(
    tmp_path: Path,
) -> None:
    malformed_pdf_path = (
        tmp_path / "malformed.pdf"
    )

    malformed_pdf_path.write_bytes(
        b"%PDF-This is not a complete PDF"
    )

    with pytest.raises(
        PDFExtractionError,
        match="could not extract text",
    ):
        extract_pdf_pages(
            malformed_pdf_path
        )