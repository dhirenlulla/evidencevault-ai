import base64
from pathlib import Path

import pymupdf
import pytest

from app.core.exceptions import (
    PDFEncryptedError,
    PDFMalformedError,
)
from app.services.pdf_classification import (
    PDFDocumentClassification,
    PDFPageClassification,
    classify_pdf_document,
)
from app.services.pdf_extraction import (
    extract_pdf_pages,
)


ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
    "CAQAAAC1HAwCAAAAC0lEQVR42mNk+A8A"
    "AQUBAScY42YAAAAASUVORK5CYII="
)


def create_text_pdf(
    pdf_path: Path,
    page_texts: list[str],
) -> None:
    """
    Create a PDF containing selectable text pages.
    """

    with pymupdf.open() as document:
        for page_text in page_texts:
            page = document.new_page(
                width=300,
                height=300,
            )

            if page_text:
                page.insert_text(
                    (40, 60),
                    page_text,
                )

        document.save(
            pdf_path
        )


def create_image_only_pdf(
    pdf_path: Path,
    *,
    page_count: int = 1,
) -> None:
    """
    Create pages covered by raster images without text.
    """

    with pymupdf.open() as document:
        for _ in range(page_count):
            page = document.new_page(
                width=300,
                height=300,
            )

            page.insert_image(
                page.rect,
                stream=ONE_PIXEL_PNG,
            )

        document.save(
            pdf_path
        )


def create_encrypted_pdf(
    pdf_path: Path,
) -> None:
    """
    Create a password-protected PDF.
    """

    with pymupdf.open() as document:
        page = document.new_page()

        page.insert_text(
            (72, 72),
            "Protected EvidenceVault content",
        )

        document.save(
            pdf_path,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner-password",
            user_pw="user-password",
        )


def test_text_pdf_is_classified_as_text_based(
    tmp_path: Path,
) -> None:
    pdf_path = (
        tmp_path / "text-document.pdf"
    )

    create_text_pdf(
        pdf_path,
        [
            (
                "This is the first page with enough "
                "extractable text for classification."
            ),
            (
                "This is the second page with enough "
                "extractable text for classification."
            ),
        ],
    )

    result = classify_pdf_document(
        pdf_path
    )

    assert (
        result.classification
        == PDFDocumentClassification.TEXT_BASED
    )

    assert result.page_count == 2
    assert result.extractable_page_count == 2
    assert result.image_only_page_count == 0
    assert result.empty_page_count == 0
    assert result.has_usable_text is True
    assert result.requires_ocr is False
    assert result.can_continue_to_chunking is True


def test_blank_pdf_is_classified_as_empty(
    tmp_path: Path,
) -> None:
    pdf_path = (
        tmp_path / "blank.pdf"
    )

    create_text_pdf(
        pdf_path,
        [""],
    )

    result = classify_pdf_document(
        pdf_path
    )

    assert (
        result.classification
        == PDFDocumentClassification.EMPTY
    )

    assert result.page_count == 1
    assert result.extractable_page_count == 0
    assert result.image_only_page_count == 0
    assert result.empty_page_count == 1

    assert (
        result.pages[0].classification
        == PDFPageClassification.EMPTY
    )

    assert result.has_usable_text is False
    assert result.requires_ocr is False
    assert result.can_continue_to_chunking is False


def test_image_pdf_is_classified_as_scanned_or_image_only(
    tmp_path: Path,
) -> None:
    pdf_path = (
        tmp_path / "image-document.pdf"
    )

    create_image_only_pdf(
        pdf_path,
        page_count=2,
    )

    result = classify_pdf_document(
        pdf_path
    )

    assert (
        result.classification
        == (
            PDFDocumentClassification
            .SCANNED_OR_IMAGE_ONLY
        )
    )

    assert result.page_count == 2
    assert result.extractable_page_count == 0
    assert result.image_only_page_count == 2
    assert result.empty_page_count == 0
    assert result.requires_ocr is True
    assert result.has_usable_text is False
    assert result.can_continue_to_chunking is False

    assert all(
        page.classification
        == PDFPageClassification.IMAGE_ONLY
        for page in result.pages
    )

    assert all(
        page.image_coverage_ratio >= 0.50
        for page in result.pages
    )


def test_mixed_pdf_is_partially_extractable(
    tmp_path: Path,
) -> None:
    pdf_path = (
        tmp_path / "partial-document.pdf"
    )

    with pymupdf.open() as document:
        text_page = document.new_page(
            width=300,
            height=300,
        )

        text_page.insert_text(
            (40, 60),
            (
                "This page contains enough extractable "
                "text for the EvidenceVault pipeline."
            ),
        )

        image_page = document.new_page(
            width=300,
            height=300,
        )

        image_page.insert_image(
            image_page.rect,
            stream=ONE_PIXEL_PNG,
        )

        document.save(
            pdf_path
        )

    result = classify_pdf_document(
        pdf_path
    )

    assert (
        result.classification
        == (
            PDFDocumentClassification
            .PARTIALLY_EXTRACTABLE
        )
    )

    assert result.page_count == 2
    assert result.extractable_page_count == 1
    assert result.image_only_page_count == 1
    assert result.has_usable_text is True
    assert result.requires_ocr is True
    assert result.can_continue_to_chunking is True


def test_encrypted_pdf_is_classified_without_password(
    tmp_path: Path,
) -> None:
    pdf_path = (
        tmp_path / "encrypted.pdf"
    )

    create_encrypted_pdf(
        pdf_path
    )

    result = classify_pdf_document(
        pdf_path
    )

    assert (
        result.classification
        == PDFDocumentClassification.ENCRYPTED
    )

    assert result.requires_password is True
    assert result.has_usable_text is False
    assert result.can_continue_to_chunking is False
    assert result.pages == ()


def test_malformed_pdf_returns_malformed_classification(
    tmp_path: Path,
) -> None:
    pdf_path = (
        tmp_path / "malformed.pdf"
    )

    pdf_path.write_bytes(
        b"This is not a valid PDF document."
    )

    result = classify_pdf_document(
        pdf_path
    )

    assert (
        result.classification
        == PDFDocumentClassification.MALFORMED
    )

    assert result.page_count == 0
    assert result.pages == ()
    assert result.has_usable_text is False
    assert result.can_continue_to_chunking is False
    assert result.error_message is not None


def test_extraction_raises_specific_encrypted_error(
    tmp_path: Path,
) -> None:
    pdf_path = (
        tmp_path / "encrypted-extraction.pdf"
    )

    create_encrypted_pdf(
        pdf_path
    )

    with pytest.raises(
        PDFEncryptedError,
        match="requires a password",
    ):
        extract_pdf_pages(
            pdf_path
        )


def test_extraction_raises_specific_malformed_error(
    tmp_path: Path,
) -> None:
    pdf_path = (
        tmp_path / "malformed-extraction.pdf"
    )

    pdf_path.write_bytes(
        b"Not a valid PDF"
    )

    with pytest.raises(
        PDFMalformedError,
        match="could not extract text",
    ):
        extract_pdf_pages(
            pdf_path
        )