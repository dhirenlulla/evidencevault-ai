import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

import app.services.document_processing as processing_module
from app.core.exceptions import (
    DocumentAlreadyProcessingError,
    DocumentNotFoundError,
    DocumentStoragePathError,
)
from app.repositories.document import (
    update_document_processing_state,
)
from app.services.document_processing import (
    DocumentProcessingStatus,
    process_document,
    resolve_document_storage_path,
)
from app.services.pdf_classification import (
    PDFDocumentAnalysis,
    PDFDocumentClassification,
)
from app.services.pdf_extraction import (
    ExtractedDocument,
    ExtractedPage,
)


def build_document(
    *,
    status: str = "uploaded",
    storage_path: str | None = "uploads/test.pdf",
):
    """
    Create a lightweight document-like object.
    """

    return SimpleNamespace(
        id=uuid4(),
        status=status,
        storage_path=storage_path,
        page_count=None,
        error_message=None,
    )


def build_analysis(
    classification: PDFDocumentClassification,
    *,
    page_count: int = 2,
    extractable_page_count: int = 2,
    image_only_page_count: int = 0,
    empty_page_count: int = 0,
    error_message: str | None = None,
) -> PDFDocumentAnalysis:
    """
    Create a controlled classification result.
    """

    return PDFDocumentAnalysis(
        source_path=Path("test.pdf"),
        classification=classification,
        page_count=page_count,
        pages=(),
        extractable_page_count=(
            extractable_page_count
        ),
        image_only_page_count=(
            image_only_page_count
        ),
        empty_page_count=empty_page_count,
        total_text_characters=100,
        total_words=20,
        was_repaired=False,
        reason="Controlled classification result.",
        error_message=error_message,
    )


def build_extraction() -> ExtractedDocument:
    """
    Create a controlled extraction result.
    """

    pages = (
        ExtractedPage(
            page_number=1,
            text="First page text",
            character_count=15,
            word_count=3,
            is_empty=False,
        ),
        ExtractedPage(
            page_number=2,
            text="Second page text",
            character_count=16,
            word_count=3,
            is_empty=False,
        ),
    )

    return ExtractedDocument(
        source_path=Path("test.pdf"),
        page_count=2,
        pages=pages,
        total_characters=31,
        total_words=6,
        empty_page_numbers=(),
    )


def install_common_workflow_mocks(
    monkeypatch,
    document,
):
    """
    Install shared repository and path substitutes.
    """

    get_mock = AsyncMock(
        return_value=document
    )

    async def fake_update(
        session,
        *,
        document,
        status,
        page_count,
        error_message,
    ):
        document.status = status
        document.page_count = page_count
        document.error_message = error_message
        return document

    update_mock = AsyncMock(
        side_effect=fake_update
    )

    monkeypatch.setattr(
        processing_module,
        "get_document_by_id",
        get_mock,
    )

    monkeypatch.setattr(
        processing_module,
        "update_document_processing_state",
        update_mock,
    )

    monkeypatch.setattr(
        processing_module,
        "resolve_document_storage_path",
        lambda storage_path: Path("test.pdf"),
    )

    return get_mock, update_mock


def test_text_document_becomes_extracted(
    monkeypatch,
) -> None:
    document = build_document()

    _, update_mock = install_common_workflow_mocks(
        monkeypatch,
        document,
    )

    monkeypatch.setattr(
        processing_module,
        "classify_pdf_document",
        lambda path: build_analysis(
            PDFDocumentClassification.TEXT_BASED
        ),
    )

    monkeypatch.setattr(
        processing_module,
        "extract_pdf_pages",
        lambda path: build_extraction(),
    )

    result = asyncio.run(
        process_document(
            session=AsyncMock(
                spec=AsyncSession
            ),
            document_id=document.id,
        )
    )

    assert (
        result.status
        == DocumentProcessingStatus.EXTRACTED
    )

    assert result.page_count == 2
    assert result.total_characters == 31
    assert result.total_words == 6
    assert result.can_continue_to_chunking is True

    assert update_mock.await_count == 2

    assert (
        update_mock.await_args_list[0]
        .kwargs["status"]
        == "processing"
    )

    assert (
        update_mock.await_args_list[1]
        .kwargs["status"]
        == "extracted"
    )


def test_partial_document_becomes_extracted_with_warnings(
    monkeypatch,
) -> None:
    document = build_document()

    _, update_mock = install_common_workflow_mocks(
        monkeypatch,
        document,
    )

    monkeypatch.setattr(
        processing_module,
        "classify_pdf_document",
        lambda path: build_analysis(
            (
                PDFDocumentClassification
                .PARTIALLY_EXTRACTABLE
            ),
            extractable_page_count=1,
            image_only_page_count=1,
        ),
    )

    monkeypatch.setattr(
        processing_module,
        "extract_pdf_pages",
        lambda path: build_extraction(),
    )

    result = asyncio.run(
        process_document(
            session=AsyncMock(
                spec=AsyncSession
            ),
            document_id=document.id,
        )
    )

    assert result.status == (
        DocumentProcessingStatus
        .EXTRACTED_WITH_WARNINGS
    )

    assert result.image_only_page_count == 1
    assert result.can_continue_to_chunking is True

    assert (
        update_mock.await_args_list[1]
        .kwargs["status"]
        == "extracted_with_warnings"
    )


def test_image_only_document_becomes_ocr_required(
    monkeypatch,
) -> None:
    document = build_document()

    _, update_mock = install_common_workflow_mocks(
        monkeypatch,
        document,
    )

    monkeypatch.setattr(
        processing_module,
        "classify_pdf_document",
        lambda path: build_analysis(
            (
                PDFDocumentClassification
                .SCANNED_OR_IMAGE_ONLY
            ),
            extractable_page_count=0,
            image_only_page_count=2,
        ),
    )

    extraction_mock = Mock()

    monkeypatch.setattr(
        processing_module,
        "extract_pdf_pages",
        extraction_mock,
    )

    result = asyncio.run(
        process_document(
            session=AsyncMock(
                spec=AsyncSession
            ),
            document_id=document.id,
        )
    )

    assert result.status == (
        DocumentProcessingStatus.OCR_REQUIRED
    )

    assert result.can_continue_to_chunking is False

    assert (
        update_mock.await_args_list[1]
        .kwargs["status"]
        == "ocr_required"
    )

    extraction_mock.assert_not_called()


def test_encrypted_document_becomes_password_required(
    monkeypatch,
) -> None:
    document = build_document()

    _, update_mock = install_common_workflow_mocks(
        monkeypatch,
        document,
    )

    monkeypatch.setattr(
        processing_module,
        "classify_pdf_document",
        lambda path: build_analysis(
            PDFDocumentClassification.ENCRYPTED,
            extractable_page_count=0,
        ),
    )

    result = asyncio.run(
        process_document(
            session=AsyncMock(
                spec=AsyncSession
            ),
            document_id=document.id,
        )
    )

    assert result.status == (
        DocumentProcessingStatus
        .PASSWORD_REQUIRED
    )

    assert (
        update_mock.await_args_list[1]
        .kwargs["status"]
        == "password_required"
    )


def test_malformed_document_becomes_failed(
    monkeypatch,
) -> None:
    document = build_document()

    _, update_mock = install_common_workflow_mocks(
        monkeypatch,
        document,
    )

    monkeypatch.setattr(
        processing_module,
        "classify_pdf_document",
        lambda path: build_analysis(
            PDFDocumentClassification.MALFORMED,
            page_count=0,
            extractable_page_count=0,
            error_message="Invalid PDF structure.",
        ),
    )

    result = asyncio.run(
        process_document(
            session=AsyncMock(
                spec=AsyncSession
            ),
            document_id=document.id,
        )
    )

    assert (
        result.status
        == DocumentProcessingStatus.FAILED
    )

    assert result.can_continue_to_chunking is False

    assert (
        update_mock.await_args_list[1]
        .kwargs["status"]
        == "failed"
    )


def test_missing_document_raises_not_found(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        processing_module,
        "get_document_by_id",
        AsyncMock(
            return_value=None
        ),
    )

    with pytest.raises(
        DocumentNotFoundError,
        match="Document not found",
    ):
        asyncio.run(
            process_document(
                session=AsyncMock(
                    spec=AsyncSession
                ),
                document_id=uuid4(),
            )
        )


def test_already_processing_document_is_rejected(
    monkeypatch,
) -> None:
    document = build_document(
        status="processing"
    )

    monkeypatch.setattr(
        processing_module,
        "get_document_by_id",
        AsyncMock(
            return_value=document
        ),
    )

    with pytest.raises(
        DocumentAlreadyProcessingError,
        match="already being processed",
    ):
        asyncio.run(
            process_document(
                session=AsyncMock(
                    spec=AsyncSession
                ),
                document_id=document.id,
            )
        )


def test_storage_path_cannot_escape_upload_directory(
    tmp_path: Path,
) -> None:
    upload_directory = (
        tmp_path / "uploads"
    )

    upload_directory.mkdir()

    outside_pdf = (
        tmp_path / "outside.pdf"
    )

    outside_pdf.write_bytes(
        b"%PDF-1.4"
    )

    with pytest.raises(
        DocumentStoragePathError,
        match="outside",
    ):
        resolve_document_storage_path(
            "../outside.pdf",
            upload_directory=upload_directory,
        )


def test_repository_state_update_commits_and_refreshes() -> None:
    session = AsyncMock(
        spec=AsyncSession
    )

    document = build_document()

    result = asyncio.run(
        update_document_processing_state(
            session=session,
            document=document,
            status="processing",
            page_count=None,
            error_message=None,
        )
    )

    assert result is document
    assert document.status == "processing"

    session.commit.assert_awaited_once()
    session.refresh.assert_awaited_once_with(
        document
    )

    session.rollback.assert_not_awaited()


def test_repository_state_update_rolls_back_on_error() -> None:
    session = AsyncMock(
        spec=AsyncSession
    )

    session.commit.side_effect = (
        SQLAlchemyError(
            "Forced database failure"
        )
    )

    document = build_document()

    with pytest.raises(
        SQLAlchemyError,
        match="Forced database failure",
    ):
        asyncio.run(
            update_document_processing_state(
                session=session,
                document=document,
                status="processing",
                page_count=None,
                error_message=None,
            )
        )

    session.rollback.assert_awaited_once()