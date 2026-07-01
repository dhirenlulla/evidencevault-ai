from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import (
    DocumentAlreadyProcessingError,
    DocumentNotFoundError,
    DocumentStoragePathError,
    PDFProcessingError,
)
from app.repositories.document import (
    get_document_by_id,
    update_document_processing_state,
)
from app.services.pdf_classification import (
    PDFDocumentAnalysis,
    PDFDocumentClassification,
    classify_pdf_document,
)
from app.services.pdf_extraction import (
    ExtractedDocument,
    extract_pdf_pages,
)


settings = get_settings()


class DocumentProcessingStatus(str, Enum):
    """
    Processing states stored in PostgreSQL.
    """

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    EXTRACTED = "extracted"
    EXTRACTED_WITH_WARNINGS = (
        "extracted_with_warnings"
    )
    CHUNKED = "chunked"
    OCR_REQUIRED = "ocr_required"
    PASSWORD_REQUIRED = "password_required"
    FAILED = "failed"


@dataclass(
    frozen=True,
    slots=True,
)
class DocumentProcessingResult:
    """
    Structured result produced by one processing run.
    """

    document_id: UUID
    status: DocumentProcessingStatus
    classification: (
        PDFDocumentClassification | None
    )
    page_count: int
    extractable_page_count: int
    image_only_page_count: int
    empty_page_count: int
    total_characters: int
    total_words: int
    message: str

    @property
    def succeeded(self) -> bool:
        """
        Return True when usable text was extracted.
        """

        return self.status in {
            DocumentProcessingStatus.EXTRACTED,
            (
                DocumentProcessingStatus
                .EXTRACTED_WITH_WARNINGS
            ),
        }

    @property
    def can_continue_to_chunking(self) -> bool:
        """
        Return True when the extraction result can be chunked.
        """

        return self.succeeded


def resolve_document_storage_path(
    storage_path: str | None,
    *,
    upload_directory: Path | None = None,
) -> Path:
    """
    Resolve and validate a document's local storage path.

    The resolved path must remain inside the configured
    upload directory.
    """

    if not storage_path:
        raise DocumentStoragePathError(
            "The document does not have a storage path."
        )

    resolved_upload_directory = (
        upload_directory.resolve()
        if upload_directory is not None
        else settings.upload_path.resolve()
    )

    stored_path = Path(
        storage_path
    )

    if stored_path.is_absolute():
        resolved_path = stored_path.resolve()

    else:
        backend_directory = (
            resolved_upload_directory.parent
        )

        resolved_path = (
            backend_directory / stored_path
        ).resolve()

    try:
        resolved_path.relative_to(
            resolved_upload_directory
        )

    except ValueError as exc:
        raise DocumentStoragePathError(
            "The document storage path is outside "
            "the configured upload directory."
        ) from exc

    if not resolved_path.exists():
        raise DocumentStoragePathError(
            f"The stored PDF does not exist: "
            f"{resolved_path}"
        )

    if not resolved_path.is_file():
        raise DocumentStoragePathError(
            f"The stored PDF path is not a file: "
            f"{resolved_path}"
        )

    if resolved_path.suffix.lower() != ".pdf":
        raise DocumentStoragePathError(
            "The stored document does not have "
            "a .pdf extension."
        )

    return resolved_path


def build_non_extractable_result(
    *,
    document_id: UUID,
    status: DocumentProcessingStatus,
    analysis: PDFDocumentAnalysis,
    message: str,
) -> DocumentProcessingResult:
    """
    Build a result for a PDF that cannot continue to chunking.
    """

    return DocumentProcessingResult(
        document_id=document_id,
        status=status,
        classification=analysis.classification,
        page_count=analysis.page_count,
        extractable_page_count=(
            analysis.extractable_page_count
        ),
        image_only_page_count=(
            analysis.image_only_page_count
        ),
        empty_page_count=analysis.empty_page_count,
        total_characters=(
            analysis.total_text_characters
        ),
        total_words=analysis.total_words,
        message=message,
    )


def build_extracted_result(
    *,
    document_id: UUID,
    status: DocumentProcessingStatus,
    analysis: PDFDocumentAnalysis,
    extraction: ExtractedDocument,
    message: str,
) -> DocumentProcessingResult:
    """
    Build a successful page-extraction result.
    """

    return DocumentProcessingResult(
        document_id=document_id,
        status=status,
        classification=analysis.classification,
        page_count=extraction.page_count,
        extractable_page_count=(
            extraction.text_page_count
        ),
        image_only_page_count=(
            analysis.image_only_page_count
        ),
        empty_page_count=len(
            extraction.empty_page_numbers
        ),
        total_characters=(
            extraction.total_characters
        ),
        total_words=extraction.total_words,
        message=message,
    )


async def persist_processing_state(
    session: AsyncSession,
    *,
    document,
    status: DocumentProcessingStatus,
    page_count: int | None,
    error_message: str | None,
):
    """
    Persist one processing state using its string value.
    """

    return await update_document_processing_state(
        session=session,
        document=document,
        status=status.value,
        page_count=page_count,
        error_message=error_message,
    )


async def process_document(
    session: AsyncSession,
    document_id: UUID,
) -> DocumentProcessingResult:
    """
    Classify and extract one stored PDF document.

    The workflow persists each important status transition
    in PostgreSQL.
    """

    document = await get_document_by_id(
        session=session,
        document_id=document_id,
    )

    if document is None:
        raise DocumentNotFoundError(
            f"Document not found: {document_id}"
        )

    if (
        document.status
        == DocumentProcessingStatus.PROCESSING.value
    ):
        raise DocumentAlreadyProcessingError(
            "The document is already being processed."
        )

    await persist_processing_state(
        session=session,
        document=document,
        status=DocumentProcessingStatus.PROCESSING,
        page_count=None,
        error_message=None,
    )

    try:
        pdf_path = resolve_document_storage_path(
            document.storage_path
        )

        analysis = classify_pdf_document(
            pdf_path
        )

        if (
            analysis.classification
            == PDFDocumentClassification.ENCRYPTED
        ):
            await persist_processing_state(
                session=session,
                document=document,
                status=(
                    DocumentProcessingStatus
                    .PASSWORD_REQUIRED
                ),
                page_count=analysis.page_count,
                error_message=analysis.reason,
            )

            return build_non_extractable_result(
                document_id=document_id,
                status=(
                    DocumentProcessingStatus
                    .PASSWORD_REQUIRED
                ),
                analysis=analysis,
                message=analysis.reason,
            )

        if (
            analysis.classification
            == PDFDocumentClassification.MALFORMED
        ):
            failure_message = (
                analysis.error_message
                or analysis.reason
            )

            await persist_processing_state(
                session=session,
                document=document,
                status=DocumentProcessingStatus.FAILED,
                page_count=analysis.page_count,
                error_message=failure_message,
            )

            return build_non_extractable_result(
                document_id=document_id,
                status=DocumentProcessingStatus.FAILED,
                analysis=analysis,
                message=failure_message,
            )

        if (
            analysis.classification
            == PDFDocumentClassification.EMPTY
        ):
            await persist_processing_state(
                session=session,
                document=document,
                status=DocumentProcessingStatus.FAILED,
                page_count=analysis.page_count,
                error_message=analysis.reason,
            )

            return build_non_extractable_result(
                document_id=document_id,
                status=DocumentProcessingStatus.FAILED,
                analysis=analysis,
                message=analysis.reason,
            )

        if (
            analysis.classification
            == (
                PDFDocumentClassification
                .SCANNED_OR_IMAGE_ONLY
            )
        ):
            await persist_processing_state(
                session=session,
                document=document,
                status=(
                    DocumentProcessingStatus
                    .OCR_REQUIRED
                ),
                page_count=analysis.page_count,
                error_message=analysis.reason,
            )

            return build_non_extractable_result(
                document_id=document_id,
                status=(
                    DocumentProcessingStatus
                    .OCR_REQUIRED
                ),
                analysis=analysis,
                message=analysis.reason,
            )

        extraction = extract_pdf_pages(
            pdf_path
        )

        if (
            analysis.classification
            == (
                PDFDocumentClassification
                .PARTIALLY_EXTRACTABLE
            )
        ):
            final_status = (
                DocumentProcessingStatus
                .EXTRACTED_WITH_WARNINGS
            )

            message = (
                "Usable text was extracted, but one "
                "or more pages may require OCR."
            )

        else:
            final_status = (
                DocumentProcessingStatus.EXTRACTED
            )

            message = (
                "PDF text was extracted successfully."
            )

        await persist_processing_state(
            session=session,
            document=document,
            status=final_status,
            page_count=extraction.page_count,
            error_message=None,
        )

        return build_extracted_result(
            document_id=document_id,
            status=final_status,
            analysis=analysis,
            extraction=extraction,
            message=message,
        )

    except (
        DocumentStoragePathError,
        PDFProcessingError,
    ) as exc:
        await persist_processing_state(
            session=session,
            document=document,
            status=DocumentProcessingStatus.FAILED,
            page_count=None,
            error_message=str(exc),
        )

        return DocumentProcessingResult(
            document_id=document_id,
            status=DocumentProcessingStatus.FAILED,
            classification=None,
            page_count=0,
            extractable_page_count=0,
            image_only_page_count=0,
            empty_page_count=0,
            total_characters=0,
            total_words=0,
            message=str(exc),
        )