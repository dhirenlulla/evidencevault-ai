import logging
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    DocumentAlreadyProcessingError,
    DocumentNotFoundError,
    FileStorageError,
    FileTooLargeError,
    InvalidFileError,
    NoChunksGeneratedError,
    UnsupportedFileTypeError,
)
from app.db.session import get_db_session
from app.repositories.document import (
    create_document,
    get_document_by_id,
    list_documents,
)
from app.repositories.document_chunk import (
    count_document_chunks,
    list_document_chunks as fetch_document_chunks,
)
from app.schemas.document import DocumentResponse
from app.schemas.document_chunk import (
    DocumentChunkListResponse,
    DocumentChunkResponse,
)
from app.schemas.document_processing import (
    DocumentProcessingResponse,
)
from app.services.document_chunk_persistence import (
    generate_and_persist_document_chunks,
)
from app.services.document_processing import (
    process_document as run_document_processing,
)
from app.services.local_storage import (
    StoredFile,
    delete_local_file,
    store_pdf_locally,
)

from app.core.dependencies import (
    get_retrieval_service,
    get_generation_service,
)

from app.schemas.retrieval import (
    RetrievalResult,
)

from app.schemas.retrieval_request import (
    RetrievalRequest,
)

from app.services.retrieval import (
    RetrievalService,
)

from app.schemas.answer import (
    AnswerResponse,
    AnswerRequest,
)

from app.services.generation import (
    GenerationService,
)

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF document",
    description=(
        "Validate a PDF, store it locally, and create "
        "its metadata record in PostgreSQL."
    ),
)
async def upload_document(
    file: Annotated[
        UploadFile,
        File(
            description=(
                "A PDF document to upload. "
                "The maximum allowed size is configured "
                "by the server."
            )
        ),
    ],
    session: Annotated[
        AsyncSession,
        Depends(get_db_session),
    ],
) -> DocumentResponse:
    """
    Upload, validate, store, and register one PDF document.

    The PDF is stored on disk first. Its metadata is then
    persisted in PostgreSQL.

    If PostgreSQL persistence fails after the file has been
    written, the stored file is deleted to prevent an orphan.
    """

    document_id = uuid4()

    stored_file: StoredFile | None = None

    try:
        stored_file = await store_pdf_locally(
            upload=file,
            document_id=document_id,
        )

        document = await create_document(
            session=session,
            document_id=document_id,
            filename=stored_file.internal_filename,
            original_filename=(
                stored_file.original_filename
            ),
            content_type=(
                file.content_type
                or "application/pdf"
            ),
            storage_path=stored_file.storage_path,
            status="uploaded",
        )

        return DocumentResponse.model_validate(
            document
        )

    except InvalidFileError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
            ),
            detail=str(exc),
        ) from exc

    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_413_CONTENT_TOO_LARGE
            ),
            detail=str(exc),
        ) from exc

    except FileStorageError as exc:
        logger.exception(
            "The uploaded PDF could not be "
            "stored locally."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "The document could not be stored."
            ),
        ) from exc

    except SQLAlchemyError as exc:
        logger.exception(
            "The document metadata could not be "
            "saved to PostgreSQL."
        )

        if stored_file is not None:
            await delete_local_file(
                stored_file.absolute_path
            )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "The document metadata could not "
                "be saved."
            ),
        ) from exc

    finally:
        await file.close()


@router.get(
    "",
    response_model=list[DocumentResponse],
    summary="List uploaded documents",
    description=(
        "Return uploaded documents ordered from "
        "newest to oldest."
    ),
)
async def list_documents_endpoint(
    session: Annotated[
        AsyncSession,
        Depends(get_db_session),
    ],
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=100,
            description=(
                "Maximum number of documents "
                "to return."
            ),
        ),
    ] = 20,
    offset: Annotated[
        int,
        Query(
            ge=0,
            description=(
                "Number of earlier documents "
                "to skip."
            ),
        ),
    ] = 0,
) -> list[DocumentResponse]:
    """
    Return a paginated collection of uploaded documents.
    """

    documents = await list_documents(
        session=session,
        limit=limit,
        offset=offset,
    )

    return [
        DocumentResponse.model_validate(
            document
        )
        for document in documents
    ]


@router.post(
    "/{document_id}/process",
    response_model=DocumentProcessingResponse,
    summary="Process and chunk a document",
    description=(
        "Classify the PDF, extract page-aware text, "
        "generate deterministic chunks, and persist "
        "those chunks in PostgreSQL."
    ),
)
async def process_document_endpoint(
    document_id: UUID,
    session: Annotated[
        AsyncSession,
        Depends(get_db_session),
    ],
) -> DocumentProcessingResponse:
    """
    Run the complete document-processing and chunking pipeline.

    The workflow first classifies and extracts the PDF. When
    usable text is available, the resulting chunks are generated
    and persisted transactionally in PostgreSQL.
    """

    try:
        processing_result = (
            await run_document_processing(
                session=session,
                document_id=document_id,
            )
        )

        if not (
            processing_result
            .can_continue_to_chunking
        ):
            return DocumentProcessingResponse(
                document_id=document_id,
                status=(
                    processing_result.status.value
                ),
                classification=(
                    processing_result
                    .classification.value
                    if (
                        processing_result
                        .classification
                        is not None
                    )
                    else None
                ),
                page_count=(
                    processing_result.page_count
                ),
                extractable_page_count=(
                    processing_result
                    .extractable_page_count
                ),
                image_only_page_count=(
                    processing_result
                    .image_only_page_count
                ),
                empty_page_count=(
                    processing_result
                    .empty_page_count
                ),
                total_characters=(
                    processing_result
                    .total_characters
                ),
                total_words=(
                    processing_result.total_words
                ),
                chunk_count=0,
                ready_for_indexing=False,
                message=(
                    processing_result.message
                ),
            )

        persisted_result = await (
            generate_and_persist_document_chunks(
                session=session,
                document_id=document_id,
            )
        )

        return DocumentProcessingResponse(
            document_id=document_id,
            status=(
                persisted_result.status.value
            ),
            classification=(
                processing_result
                .classification.value
                if (
                    processing_result
                    .classification
                    is not None
                )
                else None
            ),
            page_count=(
                processing_result.page_count
            ),
            extractable_page_count=(
                processing_result
                .extractable_page_count
            ),
            image_only_page_count=(
                processing_result
                .image_only_page_count
            ),
            empty_page_count=(
                processing_result
                .empty_page_count
            ),
            total_characters=(
                processing_result
                .total_characters
            ),
            total_words=(
                processing_result.total_words
            ),
            chunk_count=(
                persisted_result.chunk_count
            ),
            ready_for_indexing=True,
            message=(
                f"{processing_result.message} "
                f"Generated and persisted "
                f"{persisted_result.chunk_count} "
                f"chunks."
            ),
        )

    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except DocumentAlreadyProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except NoChunksGeneratedError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_CONTENT
            ),
            detail=str(exc),
        ) from exc

    except SQLAlchemyError as exc:
        logger.exception(
            "Document processing could not be "
            "persisted in PostgreSQL."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Document processing could not "
                "be completed."
            ),
        ) from exc


@router.get(
    "/{document_id}/chunks",
    response_model=DocumentChunkListResponse,
    summary="List persisted document chunks",
    description=(
        "Return citation-ready text chunks ordered "
        "by their original document position."
    ),
)
async def get_document_chunks_endpoint(
    document_id: UUID,
    session: Annotated[
        AsyncSession,
        Depends(get_db_session),
    ],
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=100,
            description=(
                "Maximum number of chunks "
                "to return."
            ),
        ),
    ] = 50,
    offset: Annotated[
        int,
        Query(
            ge=0,
            description=(
                "Number of earlier chunks "
                "to skip."
            ),
        ),
    ] = 0,
) -> DocumentChunkListResponse:
    """
    Return one document's persisted chunks with pagination.
    """

    document = await get_document_by_id(
        session=session,
        document_id=document_id,
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    chunks = await fetch_document_chunks(
        session=session,
        document_id=document_id,
        limit=limit,
        offset=offset,
    )

    total = await count_document_chunks(
        session=session,
        document_id=document_id,
    )

    return DocumentChunkListResponse(
        document_id=document_id,
        total=total,
        limit=limit,
        offset=offset,
        chunks=[
            DocumentChunkResponse.model_validate(
                chunk
            )
            for chunk in chunks
        ],
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get one uploaded document",
    description=(
        "Retrieve document metadata using its UUID."
    ),
)
async def get_document_endpoint(
    document_id: UUID,
    session: Annotated[
        AsyncSession,
        Depends(get_db_session),
    ],
) -> DocumentResponse:
    """
    Retrieve one document using its PostgreSQL UUID.
    """

    document = await get_document_by_id(
        session=session,
        document_id=document_id,
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return DocumentResponse.model_validate(
        document
    )
    
    
@router.post(
    "/{document_id}/retrieve",
    response_model=RetrievalResult,
    summary="Retrieve relevant document chunks.",
    description=(
        "Perform semantic search against one "
        "document using vector similarity."
    ),
)
async def retrieve_document_chunks(
    document_id: UUID,
    request: RetrievalRequest,
    retrieval_service: Annotated[
        RetrievalService,
        Depends(get_retrieval_service),
    ],
) -> RetrievalResult:
    """ 
    Retrieve the most relevant chunks from one 
    processed document.
    """
    
    return await retrieval_service.retrieve(
        document_id=document_id,
        query=request.query,
        limit=request.limit,
    )
    
@router.post(
    "/{document_id}/answer",
    response_model=AnswerResponse,
    summary="Generate a grounded answer.",
    description=(
        "Retrieve relevant document chunks and "
        "generate a grounded answer using the "
        "configured language model."
    ),
)
async def generate_document_answer(
    document_id: UUID,
    request: AnswerRequest,
    generation_service: Annotated[
        GenerationService,
        Depends(get_generation_service),
    ],
) -> AnswerResponse:
    """ 
    Execute the complete RAG pipeline.
    """
    
    return await generation_service.generate_answer(
        document_id=document_id,
        query=request.query,
    )