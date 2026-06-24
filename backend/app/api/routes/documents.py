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
    status
)

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    FileStorageError,
    FileTooLargeError,
    InvalidFileError,
    UnsupportedFileTypeError,
)

from app.db.session import get_db_session
from app.repositories.document import (
    create_document,
    get_document_by_id,
    list_documents
)

from app.schemas.document import DocumentResponse
from app.services.local_storage import (
    StoredFile,
    delete_local_file,
    store_pdf_locally,
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
                "The maximum allowed size is configured by the server"
            )
        )
    ],
    session: Annotated[
        AsyncSession,
        Depends(get_db_session),
    ]
) -> DocumentResponse:
    """
    Upload, validate and persist one PDF document.
    
    The PDF is first stored on disk. Its metadata is then saved
    in PostgreSQL. If the database operation fails, the stored
    file is removed to avoid leaving and orphan file.
    """
    
    document_id = uuid4()
    
    stored_file : StoredFile | None = None
    
    try:
        stored_file = await store_pdf_locally(
            upload=file,
            document_id=document_id
        )
        
        document = await create_document(
            session=session,
            document_id=document_id,
            filename=stored_file.internal_filename,
            original_filename=stored_file.original_filename,
            content_type=(
                file.content_type or "applications/pdf"
            ),
            storage_path=stored_file.storage_path,
            status="uploaded"
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
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc)
        ) from exc
    
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(exc)
        ) from exc
    
    except FileStorageError as exc:
        logger.exception(
            "The uploaded PDF could not be stored locally."
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The document could not be stored."
        ) from exc
        
    except SQLAlchemyError as exc:
        logger.exception(
            "The document metadata could not be saved to PostgreSQL."
        )
        
        if stored_file is not None:
            await delete_local_file(
                stored_file.absolute_path
            )
            
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The document metadata could not be saved."
            )
        ) from exc
        
    finally:
        await file.close()
        

@router.get(
    "",
    response_model=list[DocumentResponse],
    summary="List uploaded documents",
    description=(
        "Return uploaded documents ordered from newest to oldest."
    ),
)
async def get_documents(
    session: Annotated[
        AsyncSession,
        Depends(get_db_session),
    ],
    limit:Annotated[
        int,
        Query(
            ge=1,
            le=100,
            description=(
                "Maximum number of documents to return."
            ),
        ),
    ] = 20,
    offset: Annotated[
        int,
        Query(
            ge=0,
            description=(
                "Number of earlier documents to skip."
            ),
        ),
    ]=0,
) -> list[DocumentResponse]:
    """
    Return a paginated list of uploaded documents.
    """
    
    documents = await list_documents(
        session=session,
        limit=limit,
        offset=offset,
    )
    
    return [
        DocumentResponse.model_validate(document) for document in documents
    ]
    
    
@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get one uploaded document",
    description=(
        "Retrieve document metadata using its UUID."
    )
)
async def get_documents(
    document_id:UUID,
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