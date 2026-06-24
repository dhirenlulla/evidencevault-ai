import asyncio
from io import BytesIO
from uuid import uuid4

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from app.core.exceptions import (
    FileTooLargeError, 
    UnsupportedFileTypeError,
)

from app.services.local_storage import (
    sanitize_original_filename,
    store_pdf_locally,
)

VALID_MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj\n"
    b"<< /Type /Catalog >>\n"
    b"endobj\n"
    b"trailer\n"
    b"<<>>\n"
    b"%%EOF\n"
)

def create_upload_file(
    *,
    filename: str,
    content: bytes,
    content_type: str,
) -> UploadFile:
    """
    Create an in-memory UploadFile for unit testing.
    """
    
    return UploadFile(
        file=BytesIO(content),
        filename=filename,
        size=None,
        headers=Headers(
            {
                "content-type": content_type,
            }
        )
    )
    
    
def test_windows_path_is_removed_from_filename() -> None:
    filename = sanitize_original_filename(
        r"C:\Users\Test\Documents\employee-policy.pdf"
    )
    
    assert filename == "employee-policy.pdf"
    

def  test_valid_pdf_is_saved(
    tmp_path,
) -> None:
    async def run_test() -> None:
        upload = create_upload_file(
            filename="employee-policy.pdf",
            content=VALID_MINIMAL_PDF,
            content_type="application/pdf",
        )
        
        try:
            result = await store_pdf_locally(
                upload=upload,
                document_id=uuid4(),
                upload_directory=tmp_path,
                max_size_bytes=1024,
                chunk_size_bytes=16,
            )
            
            assert result.absolute_path.exists()
            
            assert (
                result.original_filename == "employee-policy.pdf"
            )
            
            assert result.internal_filename.endswith(
                ".pdf"
            )
            
            assert result.size_bytes == len(VALID_MINIMAL_PDF)
            
            saved_content = (
                result.absolute_path.read_bytes()
            )
            
        finally:
            await upload.close()
            
    asyncio.run(run_test())
    
    
    
def test_non_pdf_extension_is_rejected(
    tmp_path,
) -> None:
    
    async def run_test() -> None: 
        upload = create_upload_file(
            filename="employee-policy.txt",
            content=VALID_MINIMAL_PDF,
            content_type="application/pdf"
        )
        
        try:
            with pytest.raises(
                UnsupportedFileTypeError
            ):
                await store_pdf_locally(
                    upload=upload,
                    document_id=uuid4(),
                    upload_directory=tmp_path
                )
                
        finally:
            await upload.close()
            
    asyncio.run(run_test())
    
    
def test_invalid_content_type_is_rejected(
    tmp_path,
) -> None:
    async def run_test() -> None:
        upload = create_upload_file(
            filename="employee-policy.pdf",
            content=VALID_MINIMAL_PDF,
            content_type="text/plain",
        )

        try:
            with pytest.raises(
                UnsupportedFileTypeError
            ):
                await store_pdf_locally(
                    upload=upload,
                    document_id=uuid4(),
                    upload_directory=tmp_path,
                )

        finally:
            await upload.close()

    asyncio.run(run_test())
    
    
def test_invalid_pdf_signature_is_rejected(
    tmp_path,
) -> None:
    async def run_test() -> None:
        upload = create_upload_file(
            filename="fake.pdf",
            content=b"This is not a real PDF file.",
            content_type="application/pdf",
        )

        try:
            with pytest.raises(
                UnsupportedFileTypeError
            ):
                await store_pdf_locally(
                    upload=upload,
                    document_id=uuid4(),
                    upload_directory=tmp_path,
                )

            assert list(
                tmp_path.iterdir()
            ) == []

        finally:
            await upload.close()
            
    asyncio.run(run_test())
    
    
def test_oversized_pdf_is_removed(
    tmp_path,
) -> None:
    async def run_test() -> None:
        oversized_content = (
            VALID_MINIMAL_PDF
            + (b"A" * 500)
        )

        upload = create_upload_file(
            filename="large-policy.pdf",
            content=oversized_content,
            content_type="application/pdf",
        )

        try:
            with pytest.raises(
                FileTooLargeError
            ):
                await store_pdf_locally(
                    upload=upload,
                    document_id=uuid4(),
                    upload_directory=tmp_path,
                    max_size_bytes=100,
                    chunk_size_bytes=32,
                )

            assert list(
                tmp_path.iterdir()
            ) == []

        finally:
            await upload.close()

    asyncio.run(run_test())