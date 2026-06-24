import asyncio
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import UUID

import aiofiles
from fastapi import UploadFile

from app.core.config import get_settings
from app.core.exceptions import (
    FileStorageError,
    FileTooLargeError,
    InvalidFileError,
    UnsupportedFileTypeError,
)


settings = get_settings()


ALLOWED_PDF_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
}

PDF_SIGNATURE = b"%PDF-"


@dataclass(frozen=True)
class StoredFile:
    """
    Structured information about a successfully stored file.
    """

    internal_filename: str
    original_filename: str
    storage_path: str
    absolute_path: Path
    size_bytes: int


def ensure_upload_directory() -> Path:
    """
    Create the configured upload directory if it does not exist.

    Returns the absolute upload-directory path.
    """

    settings.upload_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return settings.upload_path


def sanitize_original_filename(
    filename: str | None,
) -> str:
    """
    Remove path components from a client-provided filename.

    Both Windows backslashes and POSIX forward slashes are normalized
    before extracting only the final filename.
    """

    if filename is None:
        raise InvalidFileError(
            "The uploaded file does not have a filename."
        )

    normalized_filename = filename.replace("\\", "/")

    safe_filename = PurePosixPath(
        normalized_filename
    ).name.strip()

    if safe_filename in {"", ".", ".."}:
        raise InvalidFileError(
            "The uploaded file has an invalid or empty filename."
        )

    if len(safe_filename) > 255:
        raise InvalidFileError(
            "The uploaded filename is longer than 255 characters."
        )

    if any(
        ord(character) < 32
        for character in safe_filename
    ):
        raise InvalidFileError(
            "The uploaded filename contains invalid control characters."
        )

    return safe_filename


def validate_pdf_metadata(
    upload: UploadFile,
) -> str:
    """
    Validate the original filename, extension and declared MIME type.

    Returns the sanitized original filename when validation succeeds.
    """

    safe_filename = sanitize_original_filename(
        upload.filename
    )

    extension = Path(
        safe_filename
    ).suffix.lower()

    if extension != ".pdf":
        raise UnsupportedFileTypeError(
            "Only files with the .pdf extension are supported."
        )

    declared_content_type = (
        upload.content_type
        .split(";", maxsplit=1)[0]
        .strip()
        .lower()
        if upload.content_type
        else ""
    )

    if declared_content_type not in ALLOWED_PDF_CONTENT_TYPES:
        raise UnsupportedFileTypeError(
            "The uploaded file does not have an accepted PDF content type."
        )

    return safe_filename


async def delete_local_file(
    path: Path,
) -> None:
    """
    Delete a file if it exists.

    The filesystem operation is delegated to a worker thread so it
    does not directly block the main asyncio event loop.
    """

    await asyncio.to_thread(
        path.unlink,
        missing_ok=True,
    )


async def store_pdf_locally(
    upload: UploadFile,
    document_id: UUID,
    *,
    upload_directory: Path | None = None,
    max_size_bytes: int | None = None,
    chunk_size_bytes: int | None = None,
) -> StoredFile:
    """
    Validate and save one uploaded PDF to local storage.

    The optional keyword arguments allow unit tests to use temporary
    directories and artificially small file-size limits.
    """

    safe_original_filename = validate_pdf_metadata(
        upload
    )

    target_directory = (
        upload_directory.resolve()
        if upload_directory is not None
        else settings.upload_path
    )

    maximum_size = (
        max_size_bytes
        if max_size_bytes is not None
        else settings.max_upload_size_bytes
    )

    chunk_size = (
        chunk_size_bytes
        if chunk_size_bytes is not None
        else settings.upload_chunk_size_bytes
    )

    if maximum_size <= 0:
        raise FileStorageError(
            "The maximum upload size must be greater than zero."
        )

    if chunk_size <= 0:
        raise FileStorageError(
            "The upload chunk size must be greater than zero."
        )

    await asyncio.to_thread(
        target_directory.mkdir,
        parents=True,
        exist_ok=True,
    )

    internal_filename = f"{document_id}.pdf"

    target_path = (
        target_directory / internal_filename
    ).resolve()

    if target_path.parent != target_directory:
        raise FileStorageError(
            "The generated file path escaped the upload directory."
        )

    reported_size = getattr(
        upload,
        "size",
        None,
    )

    if (
        reported_size is not None
        and reported_size > maximum_size
    ):
        raise FileTooLargeError(
            "The PDF exceeds the maximum allowed upload size."
        )

    await upload.seek(0)

    first_chunk = await upload.read(
        chunk_size
    )

    if not first_chunk:
        raise InvalidFileError(
            "The uploaded PDF is empty."
        )

    if not first_chunk.startswith(
        PDF_SIGNATURE
    ):
        raise UnsupportedFileTypeError(
            "The uploaded file does not contain a valid PDF signature."
        )

    total_bytes = 0

    try:
        async with aiofiles.open(
            target_path,
            mode="xb",
        ) as output_file:
            current_chunk = first_chunk

            while current_chunk:
                total_bytes += len(
                    current_chunk
                )

                if total_bytes > maximum_size:
                    raise FileTooLargeError(
                        "The PDF exceeds the maximum allowed upload size."
                    )

                await output_file.write(
                    current_chunk
                )

                current_chunk = await upload.read(
                    chunk_size
                )

    except FileTooLargeError:
        await delete_local_file(
            target_path
        )
        raise

    except FileExistsError as exc:
        raise FileStorageError(
            "A file with the generated internal name already exists."
        ) from exc

    except OSError as exc:
        await delete_local_file(
            target_path
        )

        raise FileStorageError(
            "The PDF could not be written to local storage."
        ) from exc

    if upload_directory is None:
        relative_path = (
            Path(settings.upload_directory)
            / internal_filename
        )

        storage_path = str(
            relative_path
        ).replace("\\", "/")

    else:
        storage_path = str(
            target_path
        ).replace("\\", "/")

    return StoredFile(
        internal_filename=internal_filename,
        original_filename=safe_original_filename,
        storage_path=storage_path,
        absolute_path=target_path,
        size_bytes=total_bytes,
    )