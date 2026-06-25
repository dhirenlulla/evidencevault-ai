from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

import app.api.routes.documents as documents_route
from app.services.local_storage import (
    store_pdf_locally as real_store_pdf_locally,
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


def build_document(
    *,
    document_id: UUID | None = None,
    filename: str | None = None,
    original_filename: str = "employee-policy.pdf",
    content_type: str = "application/pdf",
    storage_path: str | None = None,
    status: str = "uploaded",
) -> SimpleNamespace:
    """
    Build an object that resembles the SQLAlchemy Document model.

    Pydantic's DocumentResponse can read this object because
    its configuration uses from_attributes=True.
    """

    resolved_id = document_id or uuid4()
    timestamp = datetime.now(
        timezone.utc
    )

    return SimpleNamespace(
        id=resolved_id,
        filename=(
            filename
            or f"{resolved_id}.pdf"
        ),
        original_filename=original_filename,
        content_type=content_type,
        storage_path=(
            storage_path
            or f"uploads/{resolved_id}.pdf"
        ),
        status=status,
        page_count=None,
        chunk_count=0,
        error_message=None,
        created_at=timestamp,
        updated_at=timestamp,
    )


def install_temporary_storage(
    monkeypatch,
    tmp_path: Path,
    *,
    max_size_bytes: int = 1_048_576,
) -> None:
    """
    Replace the route's storage function with a wrapper that uses
    pytest's temporary directory instead of backend/uploads.
    """

    async def store_in_temporary_directory(
        upload,
        document_id,
    ):
        return await real_store_pdf_locally(
            upload=upload,
            document_id=document_id,
            upload_directory=tmp_path,
            max_size_bytes=max_size_bytes,
            chunk_size_bytes=16,
        )

    monkeypatch.setattr(
        documents_route,
        "store_pdf_locally",
        store_in_temporary_directory,
    )


def test_successful_pdf_upload(
    client: TestClient,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    A valid PDF should be stored and returned as a new document.
    """

    install_temporary_storage(
        monkeypatch,
        tmp_path,
    )

    async def fake_create_document(
        session,
        *,
        document_id,
        filename,
        original_filename,
        content_type,
        storage_path,
        status,
    ):
        return build_document(
            document_id=document_id,
            filename=filename,
            original_filename=original_filename,
            content_type=content_type,
            storage_path=storage_path,
            status=status,
        )

    create_mock = AsyncMock(
        side_effect=fake_create_document
    )

    monkeypatch.setattr(
        documents_route,
        "create_document",
        create_mock,
    )

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "employee-policy.pdf",
                VALID_MINIMAL_PDF,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201

    response_data = response.json()

    assert (
        response_data["original_filename"]
        == "employee-policy.pdf"
    )

    assert (
        response_data["content_type"]
        == "application/pdf"
    )

    assert response_data["status"] == "uploaded"
    assert response_data["page_count"] is None
    assert response_data["chunk_count"] == 0
    assert response_data["error_message"] is None

    UUID(
        response_data["id"]
    )

    stored_files = list(
        tmp_path.glob("*.pdf")
    )

    assert len(stored_files) == 1
    assert (
        stored_files[0].read_bytes()
        == VALID_MINIMAL_PDF
    )

    create_mock.assert_awaited_once()

    create_arguments = (
        create_mock.await_args.kwargs
    )

    assert (
        create_arguments["original_filename"]
        == "employee-policy.pdf"
    )

    assert (
        create_arguments["status"]
        == "uploaded"
    )


def test_non_pdf_extension_returns_415(
    client: TestClient,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    A non-PDF extension should be rejected before database creation.
    """

    install_temporary_storage(
        monkeypatch,
        tmp_path,
    )

    create_mock = AsyncMock()

    monkeypatch.setattr(
        documents_route,
        "create_document",
        create_mock,
    )

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "employee-policy.txt",
                VALID_MINIMAL_PDF,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 415

    assert response.json() == {
        "detail": (
            "Only files with the .pdf extension "
            "are supported."
        )
    }

    assert list(
        tmp_path.iterdir()
    ) == []

    create_mock.assert_not_awaited()


def test_invalid_pdf_signature_returns_415(
    client: TestClient,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    A fake PDF should be rejected even when its extension and
    MIME type claim that it is a PDF.
    """

    install_temporary_storage(
        monkeypatch,
        tmp_path,
    )

    create_mock = AsyncMock()

    monkeypatch.setattr(
        documents_route,
        "create_document",
        create_mock,
    )

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "fake.pdf",
                b"This is not a real PDF.",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 415

    assert response.json() == {
        "detail": (
            "The uploaded file does not contain "
            "a valid PDF signature."
        )
    }

    assert list(
        tmp_path.iterdir()
    ) == []

    create_mock.assert_not_awaited()


def test_oversized_pdf_returns_413_and_leaves_no_file(
    client: TestClient,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    An oversized upload should return 413 and leave no partial file.
    """

    install_temporary_storage(
        monkeypatch,
        tmp_path,
        max_size_bytes=64,
    )

    oversized_pdf = (
        VALID_MINIMAL_PDF
        + (b"A" * 200)
    )

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "large-policy.pdf",
                oversized_pdf,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413

    assert response.json() == {
        "detail": (
            "The PDF exceeds the maximum "
            "allowed upload size."
        )
    }

    assert list(
        tmp_path.iterdir()
    ) == []


def test_database_failure_removes_stored_pdf(
    client: TestClient,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    When storage succeeds but PostgreSQL persistence fails,
    the route must delete the newly stored PDF.
    """

    install_temporary_storage(
        monkeypatch,
        tmp_path,
    )

    create_mock = AsyncMock(
        side_effect=SQLAlchemyError(
            "Forced database failure"
        )
    )

    monkeypatch.setattr(
        documents_route,
        "create_document",
        create_mock,
    )

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "employee-policy.pdf",
                VALID_MINIMAL_PDF,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 500

    assert response.json() == {
        "detail": (
            "The document metadata could "
            "not be saved."
        )
    }

    assert list(
        tmp_path.iterdir()
    ) == []

    create_mock.assert_awaited_once()


def test_list_documents_returns_paginated_results(
    client: TestClient,
    monkeypatch,
) -> None:
    """
    The list endpoint should return repository results and
    forward pagination parameters.
    """

    first_document = build_document(
        original_filename="first.pdf"
    )

    second_document = build_document(
        original_filename="second.pdf"
    )

    list_mock = AsyncMock(
        return_value=[
            first_document,
            second_document,
        ]
    )

    monkeypatch.setattr(
        documents_route,
        "list_documents",
        list_mock,
    )

    response = client.get(
        "/api/v1/documents",
        params={
            "limit": 5,
            "offset": 10,
        },
    )

    assert response.status_code == 200

    response_data = response.json()

    assert len(response_data) == 2

    assert (
        response_data[0]["original_filename"]
        == "first.pdf"
    )

    assert (
        response_data[1]["original_filename"]
        == "second.pdf"
    )

    list_mock.assert_awaited_once()

    repository_arguments = (
        list_mock.await_args.kwargs
    )

    assert repository_arguments["limit"] == 5
    assert repository_arguments["offset"] == 10


def test_get_document_by_id_returns_document(
    client: TestClient,
    monkeypatch,
) -> None:
    """
    A valid existing document UUID should return its metadata.
    """

    document = build_document()

    get_mock = AsyncMock(
        return_value=document
    )

    monkeypatch.setattr(
        documents_route,
        "get_document_by_id",
        get_mock,
    )

    response = client.get(
        f"/api/v1/documents/{document.id}"
    )

    assert response.status_code == 200

    response_data = response.json()

    assert (
        response_data["id"]
        == str(document.id)
    )

    assert (
        response_data["original_filename"]
        == document.original_filename
    )

    get_mock.assert_awaited_once()


def test_missing_document_returns_404(
    client: TestClient,
    monkeypatch,
) -> None:
    """
    A valid UUID with no matching record should return 404.
    """

    get_mock = AsyncMock(
        return_value=None
    )

    monkeypatch.setattr(
        documents_route,
        "get_document_by_id",
        get_mock,
    )

    missing_id = uuid4()

    response = client.get(
        f"/api/v1/documents/{missing_id}"
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": "Document not found."
    }


def test_malformed_document_id_returns_422(
    client: TestClient,
) -> None:
    """
    FastAPI should reject a malformed UUID before entering the route.
    """

    response = client.get(
        "/api/v1/documents/not-a-valid-uuid"
    )

    assert response.status_code == 422


def test_invalid_pagination_returns_422(
    client: TestClient,
) -> None:
    """
    Query validation should reject unsupported pagination values.
    """

    response = client.get(
        "/api/v1/documents",
        params={
            "limit": 0,
            "offset": -1,
        },
    )

    assert response.status_code == 422