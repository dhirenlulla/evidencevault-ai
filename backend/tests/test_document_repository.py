import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.repositories.document import (
    create_document,
    get_document_by_id,
)
from app.db.models.document import Document


class FakeAsyncSession:
    """
    Lightweight fake AsyncSession for repository tests.
    """

    def __init__(
        self,
        *,
        get_result=None,
        execute_result=None,
        commit_error=False,
    ):
        self.get_result = get_result
        self.commit_error = commit_error

        self.added = []
        self.commit_called = False
        self.refresh_called = False
        self.rollback_called = False
        self.get_calls = []
        self.execute_result = execute_result
        self.executed_statement = None
        self.add_all_called = False
        self.added_objects = None

    def add(
        self,
        obj,
    ):
        self.added.append(obj)

    def add_all(
        self,
        objects,
    ):
        self.add_all_called = True
        self.added_objects = objects

    async def commit(self):
        self.commit_called = True

        if self.commit_error:
            raise SQLAlchemyError(
                "forced commit failure"
            )

    async def refresh(
        self,
        obj,
    ):
        self.refresh_called = True

    async def rollback(self):
        self.rollback_called = True

    async def get(
        self,
        model,
        document_id,
    ):
        self.get_calls.append(
            (model, document_id)
        )
        return self.get_result
    
    async def execute(
    self,
    statement,
    ):
        self.executed_statement = statement
        return self.execute_result
    
def test_create_document_persists_document() -> None:
    """
    Repository should persist one document.
    """

    session = FakeAsyncSession()

    document_id = uuid4()

    document = asyncio.run(
        create_document(
            session,
            document_id=document_id,
            filename="stored.pdf",
            original_filename="paper.pdf",
            content_type="application/pdf",
            storage_path="/uploads/file.pdf",
        )
    )

    assert isinstance(
        document,
        Document,
    )

    assert document.id == document_id
    assert document.filename == "stored.pdf"
    assert (
        document.original_filename
        == "paper.pdf"
    )

    assert session.commit_called
    assert session.refresh_called

    assert len(session.added) == 1
    
def test_create_document_rolls_back_on_commit_failure() -> None:
    """
    Commit failures should rollback.
    """

    session = FakeAsyncSession(
        commit_error=True,
    )

    with pytest.raises(
        SQLAlchemyError,
    ):
        asyncio.run(
            create_document(
                session,
                document_id=uuid4(),
                filename="stored.pdf",
                original_filename="paper.pdf",
                content_type="application/pdf",
                storage_path="/uploads/file.pdf",
            )
        )

    assert session.rollback_called
    
def test_get_document_by_id_returns_document() -> None:
    """
    Existing document should be returned.
    """

    document = Document()

    session = FakeAsyncSession(
        get_result=document,
    )

    result = asyncio.run(
        get_document_by_id(
            session,
            uuid4(),
        )
    )

    assert result is document
    
def test_get_document_by_id_returns_none_when_missing() -> None:
    """
    Missing document should return None.
    """

    session = FakeAsyncSession(
        get_result=None,
    )

    result = asyncio.run(
        get_document_by_id(
            session,
            uuid4(),
        )
    )

    assert result is None
    
def test_list_documents_returns_documents() -> None:
    """
    Documents should be returned in the
    order produced by SQLAlchemy.
    """

    document1 = Document()
    document2 = Document()

    execute_result = SimpleNamespace(
        scalars=lambda: SimpleNamespace(
            all=lambda: [
                document1,
                document2,
            ]
        )
    )

    session = FakeAsyncSession(
        execute_result=execute_result,
    )

    from app.repositories.document import (
        list_documents,
    )

    result = asyncio.run(
        list_documents(
            session,
        )
    )

    assert result == [
        document1,
        document2,
    ]
    
def test_list_documents_returns_empty_list() -> None:
    """
    No documents should return an empty list.
    """

    execute_result = SimpleNamespace(
        scalars=lambda: SimpleNamespace(
            all=lambda: []
        )
    )

    session = FakeAsyncSession(
        execute_result=execute_result,
    )

    from app.repositories.document import (
        list_documents,
    )

    result = asyncio.run(
        list_documents(
            session,
        )
    )

    assert result == []
    
def test_update_processing_state_updates_document() -> None:
    """
    Repository should persist processing state.
    """

    document = Document()
    document.status = "processing"

    session = FakeAsyncSession()

    from app.repositories.document import (
        update_document_processing_state,
    )

    updated = asyncio.run(
        update_document_processing_state(
            session,
            document=document,
            status="processed",
            page_count=14,
            error_message=None,
        )
    )

    assert updated.status == "processed"
    assert updated.page_count == 14
    assert updated.error_message is None

    assert session.commit_called
    assert session.refresh_called
    
def test_update_processing_state_rolls_back_on_failure() -> None:
    """
    Commit failures should rollback.
    """

    document = Document()

    session = FakeAsyncSession(
        commit_error=True,
    )

    from app.repositories.document import (
        update_document_processing_state,
    )

    with pytest.raises(
        SQLAlchemyError,
    ):
        asyncio.run(
            update_document_processing_state(
                session,
                document=document,
                status="failed",
                page_count=None,
                error_message="boom",
            )
        )

    assert session.rollback_called
    
def test_replace_document_chunks_persists_chunks() -> None:
    """
    Existing chunks should be replaced atomically.
    """

    from uuid import uuid4

    from app.db.models.document import Document
    from app.repositories.document_chunk import (
        DocumentChunkInput,
        replace_document_chunks,
    )

    document = Document()
    document.id = uuid4()

    chunks = [
        DocumentChunkInput(
            id=uuid4(),
            document_id=document.id,
            chunk_index=0,
            page_number=1,
            page_chunk_index=0,
            text="First chunk",
            character_count=11,
            word_count=2,
            content_hash="hash1",
        ),
        DocumentChunkInput(
            id=uuid4(),
            document_id=document.id,
            chunk_index=1,
            page_number=1,
            page_chunk_index=1,
            text="Second chunk",
            character_count=12,
            word_count=2,
            content_hash="hash2",
        ),
    ]

    session = FakeAsyncSession()

    result = asyncio.run(
        replace_document_chunks(
            session,
            document=document,
            chunks=chunks,
        )
    )

    assert len(result) == 2
    assert session.add_all_called
    assert session.commit_called
    assert session.refresh_called

    assert document.status == "chunked"
    assert document.chunk_count == 2
    
def test_replace_document_chunks_rolls_back_on_failure() -> None:
    """
    Database failures should rollback.
    """

    from uuid import uuid4

    from app.db.models.document import Document
    from app.repositories.document_chunk import (
        replace_document_chunks,
    )

    document = Document()
    document.id = uuid4()

    session = FakeAsyncSession(
        commit_error=True,
    )

    with pytest.raises(
        SQLAlchemyError,
    ):
        asyncio.run(
            replace_document_chunks(
                session,
                document=document,
                chunks=[],
            )
        )

    assert session.rollback_called
    
def test_list_document_chunks_returns_rows() -> None:
    """
    Repository should return persisted chunks.
    """

    from app.db.models.document_chunk import (
        DocumentChunk,
    )
    from app.repositories.document_chunk import (
        list_document_chunks,
    )

    chunk1 = DocumentChunk()
    chunk2 = DocumentChunk()

    execute_result = SimpleNamespace(
        scalars=lambda: SimpleNamespace(
            all=lambda: [
                chunk1,
                chunk2,
            ]
        )
    )

    session = FakeAsyncSession(
        execute_result=execute_result,
    )

    result = asyncio.run(
        list_document_chunks(
            session,
            document_id=uuid4(),
        )
    )

    assert result == [
        chunk1,
        chunk2,
    ]
    
def test_count_document_chunks_returns_count() -> None:
    """
    Repository should return total chunk count.
    """

    from app.repositories.document_chunk import (
        count_document_chunks,
    )

    execute_result = SimpleNamespace(
        scalar_one=lambda: 27,
    )

    session = FakeAsyncSession(
        execute_result=execute_result,
    )

    result = asyncio.run(
        count_document_chunks(
            session,
            document_id=uuid4(),
        )
    )

    assert result == 27