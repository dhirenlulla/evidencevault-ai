import asyncio
import sys
from pathlib import Path

from sqlalchemy import select

from app.core.config import get_settings
from app.db.models.document import Document
from app.db.session import (
    AsyncSessionFactory,
    close_database_engine,
)


if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )


settings = get_settings()


def resolve_storage_path(
    storage_path: str,
) -> Path:
    """
    Convert a stored relative or absolute path into an absolute path.
    """

    path = Path(
        storage_path
    )

    if path.is_absolute():
        return path.resolve()

    backend_directory = (
        settings.upload_path.parent
    )

    return (
        backend_directory / path
    ).resolve()


async def verify_storage_consistency() -> bool:
    """
    Compare PostgreSQL document records with locally stored PDFs.

    This script reports inconsistencies but does not delete anything.
    """

    try:
        async with AsyncSessionFactory() as session:
            result = await session.execute(
                select(Document)
            )

            documents = list(
                result.scalars().all()
            )

        referenced_paths: set[Path] = set()

        missing_files: list[
            tuple[str, str, Path]
        ] = []

        for document in documents:
            if not document.storage_path:
                continue

            resolved_path = resolve_storage_path(
                document.storage_path
            )

            referenced_paths.add(
                resolved_path
            )

            if not resolved_path.exists():
                missing_files.append(
                    (
                        str(document.id),
                        document.original_filename,
                        resolved_path,
                    )
                )

        stored_pdf_paths = {
            path.resolve()
            for path in settings.upload_path.glob(
                "*.pdf"
            )
            if path.is_file()
        }

        orphan_files = sorted(
            stored_pdf_paths - referenced_paths
        )

        print("\nEvidenceVault storage verification")
        print("=" * 38)

        print(
            f"PostgreSQL document records: "
            f"{len(documents)}"
        )

        print(
            f"PDF files in upload directory: "
            f"{len(stored_pdf_paths)}"
        )

        print(
            f"Missing PDF files: "
            f"{len(missing_files)}"
        )

        print(
            f"Orphan PDF files: "
            f"{len(orphan_files)}"
        )

        if missing_files:
            print(
                "\nDatabase records with missing files:"
            )

            for (
                document_id,
                original_filename,
                resolved_path,
            ) in missing_files:
                print(
                    f"- ID: {document_id}"
                )

                print(
                    f"  Original name: "
                    f"{original_filename}"
                )

                print(
                    f"  Expected path: "
                    f"{resolved_path}"
                )

        if orphan_files:
            print(
                "\nFiles without database records:"
            )

            for orphan_path in orphan_files:
                print(
                    f"- {orphan_path}"
                )

        is_consistent = (
            not missing_files
            and not orphan_files
        )

        print()

        if is_consistent:
            print(
                "Storage consistency: OK"
            )

        else:
            print(
                "Storage consistency: PROBLEMS FOUND"
            )

        return is_consistent

    finally:
        await close_database_engine()


def main() -> None:
    """
    Run the asynchronous verification and return a shell exit code.
    """

    is_consistent = asyncio.run(
        verify_storage_consistency()
    )

    raise SystemExit(
        0 if is_consistent else 1
    )


if __name__ == "__main__":
    main()