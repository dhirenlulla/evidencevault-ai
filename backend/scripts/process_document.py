import argparse
import asyncio
import sys
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

from app.core.exceptions import DocumentProcessingWorkflowError

from app.db.session import (
    AsyncSessionFactory,
    close_database_engine
    
)

from app.services.document_processing import (
    DocumentProcessingResult,
    process_document
)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )
    
def display_result(result: DocumentProcessingResult) -> None:
    """ 
    Display one document-processing result.
    """
    
    print()
    print("EvidenceVault document processing")
    print("=" * 42)
    
    print(f"Document ID: {result.document_id}")
    
    print(f"Status: {result.status.value}")
    
    print(
        "Classification: "
        + (
            result.classification.value
            if result.classification
            else "unavailable"
        )
    )
    
    print(f"Page Count: {result.page_count}")
    
    print(
        f"Pages with text: "
        f"{result.extractable_page_count}"
    )
    
    print(
        f"Image-only pages: "
        f"{result.image_only_page_count}"  
    )
    
    print(
        f"Empty Pages: "
        f"{result.empty_page_count}"
    )
    
    print(
        f"Total Characters: "
        f"{result.total_characters}"
    )
    
    print(
        f"Total words: "
        f"{result.total_words}"
    )
    
    print(
        f"Can continue to chunking: "
        f"{result.can_continue_to_chunking}"
    )

    print(
        f"Message: {result.message}"
    )
    
async def run_processing(document_id: UUID) -> DocumentProcessingResult:
    """ 
    Process one document using a real database session.
    """
    
    try:
        async with AsyncSessionFactory() as session:
            return await process_document(
                session=session,
                document_id=document_id
            )
    finally:
        await close_database_engine()
        
def parse_arguments() -> argparse.Namespace:
    """ 
    Read a document UUID from the command line.
    """
    
    parser =  argparse.ArgumentParser(
        description=(
            "Classify and extract one EvidenceVault "
            "document using its PostgreSQL UUID."
        )
    )
    
    parser.add_argument(
        "document_id",
        type=UUID,
        help="UUID of the document to process."
    )
    
    return parser.parse_args()

def main() -> None:
    """ 
    Command-line entry point.
    """
    
    arguments = parse_arguments()
    
    try:
        result = asyncio.run(
            run_processing(
                arguments.document_id
            )
        )
        
    except DocumentProcessingWorkflowError as exc:
        raise SystemExit(
            f"Processing could not start: {exc}"
        ) from exc
        
    except SQLAlchemyError as exc:
        raise SystemExit(
            f"PostgreSQL operation failed: {exc}"
        ) from exc
        
    display_result(result)
    
if __name__ == "__main__":
    main()