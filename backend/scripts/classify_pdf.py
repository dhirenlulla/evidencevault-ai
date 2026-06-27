import argparse
from pathlib import Path

from app.core.exceptions import (
    PDFProcessingError
)

from app.services.pdf_classification import (
    PDFDocumentAnalysis,
    classify_pdf_document
)

def display_page_report(result: PDFDocumentAnalysis) -> None:
    """ 
    Display page-level classification information
    """
    
    if not result.pages:
        return
    
    print()
    print("Page Analysis")
    print("-" * 42)
    
    for page in result.pages:
        coverage_percentage = (
            page.image_coverage_ratio * 100
        )
        
        print(
            f"Page {page.page_number}: "
            f"{page.classification.value}"
        )
        
        print(
            f"  Text characters: "
            f"{page.text_character_count}"
        )
        
        print(
            f"  Words: "
            f"{page.word_count}"
        )
        
        print(
            f"  Images: "
            f"{page.image_count}"
        )
        
        print(
            f"  Image Coverage: "
            f"{coverage_percentage:.1f}%"
        )
        
def display_document_report(result: PDFDocumentAnalysis) -> None:
    """ 
    Display document-level classification information
    """
    
    print()
    print("EvidenceVault PDF classification")
    print("=" * 42)
    
    print(
        f"File: {result.source_path.name}"
    )
    
    print(
        f"Classification: "
        f"{result.classification.value}"
    )
    
    print(
        f"Reason: {result.reason}"
    )
    
    print(
        f"Page Count: {result.page_count}"
    )
    
    print(
        f"Pages with text: "
        f"{result.extractable_page_count}"
    )
    
    print(
        f"Image-only pages: "
        f"{result.image_only_page_count}"
    )
    
    print(
        f"Empty pages: "
        f"{result.empty_page_count}"
    )
    
    print(
        f"Total text characters: "
        f"{result.total_text_characters}"
    )
    
    print(
        f"Total words: "
        f"{result.total_words}"
    )
    
    print(
        f"Required PDF repair: "
        f"{result.was_repaired}"
    )
    
    print(
        f"Has usable text: "
        f"{result.has_usable_text}"
    )
    
    print(
        f"Requires OCR: "
        f"{result.requires_ocr}"
    )
    
    print(
        f"Requires password: "
        f"{result.requires_password}"
    )
    
    print(
        f"Can continue to chunking: "
        f"{result.can_continue_to_chunking}"
    )
    
    if result.error_message:
        print(
            f"Technical error: "
            f"{result.error_message}"
        )
        
    display_page_report(
        result
    )
    
def parse_argument() -> argparse.Namespace:
    """ 
    Read the PDF path from the command line.
    """
    
    parser = argparse.ArgumentParser(
        description=(
            "Classify an EvidenceVault PDF as text-based, "
            "partial, image-only, encrypted or malformed"
        )
    )
    
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the PDF to classify."
    )
    
    return parser.parse_args()

def main() -> None:
    """ 
    Command-line entry point
    """
    
    arguments = parse_argument()
    
    try:
        result = classify_pdf_document(
            arguments.pdf_path
        )
        
    except PDFProcessingError as exc:
        raise SystemExit(
            str(exc)
        ) from exc
        
        
    display_document_report(result)
    
    
if __name__ == "__main__":
    main()