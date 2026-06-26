import argparse
from pathlib import Path

from app.core.exceptions import (
    PDFProcessingError,
)
from app.services.pdf_extraction import (
    ExtractedDocument,
    extract_pdf_pages,
)


PREVIEW_CHARACTER_LIMIT = 500


def create_text_preview(
    text: str,
    limit: int = PREVIEW_CHARACTER_LIMIT,
) -> str:
    """
    Create a short single-line preview for terminal output.
    """

    preview_text = " ".join(
        text.split()
    )

    if not preview_text:
        return "[No extractable text]"

    if len(preview_text) <= limit:
        return preview_text

    return (
        preview_text[:limit].rstrip()
        + "..."
    )


def display_inspection_report(
    result: ExtractedDocument,
) -> None:
    """
    Display a readable terminal report.
    """

    print()
    print("EvidenceVault PDF inspection")
    print("=" * 34)

    print(
        f"File: {result.source_path.name}"
    )

    print(
        f"Path: {result.source_path}"
    )

    print(
        f"Pages: {result.page_count}"
    )

    print()
    print("Page extraction results")
    print("-" * 34)

    for page in result.pages:
        print(
            f"Page {page.page_number}: "
            f"{page.character_count} characters, "
            f"{page.word_count} words"
        )

    print()
    print("Document summary")
    print("-" * 34)

    print(
        f"Total pages: "
        f"{result.page_count}"
    )

    print(
        f"Pages with text: "
        f"{result.text_page_count}"
    )

    print(
        f"Pages without text: "
        f"{len(result.empty_page_numbers)}"
    )

    print(
        f"Total extracted characters: "
        f"{result.total_characters}"
    )

    print(
        f"Total extracted words: "
        f"{result.total_words}"
    )

    print(
        f"Has extractable text: "
        f"{result.has_extractable_text}"
    )

    if result.empty_page_numbers:
        page_numbers = ", ".join(
            str(page_number)
            for page_number
            in result.empty_page_numbers
        )

        print(
            f"Empty page numbers: "
            f"{page_numbers}"
        )

    first_page_text = (
        result.pages[0].text
        if result.pages
        else ""
    )

    print()
    print("First-page preview")
    print("-" * 34)

    print(
        create_text_preview(
            first_page_text
        )
    )


def parse_arguments() -> argparse.Namespace:
    """
    Read the PDF path from the command line.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Inspect page-level text extraction "
            "for an EvidenceVault PDF."
        )
    )

    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the PDF to inspect.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """

    arguments = parse_arguments()

    try:
        result = extract_pdf_pages(
            arguments.pdf_path
        )

    except PDFProcessingError as exc:
        raise SystemExit(
            str(exc)
        ) from exc

    display_inspection_report(
        result
    )


if __name__ == "__main__":
    main()