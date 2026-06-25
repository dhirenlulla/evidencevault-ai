import argparse
from pathlib import Path

import pymupdf


PREVIEW_CHARACTER_LIMIT = 500


def create_text_preview(
    text: str,
    limit: int = PREVIEW_CHARACTER_LIMIT,
) -> str:
    """
    Convert extracted page text into a short single-line preview.

    Repeated spaces, line breaks and tabs are collapsed so the
    terminal output remains readable.
    """

    normalized_text = " ".join(
        text.split()
    )

    if not normalized_text:
        return "[No extractable text]"

    if len(normalized_text) <= limit:
        return normalized_text

    return (
        normalized_text[:limit].rstrip()
        + "..."
    )


def inspect_pdf(
    pdf_path: Path,
) -> None:
    """
    Open a PDF and display basic extraction information.

    This script is diagnostic only. It does not update PostgreSQL,
    create chunks or modify the source PDF.
    """

    resolved_path = pdf_path.expanduser().resolve()

    if not resolved_path.exists():
        raise SystemExit(
            f"PDF file does not exist: {resolved_path}"
        )

    if not resolved_path.is_file():
        raise SystemExit(
            f"The supplied path is not a file: {resolved_path}"
        )

    if resolved_path.suffix.lower() != ".pdf":
        raise SystemExit(
            "The supplied file must have a .pdf extension."
        )

    try:
        with pymupdf.open(
            resolved_path
        ) as document:
            page_count = document.page_count

            print()
            print("EvidenceVault PDF inspection")
            print("=" * 34)
            print(f"File: {resolved_path.name}")
            print(f"Path: {resolved_path}")
            print(f"Pages: {page_count}")

            if page_count == 0:
                print(
                    "Result: The PDF contains no pages."
                )
                return

            total_characters = 0
            empty_page_numbers: list[int] = []
            first_page_preview = ""

            print()
            print("Page extraction results")
            print("-" * 34)

            for page_index, page in enumerate(
                document
            ):
                page_number = page_index + 1

                extracted_text = page.get_text(
                    "text",
                    sort=True,
                )

                cleaned_for_count = (
                    extracted_text.strip()
                )

                character_count = len(
                    cleaned_for_count
                )

                total_characters += character_count

                if character_count == 0:
                    empty_page_numbers.append(
                        page_number
                    )

                if page_index == 0:
                    first_page_preview = (
                        create_text_preview(
                            extracted_text
                        )
                    )

                print(
                    f"Page {page_number}: "
                    f"{character_count} characters"
                )

            print()
            print("Document summary")
            print("-" * 34)

            print(
                f"Total pages: {page_count}"
            )

            print(
                f"Total extracted characters: "
                f"{total_characters}"
            )

            print(
                f"Pages with extractable text: "
                f"{page_count - len(empty_page_numbers)}"
            )

            print(
                f"Pages without extractable text: "
                f"{len(empty_page_numbers)}"
            )

            if empty_page_numbers:
                empty_page_list = ", ".join(
                    str(page_number)
                    for page_number
                    in empty_page_numbers
                )

                print(
                    f"Empty page numbers: "
                    f"{empty_page_list}"
                )

            print()
            print("First-page preview")
            print("-" * 34)
            print(first_page_preview)

    except RuntimeError as exc:
        raise SystemExit(
            f"PyMuPDF could not open or parse "
            f"the PDF: {exc}"
        ) from exc


def parse_arguments() -> argparse.Namespace:
    """
    Read the PDF path supplied through the command line.
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
        help=(
            "Path to the PDF that should be inspected."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """

    arguments = parse_arguments()

    inspect_pdf(
        arguments.pdf_path
    )


if __name__ == "__main__":
    main()