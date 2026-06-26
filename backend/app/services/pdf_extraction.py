import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.core.exceptions import (
    PDFExtractionError,
    PDFPathError,
    PDFProcessingError
)

HORIZONTAL_WHITESPACE_PATTERN = re.compile(r"[ \t]+")

EXCESSIVE_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")

CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

TEXT_CHARACTER_TRANSLATION = str.maketrans(
    {
        # Common PDF ligatures
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "st",
        "\ufb06": "st",

        # Unusual space characters
        "\u00a0": " ",
        "\u2007": " ",
        "\u202f": " ",

        # Invisible formatting characters
        "\u00ad": "",
        "\u200b": "",
    }
)

@dataclass(
    frozen=True,
    slots=True
)
class ExtractedPage:
    """ 
    Cleaned text and metadata for one PDF page.
    """
    
    page_number: int
    text: str
    character_count: int
    word_count: int
    is_empty: bool
    
@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """ 
    Complete page-level extraction result for one PDF.
    """
    
    source_path: Path
    page_count: int
    pages: tuple[ExtractedPage, ...]
    total_characters: int
    total_words: int
    empty_page_numbers: tuple[int, ...]
    
    @property
    def text_page_count(self) -> int:
        """ 
        Return the number of pages containing extractable text.
        """
        
        return (
            
            self.page_count
            -len(self.empty_page_numbers)
        )
        
    @property
    def has_extractable_text(self) -> bool:
        """ 
        Return True when the document contains cleaned text.
        """
        
        return self.total_characters > 0
    

def clean_extracted_text(raw_text: str) -> str:
    """ 
    Apply conservative text cleaning to one extracted PDF page.
    
    The cleaning process:
    1. Normalizes Unicode representation.
    2. Converts common PDF ligatures.
    3. Normalizes line endings.
    4. Replaces unusual spaces.
    5. Removes invisible control characters.
    6. Collapses repeated horizontal whitespace.
    7. Reserves paragraph boundaries.
    """
    
    if not raw_text:
        return ""
    
    normalized_text = unicodedata.normalize(
        "NFC",
        raw_text
    )
    
    normalized_text = normalized_text.translate(
        TEXT_CHARACTER_TRANSLATION
    )
    
    normalized_text = normalized_text.replace(
        "\r",
        "\n"
    )
    
    normalized_text = CONTROL_CHARACTER_PATTERN.sub(
        "",
        normalized_text
    )
    
    cleaned_lines: list[str] = []
    
    for line in normalized_text.split("\n"):
        cleaned_line = (
            HORIZONTAL_WHITESPACE_PATTERN.sub(
                " ", 
                line
            )
            .strip()
        )
        
        cleaned_lines.append(cleaned_line)
        
    cleaned_text = "\n".join(
        cleaned_lines
    )
    
    cleaned_text = (
        EXCESSIVE_BLANK_LINES_PATTERN.sub(
            "\n\n", 
            cleaned_text
        )
    )
    
    return cleaned_text.strip()


def resolve_pdf_path(pdf_path: str | Path) -> Path:
    
    """ 
    Validate and return an absolute PDF path.
    """
    
    resolved_path = (
        Path(pdf_path)
        .expanduser()
        .resolve()
    )
    
    if not resolved_path.exists():
        raise PDFPathError(
            f"PDF file does not exist: "
            f"{resolved_path}"
        )
        
    if not resolved_path.is_file():
        raise PDFPathError(
            f"The supplied PDF path is not a file: "
            f"{resolved_path}"
        )
        
    if resolved_path.suffix.lower() != ".pdf":
        raise PDFPathError(
            "The supplied file must have a .pdf extension."
        )
        
    return resolved_path


def extract_pdf_pages(pdf_path: str | Path) -> ExtractedDocument:
    
    """ 
    Extract and clean text from every page of one PDF.

    Page numbering starts at 1 because those numbers will later
    be displayed as source citations to users.
    """
    
    resolved_path = resolve_pdf_path(pdf_path)
    
    try:
        with pymupdf.open(resolved_path) as document:
            if document.needs_pass:
                raise PDFExtractionError(
                    "The PDF requires a password before "
                    "its text can be extracted."
                )

            extracted_pages: list[
                ExtractedPage
            ] = []

            empty_page_numbers: list[int] = []

            total_characters = 0
            total_words = 0

            for page_index in range(
                document.page_count
            ):
                page = document.load_page(
                    page_index
                )

                raw_text = page.get_text(
                    "text",
                    sort=True,
                )

                cleaned_text = (
                    clean_extracted_text(
                        raw_text
                    )
                )

                page_number = page_index + 1

                character_count = len(
                    cleaned_text
                )

                word_count = (
                    len(cleaned_text.split())
                    if cleaned_text
                    else 0
                )

                is_empty = (
                    character_count == 0
                )

                if is_empty:
                    empty_page_numbers.append(
                        page_number
                    )

                extracted_page = ExtractedPage(
                    page_number=page_number,
                    text=cleaned_text,
                    character_count=character_count,
                    word_count=word_count,
                    is_empty=is_empty,
                )

                extracted_pages.append(
                    extracted_page
                )

                total_characters += (
                    character_count
                )

                total_words += word_count

            return ExtractedDocument(
                source_path=resolved_path,
                page_count=document.page_count,
                pages=tuple(extracted_pages),
                total_characters=total_characters,
                total_words=total_words,
                empty_page_numbers=tuple(
                    empty_page_numbers
                ),
            )

    except PDFProcessingError:
        raise

    except (RuntimeError, ValueError) as exc:
        raise PDFExtractionError(
            "PyMuPDF could not extract text "
            f"from the PDF: {exc}"
        ) from exc