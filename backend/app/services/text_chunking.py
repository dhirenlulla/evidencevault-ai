import re
from dataclasses import dataclass
from hashlib import sha256
from uuid import UUID, uuid5

from app.core.exceptions import (
    InvalidChunkingConfigurationError,
)
from app.services.pdf_extraction import (
    ExtractedDocument,
)


PARAGRAPH_SPLIT_PATTERN = re.compile(
    r"\n{2,}"
)

SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[.!?])\s+"
)


@dataclass(
    frozen=True,
    slots=True,
)
class ChunkingOptions:
    """
    Configuration controlling document chunk creation.
    """

    max_characters: int = 1200
    overlap_characters: int = 200
    minimum_page_characters: int = 40

    def __post_init__(self) -> None:
        """
        Validate chunking configuration immediately.
        """

        if self.max_characters < 100:
            raise InvalidChunkingConfigurationError(
                "max_characters must be at least 100."
            )

        if self.overlap_characters < 0:
            raise InvalidChunkingConfigurationError(
                "overlap_characters cannot be negative."
            )

        if (
            self.overlap_characters
            >= self.max_characters
        ):
            raise InvalidChunkingConfigurationError(
                "overlap_characters must be smaller "
                "than max_characters."
            )

        if self.minimum_page_characters < 1:
            raise InvalidChunkingConfigurationError(
                "minimum_page_characters must be "
                "greater than zero."
            )

        if (
            self.minimum_page_characters
            > self.max_characters
        ):
            raise InvalidChunkingConfigurationError(
                "minimum_page_characters cannot be "
                "greater than max_characters."
            )


@dataclass(
    frozen=True,
    slots=True,
)
class TextUnit:
    """
    Small text unit used internally while constructing chunks.
    """

    text: str
    separator_before: str


@dataclass(
    frozen=True,
    slots=True,
)
class TextChunk:
    """
    One citation-ready chunk generated from one PDF page.
    """

    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    page_number: int
    page_chunk_index: int
    text: str
    character_count: int
    word_count: int
    content_hash: str


@dataclass(
    frozen=True,
    slots=True,
)
class ChunkingResult:
    """
    Complete chunking result for one extracted document.
    """

    document_id: UUID
    source_page_count: int
    source_character_count: int
    chunks: tuple[TextChunk, ...]
    chunked_page_numbers: tuple[int, ...]
    skipped_empty_page_numbers: tuple[int, ...]
    skipped_short_page_numbers: tuple[int, ...]
    total_chunk_characters: int
    total_chunk_words: int
    options: ChunkingOptions

    @property
    def chunk_count(self) -> int:
        """
        Return the total number of generated chunks.
        """

        return len(self.chunks)

    @property
    def chunked_page_count(self) -> int:
        """
        Return the number of pages that produced chunks.
        """

        return len(
            self.chunked_page_numbers
        )

    @property
    def average_chunk_characters(self) -> float:
        """
        Return the average character count per chunk.
        """

        if not self.chunks:
            return 0.0

        return (
            self.total_chunk_characters
            / self.chunk_count
        )


def split_text_by_words(
    text: str,
    maximum_characters: int,
) -> list[str]:
    """
    Split oversized text into word-aligned segments.

    Extremely long single tokens, such as malformed URLs,
    are hard-split only when they exceed the maximum size.
    """

    words = text.split()

    if not words:
        return []

    segments: list[str] = []
    current_segment = ""

    for word in words:
        if len(word) > maximum_characters:
            word_pieces = [
                word[index:index + maximum_characters]
                for index in range(
                    0,
                    len(word),
                    maximum_characters,
                )
            ]

        else:
            word_pieces = [word]

        for word_piece in word_pieces:
            candidate = (
                word_piece
                if not current_segment
                else (
                    f"{current_segment} "
                    f"{word_piece}"
                )
            )

            if len(candidate) <= maximum_characters:
                current_segment = candidate

            else:
                if current_segment:
                    segments.append(
                        current_segment
                    )

                current_segment = word_piece

    if current_segment:
        segments.append(
            current_segment
        )

    return segments


def split_page_into_units(
    text: str,
    *,
    maximum_characters: int,
    overlap_characters: int,
) -> list[TextUnit]:
    """
    Split page text into paragraph-, sentence-, or word-level units.

    The returned units are later packed into final chunks.
    """

    paragraphs = [
        paragraph.strip()
        for paragraph in (
            PARAGRAPH_SPLIT_PATTERN.split(
                text
            )
        )
        if paragraph.strip()
    ]

    text_units: list[TextUnit] = []

    maximum_new_content = (
        maximum_characters
        - overlap_characters
        if overlap_characters
        else maximum_characters
    )

    for paragraph in paragraphs:
        paragraph_separator = (
            ""
            if not text_units
            else "\n\n"
        )

        if len(paragraph) <= maximum_characters:
            text_units.append(
                TextUnit(
                    text=paragraph,
                    separator_before=(
                        paragraph_separator
                    ),
                )
            )

            continue

        sentences = [
            sentence.strip()
            for sentence in (
                SENTENCE_SPLIT_PATTERN.split(
                    paragraph
                )
            )
            if sentence.strip()
        ]

        first_sentence = True

        for sentence in sentences:
            sentence_separator = (
                paragraph_separator
                if first_sentence
                else " "
            )

            if (
                len(sentence)
                <= maximum_new_content
            ):
                sentence_pieces = [
                    sentence
                ]

            else:
                sentence_pieces = (
                    split_text_by_words(
                        sentence,
                        maximum_new_content,
                    )
                )

            for piece_index, piece in enumerate(
                sentence_pieces
            ):
                text_units.append(
                    TextUnit(
                        text=piece,
                        separator_before=(
                            sentence_separator
                            if piece_index == 0
                            else " "
                        ),
                    )
                )

            first_sentence = False

    return text_units


def combine_text(
    existing_text: str,
    new_text: str,
    separator: str,
) -> str:
    """
    Combine two text sections using the supplied separator.
    """

    if not existing_text:
        return new_text.strip()

    effective_separator = (
        separator
        if separator
        else " "
    )

    return (
        f"{existing_text}"
        f"{effective_separator}"
        f"{new_text}"
    ).strip()


def take_overlap_tail(
    text: str,
    maximum_characters: int,
) -> str:
    """
    Return a word-aligned tail from the previous chunk.
    """

    if maximum_characters <= 0:
        return ""

    if len(text) <= maximum_characters:
        return text.strip()

    start_index = (
        len(text)
        - maximum_characters
    )

    while (
        start_index < len(text)
        and not text[start_index].isspace()
    ):
        start_index += 1

    return text[start_index:].strip()


def chunk_page_text(
    text: str,
    options: ChunkingOptions,
) -> list[str]:
    """
    Convert one page's text into overlapping chunks.
    """

    text_units = split_page_into_units(
        text,
        maximum_characters=(
            options.max_characters
        ),
        overlap_characters=(
            options.overlap_characters
        ),
    )

    if not text_units:
        return []

    chunks: list[str] = []
    current_chunk = ""

    for text_unit in text_units:
        candidate = combine_text(
            existing_text=current_chunk,
            new_text=text_unit.text,
            separator=text_unit.separator_before,
        )

        if (
            len(candidate)
            <= options.max_characters
        ):
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(
                current_chunk.strip()
            )

        available_overlap_characters = max(
            0,
            (
                options.max_characters
                - len(text_unit.text)
                - 1
            ),
        )

        overlap_text = take_overlap_tail(
            current_chunk,
            min(
                options.overlap_characters,
                available_overlap_characters,
            ),
        )

        current_chunk = combine_text(
            existing_text=overlap_text,
            new_text=text_unit.text,
            separator=" ",
        )

    if current_chunk:
        chunks.append(
            current_chunk.strip()
        )

    deduplicated_chunks: list[str] = []

    for chunk_text in chunks:
        if (
            not deduplicated_chunks
            or (
                chunk_text
                != deduplicated_chunks[-1]
            )
        ):
            deduplicated_chunks.append(
                chunk_text
            )

    return deduplicated_chunks


def calculate_content_hash(
    text: str,
) -> str:
    """
    Return a SHA-256 hash for one chunk's text.
    """

    return sha256(
        text.encode("utf-8")
    ).hexdigest()


def create_deterministic_chunk_id(
    *,
    document_id: UUID,
    page_number: int,
    page_chunk_index: int,
    content_hash: str,
) -> UUID:
    """
    Create a stable UUID for one chunk.
    """

    chunk_name = (
        f"page:{page_number}:"
        f"chunk:{page_chunk_index}:"
        f"sha256:{content_hash}"
    )

    return uuid5(
        document_id,
        chunk_name,
    )


def create_text_chunk(
    *,
    document_id: UUID,
    chunk_index: int,
    page_number: int,
    page_chunk_index: int,
    text: str,
) -> TextChunk:
    """
    Build one fully populated TextChunk object.
    """

    content_hash = calculate_content_hash(
        text
    )

    chunk_id = (
        create_deterministic_chunk_id(
            document_id=document_id,
            page_number=page_number,
            page_chunk_index=(
                page_chunk_index
            ),
            content_hash=content_hash,
        )
    )

    return TextChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        page_number=page_number,
        page_chunk_index=page_chunk_index,
        text=text,
        character_count=len(text),
        word_count=len(text.split()),
        content_hash=content_hash,
    )


def chunk_extracted_document(
    document_id: UUID,
    extracted_document: ExtractedDocument,
    *,
    options: ChunkingOptions | None = None,
) -> ChunkingResult:
    """
    Generate page-aware chunks for one extracted PDF.

    Empty pages and pages below the configured minimum-content
    threshold are skipped and reported in the result.
    """

    resolved_options = (
        options
        if options is not None
        else ChunkingOptions()
    )

    chunks: list[TextChunk] = []

    chunked_page_numbers: list[int] = []
    skipped_empty_page_numbers: list[int] = []
    skipped_short_page_numbers: list[int] = []

    for page in extracted_document.pages:
        page_text = page.text.strip()

        if page.is_empty or not page_text:
            skipped_empty_page_numbers.append(
                page.page_number
            )

            continue

        if (
            len(page_text)
            < resolved_options
            .minimum_page_characters
        ):
            skipped_short_page_numbers.append(
                page.page_number
            )

            continue

        page_chunk_texts = chunk_page_text(
            page_text,
            resolved_options,
        )

        if not page_chunk_texts:
            skipped_empty_page_numbers.append(
                page.page_number
            )

            continue

        chunked_page_numbers.append(
            page.page_number
        )

        for (
            page_chunk_index,
            chunk_text,
        ) in enumerate(page_chunk_texts):
            chunk = create_text_chunk(
                document_id=document_id,
                chunk_index=len(chunks),
                page_number=page.page_number,
                page_chunk_index=(
                    page_chunk_index
                ),
                text=chunk_text,
            )

            chunks.append(
                chunk
            )

    return ChunkingResult(
        document_id=document_id,
        source_page_count=(
            extracted_document.page_count
        ),
        source_character_count=(
            extracted_document.total_characters
        ),
        chunks=tuple(chunks),
        chunked_page_numbers=tuple(
            chunked_page_numbers
        ),
        skipped_empty_page_numbers=tuple(
            skipped_empty_page_numbers
        ),
        skipped_short_page_numbers=tuple(
            skipped_short_page_numbers
        ),
        total_chunk_characters=sum(
            chunk.character_count
            for chunk in chunks
        ),
        total_chunk_words=sum(
            chunk.word_count
            for chunk in chunks
        ),
        options=resolved_options,
    )