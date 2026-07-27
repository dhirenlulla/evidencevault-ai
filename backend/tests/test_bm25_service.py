from app.services.bm25 import (
    BM25Result,
    BM25Service,
)


def test_empty_corpus_returns_empty_tuple() -> None:
    """
    Ranking an empty corpus should return
    an empty tuple.
    """

    service = BM25Service()

    result = service.rank(
        query="apple",
        documents=[],
    )

    assert result == ()


def test_returns_top_matching_document() -> None:
    """
    The most relevant lexical match should
    appear first.
    """

    service = BM25Service()

    documents = [
        "apple banana",
        "orange mango",
        "apple fruit",
    ]

    result = service.rank(
        query="apple",
        documents=documents,
        top_k=1,
    )

    assert len(result) == 1

    assert result[0].index in (0, 2)

    assert result[0].score >= 0.0


def test_top_k_is_respected() -> None:
    """
    The requested top_k should limit the
    number of returned results.
    """

    service = BM25Service()

    documents = [
        f"document {i}"
        for i in range(20)
    ]

    result = service.rank(
        query="document",
        documents=documents,
        top_k=3,
    )

    assert len(result) == 3


def test_scores_are_sorted_descending() -> None:
    """
    Results should be ordered from highest
    score to lowest score.
    """

    service = BM25Service()

    documents = [
        "apple apple apple",
        "apple banana",
        "banana",
    ]

    result = service.rank(
        query="apple",
        documents=documents,
    )

    scores = [
        item.score
        for item in result
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_document_index_is_preserved() -> None:
    """
    Returned indices should correspond to
    the original document positions.
    """

    service = BM25Service()

    documents = [
        "dog",
        "cat",
        "elephant",
        "unique keyword",
    ]

    result = service.rank(
        query="unique",
        documents=documents,
        top_k=1,
    )

    assert result[0].index == 3


def test_matching_is_case_insensitive() -> None:
    """
    Tokenization should ignore case.
    """

    service = BM25Service()

    result = service.rank(
        query="APPLE",
        documents=[
            "apple banana",
        ],
    )

    assert len(result) == 1

    assert result[0].index == 0


def test_non_matching_query_returns_results_without_failure() -> None:
    """
    Queries with no lexical overlap should
    still return ranked results.
    """

    service = BM25Service()

    documents = [
        "dog cat",
        "banana orange",
    ]

    result = service.rank(
        query="elephant",
        documents=documents,
    )

    assert len(result) == 2

    for item in result:
        assert isinstance(
            item,
            BM25Result,
        )

        assert isinstance(
            item.score,
            float,
        )


def test_multiple_relevant_documents_rank_first() -> None:
    """
    Documents containing the query should
    rank ahead of unrelated documents.
    """

    service = BM25Service()

    documents = [
        "apple banana",
        "apple orange",
        "apple apple banana",
        "grape mango",
    ]

    result = service.rank(
        query="apple",
        documents=documents,
        top_k=3,
    )

    assert len(result) == 3

    for item in result:
        assert "apple" in documents[
            item.index
        ].lower()