import pytest

from app.services.rrf import RRFScore, RRFService


def test_no_rankings_returns_empty_tuple() -> None:
    """
    Fusing zero rankings should return an empty tuple.
    """

    service = RRFService()

    result = service.fuse(rankings=[])

    assert result == ()


def test_empty_rankings_return_empty_tuple() -> None:
    """
    Fusing rankings that are themselves empty should return
    an empty tuple.
    """

    service = RRFService()

    result = service.fuse(rankings=[[], []])

    assert result == ()


def test_single_ranking_preserves_order() -> None:
    """
    Fusing a single ranking should preserve its original order,
    since there is nothing else to fuse against.
    """

    service = RRFService()

    result = service.fuse(rankings=[["a", "b", "c"]])

    assert [score.item for score in result] == ["a", "b", "c"]


def test_first_place_scores_higher_than_second_place() -> None:
    """
    An item ranked first should receive a higher RRF score than
    an item ranked second in the same ranking.
    """

    service = RRFService()

    result = service.fuse(rankings=[["a", "b"]])

    scores_by_item = {
        score.item: score.score
        for score in result
    }

    assert scores_by_item["a"] > scores_by_item["b"]


def test_item_in_both_rankings_outranks_single_ranking_item() -> None:
    """
    An item that appears near the top of both rankings should
    be fused ahead of an item that only appears in one ranking,
    even if that item was ranked first there.
    """

    service = RRFService()

    result = service.fuse(
        rankings=[
            ["shared", "dense_only"],
            ["shared", "lexical_only"],
        ]
    )

    assert result[0].item == "shared"


def test_scores_are_sorted_descending() -> None:
    """
    Fused results should be ordered from highest score to lowest.
    """

    service = RRFService()

    result = service.fuse(
        rankings=[
            ["a", "b", "c"],
            ["b", "c", "a"],
        ]
    )

    scores = [score.score for score in result]

    assert scores == sorted(scores, reverse=True)


def test_item_only_in_one_ranking_is_still_included() -> None:
    """
    An item that appears in only one of several rankings should
    still appear in the fused result.
    """

    service = RRFService()

    result = service.fuse(
        rankings=[
            ["a"],
            ["b"],
        ]
    )

    items = {score.item for score in result}

    assert items == {"a", "b"}


def test_contribution_formula_matches_reciprocal_rank_fusion() -> None:
    """
    The fused score for an item ranked at position `r` in one
    ranking should equal 1 / (k + r), matching the RRF formula
    exactly.
    """

    k = 60

    service = RRFService(k=k)

    result = service.fuse(rankings=[["a", "b", "c"]])

    scores_by_item = {
        score.item: score.score
        for score in result
    }

    assert scores_by_item["a"] == pytest.approx(1 / (k + 1))
    assert scores_by_item["b"] == pytest.approx(1 / (k + 2))
    assert scores_by_item["c"] == pytest.approx(1 / (k + 3))


def test_scores_sum_across_multiple_rankings() -> None:
    """
    An item's fused score should be the sum of its per-ranking
    contributions, not just the best or the average.
    """

    k = 60

    service = RRFService(k=k)

    result = service.fuse(
        rankings=[
            ["a", "b"],
            ["a", "b"],
        ]
    )

    scores_by_item = {
        score.item: score.score
        for score in result
    }

    expected_a = (1 / (k + 1)) + (1 / (k + 1))

    assert scores_by_item["a"] == pytest.approx(expected_a)


def test_smaller_k_increases_rank_sensitivity() -> None:
    """
    A smaller k should widen the score gap between the first and
    second ranked items, since k dampens rank differences.
    """

    small_k_service = RRFService(k=1)
    large_k_service = RRFService(k=1000)

    small_k_result = small_k_service.fuse(rankings=[["a", "b"]])
    large_k_result = large_k_service.fuse(rankings=[["a", "b"]])

    small_k_scores = {s.item: s.score for s in small_k_result}
    large_k_scores = {s.item: s.score for s in large_k_result}

    small_k_gap = small_k_scores["a"] - small_k_scores["b"]
    large_k_gap = large_k_scores["a"] - large_k_scores["b"]

    assert small_k_gap > large_k_gap


def test_zero_k_raises_value_error() -> None:
    """
    A non-positive k is not a valid smoothing constant and should
    be rejected at construction time.
    """

    with pytest.raises(ValueError):
        RRFService(k=0)


def test_negative_k_raises_value_error() -> None:
    """
    Negative k values should also be rejected.
    """

    with pytest.raises(ValueError):
        RRFService(k=-5)


def test_default_k_matches_documented_default() -> None:
    """
    The default k should match RRFService.DEFAULT_K so the
    documented default and the actual behavior never drift apart.
    """

    service = RRFService()

    result = service.fuse(rankings=[["a"]])

    expected_score = 1 / (RRFService.DEFAULT_K + 1)

    assert result[0].score == pytest.approx(expected_score)


def test_fuse_returns_rrf_score_instances() -> None:
    """
    Every fused entry should be an RRFScore instance exposing
    both the item and its score.
    """

    service = RRFService()

    result = service.fuse(rankings=[["a", "b"]])

    for entry in result:
        assert isinstance(entry, RRFScore)
        assert isinstance(entry.score, float)