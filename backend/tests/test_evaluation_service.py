from math import log2

import pytest

from app.services.evaluation import (
    RetrievalEvaluator,
    RetrievalMetricsAggregator,
    compare_metrics,
)


# ----------------------------------------------------------------
# RetrievalEvaluator.evaluate — Recall@K
# ----------------------------------------------------------------


def test_recall_counts_relevant_items_found_in_top_k() -> None:
    """
    Recall@K = (relevant items found) / (total relevant items
    that exist), regardless of their position in the ranking.
    """

    evaluator = RetrievalEvaluator()

    result = evaluator.evaluate(
        retrieved_ids=["x", "a", "y", "b", "z"],
        relevant_ids={"a", "b", "c"},
        k=5,
    )

    # Found a and b (2 of the 3 relevant items).
    assert result.recall_at_k == pytest.approx(2 / 3)


def test_recall_is_zero_when_relevant_ids_is_empty() -> None:
    """
    With no known-relevant items, recall is defined as 0 rather
    than dividing by zero.
    """

    evaluator = RetrievalEvaluator()

    result = evaluator.evaluate(
        retrieved_ids=["x", "y"],
        relevant_ids=set(),
        k=2,
    )

    assert result.recall_at_k == 0.0


def test_recall_only_considers_top_k_items() -> None:
    """
    A relevant item ranked below position k should not count
    toward recall.
    """

    evaluator = RetrievalEvaluator()

    result = evaluator.evaluate(
        retrieved_ids=["x", "y", "a"],
        relevant_ids={"a"},
        k=2,
    )

    assert result.recall_at_k == 0.0


# ----------------------------------------------------------------
# RetrievalEvaluator.evaluate — Precision@K
# ----------------------------------------------------------------


def test_precision_is_relevant_fraction_of_retrieved_items() -> None:
    """
    Precision@K = (relevant items found) / (items actually
    retrieved).
    """

    evaluator = RetrievalEvaluator()

    result = evaluator.evaluate(
        retrieved_ids=["a", "x", "b", "y"],
        relevant_ids={"a", "b"},
        k=4,
    )

    assert result.precision_at_k == pytest.approx(2 / 4)


def test_precision_uses_actual_retrieved_count_when_fewer_than_k() -> None:
    """
    If fewer than k items were actually retrieved, precision is
    divided by the number actually retrieved, not by k, so a
    short candidate list isn't unfairly penalized.
    """

    evaluator = RetrievalEvaluator()

    result = evaluator.evaluate(
        retrieved_ids=["a"],
        relevant_ids={"a"},
        k=10,
    )

    assert result.precision_at_k == pytest.approx(1.0)


def test_precision_is_zero_when_nothing_retrieved() -> None:
    """
    With an empty retrieved list, precision is 0 rather than
    dividing by zero.
    """

    evaluator = RetrievalEvaluator()

    result = evaluator.evaluate(
        retrieved_ids=[],
        relevant_ids={"a"},
        k=5,
    )

    assert result.precision_at_k == 0.0


# ----------------------------------------------------------------
# RetrievalEvaluator.evaluate — Hit Rate@K
# ----------------------------------------------------------------


def test_hit_rate_is_one_when_any_relevant_item_found() -> None:
    evaluator = RetrievalEvaluator()

    result = evaluator.evaluate(
        retrieved_ids=["x", "a"],
        relevant_ids={"a", "b"},
        k=2,
    )

    assert result.hit_rate_at_k == 1.0


def test_hit_rate_is_zero_when_no_relevant_item_found() -> None:
    evaluator = RetrievalEvaluator()

    result = evaluator.evaluate(
        retrieved_ids=["x", "y"],
        relevant_ids={"a", "b"},
        k=2,
    )

    assert result.hit_rate_at_k == 0.0


# ----------------------------------------------------------------
# RetrievalEvaluator.evaluate — MRR
# ----------------------------------------------------------------


def test_mrr_is_reciprocal_of_first_relevant_position() -> None:
    """
    MRR = 1 / rank of the first relevant item, using 1-indexed
    positions.
    """

    evaluator = RetrievalEvaluator()

    result = evaluator.evaluate(
        retrieved_ids=["x", "y", "a", "b"],
        relevant_ids={"a", "b"},
        k=4,
    )

    # First relevant item ("a") is at position 3.
    assert result.mrr == pytest.approx(1 / 3)


def test_mrr_uses_first_relevant_item_not_the_best_one() -> None:
    """
    MRR only cares about the first hit — a second relevant item
    ranked higher doesn't matter here since it never comes first.
    """

    evaluator = RetrievalEvaluator()

    result = evaluator.evaluate(
        retrieved_ids=["a", "x", "b"],
        relevant_ids={"a", "b"},
        k=3,
    )

    assert result.mrr == pytest.approx(1 / 1)


def test_mrr_is_zero_when_no_relevant_item_found() -> None:
    evaluator = RetrievalEvaluator()

    result = evaluator.evaluate(
        retrieved_ids=["x", "y"],
        relevant_ids={"a"},
        k=2,
    )

    assert result.mrr == 0.0


# ----------------------------------------------------------------
# RetrievalEvaluator.evaluate — NDCG@K
# ----------------------------------------------------------------


def test_ndcg_is_one_for_perfect_ranking() -> None:
    """
    If every relevant item is ranked first, the ranking is
    already ideal, so NDCG should be exactly 1.0.
    """

    evaluator = RetrievalEvaluator()

    result = evaluator.evaluate(
        retrieved_ids=["a", "b", "x", "y"],
        relevant_ids={"a", "b"},
        k=4,
    )

    assert result.ndcg_at_k == pytest.approx(1.0)


def test_ndcg_rewards_relevant_items_ranked_higher() -> None:
    """
    Two rankings with the same relevant items but different
    positions should score differently — the one with the
    relevant item closer to the top scores higher.
    """

    evaluator = RetrievalEvaluator()

    relevant_near_top = evaluator.evaluate(
        retrieved_ids=["a", "x", "y"],
        relevant_ids={"a"},
        k=3,
    )

    relevant_near_bottom = evaluator.evaluate(
        retrieved_ids=["x", "y", "a"],
        relevant_ids={"a"},
        k=3,
    )

    assert (
        relevant_near_top.ndcg_at_k
        > relevant_near_bottom.ndcg_at_k
    )


def test_ndcg_matches_manual_formula() -> None:
    """
    Cross-check against the DCG/IDCG formula computed by hand,
    to catch any off-by-one errors in position indexing.
    """

    evaluator = RetrievalEvaluator()

    result = evaluator.evaluate(
        retrieved_ids=["x", "a", "y"],
        relevant_ids={"a", "b"},
        k=3,
    )

    # "a" is relevant, at 1-indexed position 2.
    dcg = 1.0 / log2(2 + 1)

    # Ideal: min(|relevant|=2, k=3) = 2 relevant items ranked
    # first, at positions 1 and 2.
    idcg = (
        1.0 / log2(1 + 1)
    ) + (
        1.0 / log2(2 + 1)
    )

    assert result.ndcg_at_k == pytest.approx(dcg / idcg)


def test_ndcg_is_zero_when_no_relevant_item_found() -> None:
    evaluator = RetrievalEvaluator()

    result = evaluator.evaluate(
        retrieved_ids=["x", "y"],
        relevant_ids={"a"},
        k=2,
    )

    assert result.ndcg_at_k == 0.0


# ----------------------------------------------------------------
# RetrievalEvaluator.evaluate — validation
# ----------------------------------------------------------------


def test_zero_k_raises_value_error() -> None:
    evaluator = RetrievalEvaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(
            retrieved_ids=["a"],
            relevant_ids={"a"},
            k=0,
        )


def test_negative_k_raises_value_error() -> None:
    evaluator = RetrievalEvaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(
            retrieved_ids=["a"],
            relevant_ids={"a"},
            k=-1,
        )


# ----------------------------------------------------------------
# RetrievalMetricsAggregator
# ----------------------------------------------------------------


def test_aggregate_averages_metrics_across_queries() -> None:
    evaluator = RetrievalEvaluator()
    aggregator = RetrievalMetricsAggregator()

    query_1 = evaluator.evaluate(
        retrieved_ids=["a"],
        relevant_ids={"a"},
        k=1,
    )

    query_2 = evaluator.evaluate(
        retrieved_ids=["x"],
        relevant_ids={"a"},
        k=1,
    )

    result = aggregator.aggregate(
        results=[query_1, query_2]
    )

    # query_1 recall = 1.0, query_2 recall = 0.0 -> average 0.5
    assert result.recall_at_k == pytest.approx(0.5)
    assert result.num_queries == 2
    assert result.k == 1


def test_aggregate_empty_results_raises_value_error() -> None:
    aggregator = RetrievalMetricsAggregator()

    with pytest.raises(ValueError):
        aggregator.aggregate(results=[])


def test_aggregate_mismatched_k_raises_value_error() -> None:
    evaluator = RetrievalEvaluator()
    aggregator = RetrievalMetricsAggregator()

    query_at_k1 = evaluator.evaluate(
        retrieved_ids=["a"],
        relevant_ids={"a"},
        k=1,
    )

    query_at_k2 = evaluator.evaluate(
        retrieved_ids=["a", "b"],
        relevant_ids={"a"},
        k=2,
    )

    with pytest.raises(ValueError):
        aggregator.aggregate(
            results=[query_at_k1, query_at_k2]
        )


# ----------------------------------------------------------------
# compare_metrics
# ----------------------------------------------------------------


def test_compare_metrics_reports_positive_delta_for_improvement() -> None:
    evaluator = RetrievalEvaluator()
    aggregator = RetrievalMetricsAggregator()

    baseline = aggregator.aggregate(
        results=[
            evaluator.evaluate(
                retrieved_ids=["x"],
                relevant_ids={"a"},
                k=1,
            )
        ]
    )

    candidate = aggregator.aggregate(
        results=[
            evaluator.evaluate(
                retrieved_ids=["a"],
                relevant_ids={"a"},
                k=1,
            )
        ]
    )

    comparison = compare_metrics(
        baseline=baseline,
        candidate=candidate,
    )

    assert comparison.recall_at_k_delta == pytest.approx(1.0)
    assert comparison.mrr_delta == pytest.approx(1.0)


def test_compare_metrics_reports_negative_delta_for_regression() -> None:
    evaluator = RetrievalEvaluator()
    aggregator = RetrievalMetricsAggregator()

    baseline = aggregator.aggregate(
        results=[
            evaluator.evaluate(
                retrieved_ids=["a"],
                relevant_ids={"a"},
                k=1,
            )
        ]
    )

    candidate = aggregator.aggregate(
        results=[
            evaluator.evaluate(
                retrieved_ids=["x"],
                relevant_ids={"a"},
                k=1,
            )
        ]
    )

    comparison = compare_metrics(
        baseline=baseline,
        candidate=candidate,
    )

    assert comparison.recall_at_k_delta == pytest.approx(-1.0)