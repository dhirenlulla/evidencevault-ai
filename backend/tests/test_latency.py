import time

import pytest

from app.services.latency import (
    LatencyTimer,
    compute_latency_stats,
)


# ----------------------------------------------------------------
# LatencyTimer
# ----------------------------------------------------------------


def test_timer_measures_positive_elapsed_time() -> None:
    timer = LatencyTimer()

    with timer:
        time.sleep(0.01)

    assert timer.elapsed_seconds > 0.0


def test_timer_elapsed_time_is_zero_before_use() -> None:
    timer = LatencyTimer()

    assert timer.elapsed_seconds == 0.0


def test_timer_records_roughly_the_sleep_duration() -> None:
    timer = LatencyTimer()

    with timer:
        time.sleep(0.05)

    # Allow generous slack for CI/scheduler jitter; we only care
    # that it's in the right ballpark, not exact.
    assert 0.03 < timer.elapsed_seconds < 0.5


def test_timer_still_records_elapsed_time_if_block_raises() -> None:
    """
    Elapsed time should still be recorded even if the timed code
    raises an exception, so a failing operation is not invisible
    to latency tracking.
    """

    timer = LatencyTimer()

    with pytest.raises(RuntimeError):
        with timer:
            raise RuntimeError("boom")

    assert timer.elapsed_seconds > 0.0


# ----------------------------------------------------------------
# compute_latency_stats
# ----------------------------------------------------------------


def test_mean_is_average_of_samples() -> None:
    stats = compute_latency_stats(
        samples_seconds=[1.0, 2.0, 3.0]
    )

    assert stats.mean_seconds == pytest.approx(2.0)


def test_min_and_max_are_correct() -> None:
    stats = compute_latency_stats(
        samples_seconds=[5.0, 1.0, 3.0]
    )

    assert stats.min_seconds == pytest.approx(1.0)
    assert stats.max_seconds == pytest.approx(5.0)


def test_num_samples_matches_input_length() -> None:
    stats = compute_latency_stats(
        samples_seconds=[1.0, 2.0, 3.0, 4.0]
    )

    assert stats.num_samples == 4


def test_p50_of_single_sample_is_that_sample() -> None:
    stats = compute_latency_stats(
        samples_seconds=[2.5]
    )

    assert stats.p50_seconds == pytest.approx(2.5)
    assert stats.p95_seconds == pytest.approx(2.5)


def test_p95_is_greater_than_or_equal_to_p50() -> None:
    stats = compute_latency_stats(
        samples_seconds=[1.0, 2.0, 3.0, 4.0, 100.0]
    )

    assert stats.p95_seconds >= stats.p50_seconds


def test_empty_samples_raises_value_error() -> None:
    with pytest.raises(ValueError):
        compute_latency_stats(samples_seconds=[])