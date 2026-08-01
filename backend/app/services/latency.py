import time
from collections.abc import Sequence
from dataclasses import dataclass, field

@dataclass(slots=True)
class LatencyTimer:
    """ 
    Measure how long a block of code takes to run, in seconds.
    
    Usage:
    
        timer = LatencyTimer()
        with timer:
            do_something_slow()
        print(timer.elapsed_seconds)
        
    This has no knowledge of retrieval, chunks, or metrics - it only 
    measures elapsed wall-clock time. Keeping it generic means it can
    be reused to time any operation later (dense retrieval, reranking,
    the full RAG pipeline), not just evaluation runs.
    """
    
    elapsed_seconds : float = field(
        default=0.0,
        init=False
    )
    
    _start_time: float = field(
        default=0.0,
        init=False,
        repr=False,
    )
    
    def __enter__(self) -> "LatencyTimer":
        self._start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.elapsed_seconds = (
            time.perf_counter() - self._start_time
        )
        

@dataclass(frozen=True, slots=True)
class LatencyStats:
    """ 
    Summary statistics over a set of latency samples, in seconds.
    """
    
    mean_seconds: float
    p50_seconds: float
    p95_seconds: float
    min_seconds: float
    max_seconds: float
    num_samples: int
    
    
def compute_latency_stats(
    *,
    samples_seconds: Sequence[float],
) -> LatencyStats:
    """ 
    Compute summary statistics over a set of latency samples.

    p50 and p95 use nearest-rank percentiles on the sorted
    samples, which avoids interpolating a latency value that was
    never actually observed.
    """
    
    if not samples_seconds:
        raise ValueError(
            "Cannot compute latency stats over an empty list of samples."
        )
        
    sorted_samples = sorted(samples_seconds)
    
    count = len(sorted_samples)
    
    return LatencyStats(
        mean_seconds=(
            sum(sorted_samples) / count
        ),
        p50_seconds=_percentile(
            sorted_samples=sorted_samples,
            percentile=0.50,
        ),
        p95_seconds=_percentile(
            sorted_samples=sorted_samples,
            percentile=0.95,
        ),
        min_seconds=sorted_samples[0],
        max_seconds=sorted_samples[-1],
        num_samples=count
    )
    
def _percentile(
    *,
    sorted_samples: Sequence[float],
    percentile: float,
) -> float:
    """ 
    Nearest-rank percentile over an already-sorted sequence.
    """
    
    count = len(sorted_samples)
    
    rank = max(
        0,
        min(
            count - 1,
            int(round(percentile * (count - 1))),
        )
    )
    
    return sorted_samples[rank]