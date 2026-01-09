"""Property-based tests for statistics module.

**Feature: bedrock-latency-benchmark, Property 3: Statistics bounds invariant**
"""

import pytest
from hypothesis import given, strategies as st, settings

from src.stats import BenchmarkStatistics, calculate_statistics
from src.timer import LatencyMetrics


# Strategy for generating positive floats (valid latency values)
positive_float = st.floats(min_value=0.001, max_value=1e9, allow_nan=False, allow_infinity=False)


# Strategy for generating a list of LatencyMetrics with positive total_time_ms
latency_metrics_list_strategy = st.lists(
    st.builds(
        LatencyMetrics,
        total_time_ms=positive_float,
    ),
    min_size=1,
    max_size=100,
)


@given(metrics=latency_metrics_list_strategy)
@settings(max_examples=100)
def test_statistics_bounds_invariant(metrics: list) -> None:
    """Property 3: Statistics bounds invariant.
    
    **Feature: bedrock-latency-benchmark, Property 3: Statistics bounds invariant**
    **Validates: Requirements 2.3**
    
    For any list of latency measurements, the calculated statistics SHALL satisfy:
    min <= median <= max AND min <= avg <= max AND min <= p95 <= max AND min <= p99 <= max.
    """
    stats = calculate_statistics(metrics)
    
    # min <= median <= max
    assert stats.min_ms <= stats.median_ms, (
        f"min ({stats.min_ms}) must be <= median ({stats.median_ms})"
    )
    assert stats.median_ms <= stats.max_ms, (
        f"median ({stats.median_ms}) must be <= max ({stats.max_ms})"
    )
    
    # min <= avg <= max
    assert stats.min_ms <= stats.avg_ms, (
        f"min ({stats.min_ms}) must be <= avg ({stats.avg_ms})"
    )
    assert stats.avg_ms <= stats.max_ms, (
        f"avg ({stats.avg_ms}) must be <= max ({stats.max_ms})"
    )
    
    # min <= p95 <= max
    assert stats.min_ms <= stats.p95_ms, (
        f"min ({stats.min_ms}) must be <= p95 ({stats.p95_ms})"
    )
    assert stats.p95_ms <= stats.max_ms, (
        f"p95 ({stats.p95_ms}) must be <= max ({stats.max_ms})"
    )
    
    # min <= p99 <= max
    assert stats.min_ms <= stats.p99_ms, (
        f"min ({stats.min_ms}) must be <= p99 ({stats.p99_ms})"
    )
    assert stats.p99_ms <= stats.max_ms, (
        f"p99 ({stats.p99_ms}) must be <= max ({stats.max_ms})"
    )
    
    # Additional invariant: std_dev must be non-negative
    assert stats.std_dev_ms >= 0, (
        f"std_dev ({stats.std_dev_ms}) must be non-negative"
    )
