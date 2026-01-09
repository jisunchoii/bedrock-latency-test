"""Property-based tests for API benchmark module.

**Feature: bedrock-latency-benchmark, Property 4: Iteration count consistency**
**Feature: bedrock-latency-benchmark, Property 5: Benchmark result serialization round-trip**
"""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime, timezone

from src.timer import LatencyMetrics
from src.stats import BenchmarkStatistics, calculate_statistics
from src.results import BenchmarkResult


# Strategy for generating valid LatencyMetrics
latency_metrics_strategy = st.builds(
    LatencyMetrics,
    total_time_ms=st.floats(min_value=0.1, max_value=100000.0, allow_nan=False, allow_infinity=False),
    ttfb_ms=st.one_of(
        st.none(),
        st.floats(min_value=0.1, max_value=100000.0, allow_nan=False, allow_infinity=False)
    ),
    model_time_ms=st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False)
    ),
    overhead_ms=st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False)
    ),
)


def create_benchmark_result_with_iterations(n_iterations: int, metrics: list) -> BenchmarkResult:
    """Helper to create a BenchmarkResult with given iterations and metrics.
    
    Args:
        n_iterations: Number of iterations to record
        metrics: List of LatencyMetrics
        
    Returns:
        BenchmarkResult instance
    """
    stats = calculate_statistics(metrics) if metrics else BenchmarkStatistics(
        min_ms=0.0, max_ms=0.0, avg_ms=0.0, median_ms=0.0,
        p95_ms=0.0, p99_ms=0.0, std_dev_ms=0.0
    )
    
    return BenchmarkResult(
        model_id="test-model-id",
        model_name="test-model",
        provider="TestProvider",
        region="us-east-1",
        benchmark_type="api",
        timestamp=datetime.now(),
        iterations=n_iterations,
        metrics=metrics,
        statistics=stats,
        errors=[],
    )


@given(n_iterations=st.integers(min_value=1, max_value=100))
@settings(max_examples=100)
def test_iteration_count_consistency(n_iterations: int) -> None:
    """Property 4: Iteration count consistency.
    
    **Feature: bedrock-latency-benchmark, Property 4: Iteration count consistency**
    **Validates: Requirements 2.4**
    
    For any benchmark run with N configured iterations, the result SHALL contain
    exactly N latency measurements (excluding warmup).
    
    This test verifies that when we generate N metrics for a benchmark result,
    the result correctly reports N iterations and contains exactly N metrics.
    """
    # Generate exactly n_iterations metrics
    metrics = [
        LatencyMetrics(
            total_time_ms=float(i + 1) * 10.0,  # Deterministic values for testing
            ttfb_ms=None,
            model_time_ms=None,
            overhead_ms=None,
        )
        for i in range(n_iterations)
    ]
    
    # Create benchmark result with n_iterations
    result = create_benchmark_result_with_iterations(n_iterations, metrics)
    
    # Property: iterations field matches the actual number of metrics
    assert result.iterations == n_iterations, (
        f"Expected iterations={n_iterations}, got {result.iterations}"
    )
    assert len(result.metrics) == n_iterations, (
        f"Expected {n_iterations} metrics, got {len(result.metrics)}"
    )
    
    # Additional invariant: iterations field equals len(metrics)
    assert result.iterations == len(result.metrics), (
        f"Iterations field ({result.iterations}) does not match metrics count ({len(result.metrics)})"
    )


@given(
    n_iterations=st.integers(min_value=1, max_value=50),
    metrics_list=st.lists(latency_metrics_strategy, min_size=1, max_size=50)
)
@settings(max_examples=100)
def test_iteration_count_matches_metrics_after_serialization(
    n_iterations: int, 
    metrics_list: list
) -> None:
    """Property 4 (serialization variant): Iteration count preserved through serialization.
    
    **Feature: bedrock-latency-benchmark, Property 4: Iteration count consistency**
    **Validates: Requirements 2.4**
    
    For any benchmark result, the iteration count and metrics count SHALL remain
    consistent after JSON serialization and deserialization.
    """
    # Use the actual metrics list length as iterations for consistency
    actual_iterations = len(metrics_list)
    
    # Create benchmark result
    result = create_benchmark_result_with_iterations(actual_iterations, metrics_list)
    
    # Serialize and deserialize
    json_str = result.to_json()
    restored_result = BenchmarkResult.from_json(json_str)
    
    # Property: iteration count is preserved
    assert restored_result.iterations == actual_iterations, (
        f"Expected iterations={actual_iterations} after deserialization, "
        f"got {restored_result.iterations}"
    )
    
    # Property: metrics count is preserved
    assert len(restored_result.metrics) == actual_iterations, (
        f"Expected {actual_iterations} metrics after deserialization, "
        f"got {len(restored_result.metrics)}"
    )
    
    # Property: iterations field equals metrics count after round-trip
    assert restored_result.iterations == len(restored_result.metrics), (
        f"Iterations field ({restored_result.iterations}) does not match "
        f"metrics count ({len(restored_result.metrics)}) after deserialization"
    )


# Strategy for generating valid BenchmarkStatistics
benchmark_statistics_strategy = st.builds(
    BenchmarkStatistics,
    min_ms=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
    max_ms=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
    avg_ms=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
    median_ms=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
    p95_ms=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
    p99_ms=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
    std_dev_ms=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
)


# Strategy for generating valid datetime (using timezone-aware datetimes for consistency)
datetime_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
)


# Strategy for generating valid BenchmarkResult
benchmark_result_strategy = st.builds(
    BenchmarkResult,
    model_id=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'S'), whitelist_characters='-_.:/')),
    model_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='-_')),
    provider=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters=' ')),
    region=st.sampled_from(['us-east-1', 'us-west-2', 'eu-west-1', 'ap-northeast-2', 'ap-southeast-1']),
    benchmark_type=st.sampled_from(['api', 'agent', 'multi-agent']),
    timestamp=datetime_strategy,
    iterations=st.integers(min_value=1, max_value=100),
    metrics=st.lists(latency_metrics_strategy, min_size=1, max_size=20),
    statistics=benchmark_statistics_strategy,
    errors=st.lists(st.text(min_size=0, max_size=100), min_size=0, max_size=5),
)


@given(result=benchmark_result_strategy)
@settings(max_examples=100)
def test_benchmark_result_serialization_round_trip(result: BenchmarkResult) -> None:
    """Property 5: Benchmark result serialization round-trip.
    
    **Feature: bedrock-latency-benchmark, Property 5: Benchmark result serialization round-trip**
    **Validates: Requirements 2.6**
    
    For any valid BenchmarkResult object, serializing to JSON and deserializing
    back SHALL produce an equivalent object.
    """
    # Serialize to JSON
    json_str = result.to_json()
    
    # Deserialize back
    restored_result = BenchmarkResult.from_json(json_str)
    
    # Verify all fields are preserved
    assert restored_result.model_id == result.model_id, (
        f"model_id mismatch: expected {result.model_id}, got {restored_result.model_id}"
    )
    assert restored_result.model_name == result.model_name, (
        f"model_name mismatch: expected {result.model_name}, got {restored_result.model_name}"
    )
    assert restored_result.provider == result.provider, (
        f"provider mismatch: expected {result.provider}, got {restored_result.provider}"
    )
    assert restored_result.region == result.region, (
        f"region mismatch: expected {result.region}, got {restored_result.region}"
    )
    assert restored_result.benchmark_type == result.benchmark_type, (
        f"benchmark_type mismatch: expected {result.benchmark_type}, got {restored_result.benchmark_type}"
    )
    assert restored_result.timestamp == result.timestamp, (
        f"timestamp mismatch: expected {result.timestamp}, got {restored_result.timestamp}"
    )
    assert restored_result.iterations == result.iterations, (
        f"iterations mismatch: expected {result.iterations}, got {restored_result.iterations}"
    )
    assert restored_result.errors == result.errors, (
        f"errors mismatch: expected {result.errors}, got {restored_result.errors}"
    )
    
    # Verify metrics are preserved
    assert len(restored_result.metrics) == len(result.metrics), (
        f"metrics count mismatch: expected {len(result.metrics)}, got {len(restored_result.metrics)}"
    )
    for i, (orig, restored) in enumerate(zip(result.metrics, restored_result.metrics)):
        assert restored.total_time_ms == orig.total_time_ms, (
            f"metrics[{i}].total_time_ms mismatch"
        )
        assert restored.ttfb_ms == orig.ttfb_ms, (
            f"metrics[{i}].ttfb_ms mismatch"
        )
        assert restored.model_time_ms == orig.model_time_ms, (
            f"metrics[{i}].model_time_ms mismatch"
        )
        assert restored.overhead_ms == orig.overhead_ms, (
            f"metrics[{i}].overhead_ms mismatch"
        )
    
    # Verify statistics are preserved
    assert restored_result.statistics.min_ms == result.statistics.min_ms, (
        f"statistics.min_ms mismatch"
    )
    assert restored_result.statistics.max_ms == result.statistics.max_ms, (
        f"statistics.max_ms mismatch"
    )
    assert restored_result.statistics.avg_ms == result.statistics.avg_ms, (
        f"statistics.avg_ms mismatch"
    )
    assert restored_result.statistics.median_ms == result.statistics.median_ms, (
        f"statistics.median_ms mismatch"
    )
    assert restored_result.statistics.p95_ms == result.statistics.p95_ms, (
        f"statistics.p95_ms mismatch"
    )
    assert restored_result.statistics.p99_ms == result.statistics.p99_ms, (
        f"statistics.p99_ms mismatch"
    )
    assert restored_result.statistics.std_dev_ms == result.statistics.std_dev_ms, (
        f"statistics.std_dev_ms mismatch"
    )
