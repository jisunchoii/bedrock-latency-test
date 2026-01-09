"""Property-based tests for timer module.

**Feature: bedrock-latency-benchmark, Property 1: Latency measurements are positive**
**Feature: bedrock-latency-benchmark, Property 2: TTFB is less than or equal to total time**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import time

from src.timer import LatencyMetrics, Timer


# Strategy for generating positive floats (valid latency values)
positive_float = st.floats(min_value=0.001, max_value=1e9, allow_nan=False, allow_infinity=False)

# Strategy for generating optional positive floats
optional_positive_float = st.one_of(st.none(), positive_float)


# Strategy for generating valid LatencyMetrics with positive values
latency_metrics_strategy = st.builds(
    LatencyMetrics,
    total_time_ms=positive_float,
    ttfb_ms=optional_positive_float,
    model_time_ms=optional_positive_float,
    overhead_ms=optional_positive_float,
)


@given(metrics=latency_metrics_strategy)
@settings(max_examples=100)
def test_latency_measurements_are_positive(metrics: LatencyMetrics) -> None:
    """Property 1: Latency measurements are positive.
    
    **Feature: bedrock-latency-benchmark, Property 1: Latency measurements are positive**
    **Validates: Requirements 2.1, 3.1, 4.1**
    
    For any latency measurement (API, agent, or multi-agent), the recorded time
    in milliseconds SHALL be a positive number greater than zero.
    """
    # total_time_ms must always be positive
    assert metrics.total_time_ms > 0, f"total_time_ms must be positive, got {metrics.total_time_ms}"
    
    # Optional fields, when set, must be positive
    if metrics.ttfb_ms is not None:
        assert metrics.ttfb_ms > 0, f"ttfb_ms must be positive when set, got {metrics.ttfb_ms}"
    
    if metrics.model_time_ms is not None:
        assert metrics.model_time_ms > 0, f"model_time_ms must be positive when set, got {metrics.model_time_ms}"
    
    if metrics.overhead_ms is not None:
        assert metrics.overhead_ms > 0, f"overhead_ms must be positive when set, got {metrics.overhead_ms}"


@given(sleep_duration=st.floats(min_value=0.001, max_value=0.05))
@settings(max_examples=100)
def test_timer_produces_positive_latency(sleep_duration: float) -> None:
    """Property 1 (Timer variant): Timer produces positive latency measurements.
    
    **Feature: bedrock-latency-benchmark, Property 1: Latency measurements are positive**
    **Validates: Requirements 2.1, 3.1, 4.1**
    
    For any timer operation with a positive duration, the resulting latency
    measurement SHALL be a positive number greater than zero.
    """
    timer = Timer()
    timer.start()
    time.sleep(sleep_duration)
    metrics = timer.stop()
    
    # Timer should always produce positive total_time_ms
    assert metrics.total_time_ms > 0, f"Timer total_time_ms must be positive, got {metrics.total_time_ms}"


@given(
    sleep_before_first_byte=st.floats(min_value=0.001, max_value=0.02),
    sleep_after_first_byte=st.floats(min_value=0.001, max_value=0.02)
)
@settings(max_examples=100)
def test_timer_with_ttfb_produces_positive_values(
    sleep_before_first_byte: float,
    sleep_after_first_byte: float
) -> None:
    """Property 1 (TTFB variant): Timer with TTFB produces positive measurements.
    
    **Feature: bedrock-latency-benchmark, Property 1: Latency measurements are positive**
    **Validates: Requirements 2.1, 3.1, 4.1**
    
    For any streaming timer operation, both total_time_ms and ttfb_ms
    SHALL be positive numbers greater than zero.
    """
    timer = Timer()
    timer.start()
    time.sleep(sleep_before_first_byte)
    timer.mark_first_byte()
    time.sleep(sleep_after_first_byte)
    metrics = timer.stop()
    
    # Both measurements should be positive
    assert metrics.total_time_ms > 0, f"total_time_ms must be positive, got {metrics.total_time_ms}"
    assert metrics.ttfb_ms is not None, "ttfb_ms should be set after mark_first_byte()"
    assert metrics.ttfb_ms > 0, f"ttfb_ms must be positive, got {metrics.ttfb_ms}"


# =============================================================================
# Property 2: TTFB is less than or equal to total time
# =============================================================================

# Strategy for generating valid LatencyMetrics where TTFB <= total_time
# This tests the data model constraint
@st.composite
def latency_metrics_with_ttfb_strategy(draw):
    """Generate LatencyMetrics where ttfb_ms is set and <= total_time_ms."""
    total_time = draw(st.floats(min_value=0.001, max_value=1e6, allow_nan=False, allow_infinity=False))
    ttfb = draw(st.floats(min_value=0.001, max_value=total_time, allow_nan=False, allow_infinity=False))
    model_time = draw(optional_positive_float)
    overhead = draw(optional_positive_float)
    return LatencyMetrics(
        total_time_ms=total_time,
        ttfb_ms=ttfb,
        model_time_ms=model_time,
        overhead_ms=overhead,
    )


@given(metrics=latency_metrics_with_ttfb_strategy())
@settings(max_examples=100)
def test_ttfb_less_than_or_equal_to_total_time_data_model(metrics: LatencyMetrics) -> None:
    """Property 2: TTFB is less than or equal to total time (data model).
    
    **Feature: bedrock-latency-benchmark, Property 2: TTFB is less than or equal to total time**
    **Validates: Requirements 2.2**
    
    For any streaming API response, the Time To First Byte (TTFB) SHALL be
    less than or equal to the total response time.
    """
    assert metrics.ttfb_ms is not None, "ttfb_ms should be set for this test"
    assert metrics.ttfb_ms <= metrics.total_time_ms, (
        f"TTFB ({metrics.ttfb_ms}ms) must be <= total_time ({metrics.total_time_ms}ms)"
    )


@given(
    sleep_before_first_byte=st.floats(min_value=0.001, max_value=0.02),
    sleep_after_first_byte=st.floats(min_value=0.0, max_value=0.02)
)
@settings(max_examples=100)
def test_ttfb_less_than_or_equal_to_total_time_timer(
    sleep_before_first_byte: float,
    sleep_after_first_byte: float
) -> None:
    """Property 2: TTFB is less than or equal to total time (Timer).
    
    **Feature: bedrock-latency-benchmark, Property 2: TTFB is less than or equal to total time**
    **Validates: Requirements 2.2**
    
    For any streaming timer operation where mark_first_byte() is called,
    the resulting TTFB SHALL be less than or equal to the total response time.
    """
    timer = Timer()
    timer.start()
    time.sleep(sleep_before_first_byte)
    timer.mark_first_byte()
    time.sleep(sleep_after_first_byte)
    metrics = timer.stop()
    
    assert metrics.ttfb_ms is not None, "ttfb_ms should be set after mark_first_byte()"
    assert metrics.ttfb_ms <= metrics.total_time_ms, (
        f"TTFB ({metrics.ttfb_ms}ms) must be <= total_time ({metrics.total_time_ms}ms)"
    )
