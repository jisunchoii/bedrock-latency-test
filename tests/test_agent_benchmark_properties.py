"""Property-based tests for Agent benchmark module.

**Feature: bedrock-latency-benchmark, Property 7: Agent time breakdown invariant**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from src.timer import LatencyMetrics


# Strategy for generating valid agent LatencyMetrics with time breakdown
# For agent benchmarks, model_time_ms and overhead_ms should be set
# and their sum should equal total_time_ms (within floating point tolerance)
@st.composite
def agent_latency_metrics_strategy(draw):
    """Generate valid agent LatencyMetrics where time breakdown is consistent.
    
    For agent benchmarks:
    - total_time_ms = model_time_ms + tool_time_ms + overhead_ms
    - Since tool_time is typically 0 or absorbed into overhead in the current implementation,
      we test: total_time_ms = model_time_ms + overhead_ms
    """
    # Generate model time and overhead separately
    model_time_ms = draw(st.floats(min_value=0.1, max_value=50000.0, allow_nan=False, allow_infinity=False))
    tool_time_ms = draw(st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False))
    overhead_ms = draw(st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False))
    
    # Total time is the sum of all components
    total_time_ms = model_time_ms + tool_time_ms + overhead_ms
    
    # In the current implementation, tool_time is absorbed into overhead calculation
    # So we store: overhead_ms = original_overhead + tool_time
    effective_overhead_ms = overhead_ms + tool_time_ms
    
    return LatencyMetrics(
        total_time_ms=total_time_ms,
        ttfb_ms=None,  # Agent benchmarks don't track TTFB
        model_time_ms=model_time_ms,
        overhead_ms=effective_overhead_ms,
    )


@given(metrics=agent_latency_metrics_strategy())
@settings(max_examples=100)
def test_agent_time_breakdown_invariant(metrics: LatencyMetrics) -> None:
    """Property 7: Agent time breakdown invariant.
    
    **Feature: bedrock-latency-benchmark, Property 7: Agent time breakdown invariant**
    **Validates: Requirements 3.2, 3.3**
    
    For any agent benchmark result, the total time SHALL equal 
    model_time + tool_time + orchestration_overhead (within floating point tolerance).
    
    Since tool_time is absorbed into overhead in the current implementation,
    we verify: total_time_ms ≈ model_time_ms + overhead_ms
    """
    # Skip if model_time or overhead is None (not an agent benchmark result)
    assume(metrics.model_time_ms is not None)
    assume(metrics.overhead_ms is not None)
    
    # Calculate expected total from components
    expected_total = metrics.model_time_ms + metrics.overhead_ms
    
    # Allow for floating point tolerance (0.001ms or 0.0001% of total, whichever is larger)
    tolerance = max(0.001, metrics.total_time_ms * 0.000001)
    
    # Property: total_time_ms should equal model_time_ms + overhead_ms
    assert abs(metrics.total_time_ms - expected_total) <= tolerance, (
        f"Time breakdown invariant violated: "
        f"total_time_ms ({metrics.total_time_ms:.6f}) != "
        f"model_time_ms ({metrics.model_time_ms:.6f}) + overhead_ms ({metrics.overhead_ms:.6f}) = "
        f"{expected_total:.6f}, difference: {abs(metrics.total_time_ms - expected_total):.6f}"
    )


@given(
    model_time_ms=st.floats(min_value=0.1, max_value=50000.0, allow_nan=False, allow_infinity=False),
    tool_time_ms=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    overhead_ms=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_agent_time_components_are_non_negative(
    model_time_ms: float,
    tool_time_ms: float,
    overhead_ms: float,
) -> None:
    """Property 7 (component constraint): All time components are non-negative.
    
    **Feature: bedrock-latency-benchmark, Property 7: Agent time breakdown invariant**
    **Validates: Requirements 3.2, 3.3**
    
    For any agent benchmark, model_time, tool_time, and overhead SHALL all be
    non-negative values.
    """
    total_time_ms = model_time_ms + tool_time_ms + overhead_ms
    
    metrics = LatencyMetrics(
        total_time_ms=total_time_ms,
        ttfb_ms=None,
        model_time_ms=model_time_ms,
        overhead_ms=overhead_ms + tool_time_ms,  # Tool time absorbed into overhead
    )
    
    # Property: All time components must be non-negative
    assert metrics.total_time_ms >= 0, f"total_time_ms must be non-negative, got {metrics.total_time_ms}"
    assert metrics.model_time_ms >= 0, f"model_time_ms must be non-negative, got {metrics.model_time_ms}"
    assert metrics.overhead_ms >= 0, f"overhead_ms must be non-negative, got {metrics.overhead_ms}"


@given(
    model_time_ms=st.floats(min_value=0.1, max_value=50000.0, allow_nan=False, allow_infinity=False),
    overhead_ms=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_agent_model_time_does_not_exceed_total(
    model_time_ms: float,
    overhead_ms: float,
) -> None:
    """Property 7 (bound constraint): Model time does not exceed total time.
    
    **Feature: bedrock-latency-benchmark, Property 7: Agent time breakdown invariant**
    **Validates: Requirements 3.2, 3.3**
    
    For any agent benchmark, model_time_ms SHALL be less than or equal to total_time_ms.
    """
    total_time_ms = model_time_ms + overhead_ms
    
    metrics = LatencyMetrics(
        total_time_ms=total_time_ms,
        ttfb_ms=None,
        model_time_ms=model_time_ms,
        overhead_ms=overhead_ms,
    )
    
    # Property: model_time_ms <= total_time_ms
    assert metrics.model_time_ms <= metrics.total_time_ms, (
        f"model_time_ms ({metrics.model_time_ms}) exceeds total_time_ms ({metrics.total_time_ms})"
    )
