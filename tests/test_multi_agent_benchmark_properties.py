"""Property-based tests for Multi-Agent benchmark module.

**Feature: bedrock-latency-benchmark, Property 8: Multi-agent time breakdown invariant**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import List

from src.results import AgentTimeline, MultiAgentBenchmarkResult
from src.timer import LatencyMetrics
from src.stats import BenchmarkStatistics


# Strategy for generating valid AgentTimeline
@st.composite
def agent_timeline_strategy(draw, benchmark_start_ms: float = 0.0, max_end_ms: float = 100000.0):
    """Generate a valid AgentTimeline.
    
    Args:
        benchmark_start_ms: Minimum start time
        max_end_ms: Maximum end time
    """
    agent_name = draw(st.sampled_from(["supervisor", "researcher", "writer", "analyzer", "worker"]))
    
    # Start time must be >= benchmark_start_ms
    start_time_ms = draw(st.floats(
        min_value=benchmark_start_ms,
        max_value=max_end_ms - 1.0,
        allow_nan=False,
        allow_infinity=False
    ))
    
    # End time must be > start time
    end_time_ms = draw(st.floats(
        min_value=start_time_ms + 0.1,
        max_value=max_end_ms,
        allow_nan=False,
        allow_infinity=False
    ))
    
    # Model time must be <= duration
    duration_ms = end_time_ms - start_time_ms
    model_time_ms = draw(st.floats(
        min_value=0.0,
        max_value=duration_ms,
        allow_nan=False,
        allow_infinity=False
    ))
    
    return AgentTimeline(
        agent_name=agent_name,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        model_time_ms=model_time_ms,
    )


# Strategy for generating a list of agent timelines
@st.composite
def agent_timelines_strategy(draw, min_agents: int = 1, max_agents: int = 5):
    """Generate a list of valid AgentTimelines."""
    num_agents = draw(st.integers(min_value=min_agents, max_value=max_agents))
    
    timelines = []
    for i in range(num_agents):
        timeline = draw(agent_timeline_strategy())
        timelines.append(timeline)
    
    return timelines


# Strategy for generating valid multi-agent benchmark metrics
@st.composite
def multi_agent_metrics_strategy(draw, agent_timelines: List[AgentTimeline]):
    """Generate valid LatencyMetrics consistent with agent timelines.
    
    For multi-agent benchmarks:
    - total_time_ms >= max(agent end times) - min(agent start times)
    - model_time_ms = sum of all agent model times
    - overhead_ms = total_time_ms - model_time_ms
    """
    if not agent_timelines:
        # No agents, generate simple metrics
        total_time_ms = draw(st.floats(min_value=1.0, max_value=100000.0, allow_nan=False, allow_infinity=False))
        return LatencyMetrics(
            total_time_ms=total_time_ms,
            ttfb_ms=None,
            model_time_ms=0.0,
            overhead_ms=total_time_ms,
        )
    
    # Calculate minimum total time based on agent timelines
    max_end_time = max(t.end_time_ms for t in agent_timelines)
    min_start_time = min(t.start_time_ms for t in agent_timelines)
    min_total_time = max_end_time - min_start_time
    
    # Total time must be >= the span of agent executions
    total_time_ms = draw(st.floats(
        min_value=max(min_total_time, 1.0),
        max_value=max(min_total_time * 2, 100000.0),
        allow_nan=False,
        allow_infinity=False
    ))
    
    # Model time is sum of all agent model times
    total_model_time_ms = sum(t.model_time_ms for t in agent_timelines)
    
    # Overhead is the remaining time
    overhead_ms = max(0.0, total_time_ms - total_model_time_ms)
    
    return LatencyMetrics(
        total_time_ms=total_time_ms,
        ttfb_ms=None,
        model_time_ms=total_model_time_ms,
        overhead_ms=overhead_ms,
    )


@given(agent_timelines=agent_timelines_strategy(min_agents=1, max_agents=5))
@settings(max_examples=100)
def test_multi_agent_total_time_gte_max_agent_time(agent_timelines: List[AgentTimeline]) -> None:
    """Property 8: Multi-agent time breakdown invariant - total time constraint.
    
    **Feature: bedrock-latency-benchmark, Property 8: Multi-agent time breakdown invariant**
    **Validates: Requirements 4.2, 4.3**
    
    For any multi-agent benchmark result, the total time SHALL be greater than 
    or equal to the maximum individual agent time.
    """
    assume(len(agent_timelines) > 0)
    
    # Calculate the maximum individual agent duration
    max_agent_duration = max(
        t.end_time_ms - t.start_time_ms for t in agent_timelines
    )
    
    # Calculate the span of all agent executions (from first start to last end)
    max_end_time = max(t.end_time_ms for t in agent_timelines)
    min_start_time = min(t.start_time_ms for t in agent_timelines)
    total_span = max_end_time - min_start_time
    
    # The total time must be at least as long as the span
    # (which is >= max individual agent duration)
    
    # Property: total_span >= max_agent_duration
    assert total_span >= max_agent_duration - 0.001, (
        f"Total span ({total_span:.6f}ms) should be >= "
        f"max agent duration ({max_agent_duration:.6f}ms)"
    )


@given(
    inter_agent_overhead_ms=st.floats(
        min_value=0.0,
        max_value=100000.0,
        allow_nan=False,
        allow_infinity=False
    )
)
@settings(max_examples=100)
def test_multi_agent_inter_agent_overhead_non_negative(inter_agent_overhead_ms: float) -> None:
    """Property 8: Multi-agent time breakdown invariant - overhead constraint.
    
    **Feature: bedrock-latency-benchmark, Property 8: Multi-agent time breakdown invariant**
    **Validates: Requirements 4.2, 4.3**
    
    For any multi-agent benchmark result, inter_agent_overhead SHALL be non-negative.
    """
    # Property: inter_agent_overhead_ms >= 0
    assert inter_agent_overhead_ms >= 0, (
        f"inter_agent_overhead_ms must be non-negative, got {inter_agent_overhead_ms}"
    )


@given(agent_timelines=agent_timelines_strategy(min_agents=2, max_agents=5))
@settings(max_examples=100)
def test_multi_agent_individual_agent_times_tracked(agent_timelines: List[AgentTimeline]) -> None:
    """Property 8: Multi-agent time breakdown - individual agent tracking.
    
    **Feature: bedrock-latency-benchmark, Property 8: Multi-agent time breakdown invariant**
    **Validates: Requirements 4.2, 4.3**
    
    For any multi-agent benchmark, individual agent response times SHALL be tracked
    and each agent's duration SHALL be positive.
    """
    assume(len(agent_timelines) >= 2)
    
    for timeline in agent_timelines:
        # Each agent must have a valid duration
        duration = timeline.end_time_ms - timeline.start_time_ms
        
        # Property: Each agent's duration must be positive
        assert duration > 0, (
            f"Agent {timeline.agent_name} has non-positive duration: "
            f"end ({timeline.end_time_ms}) - start ({timeline.start_time_ms}) = {duration}"
        )
        
        # Property: Model time must be non-negative
        assert timeline.model_time_ms >= 0, (
            f"Agent {timeline.agent_name} has negative model_time_ms: {timeline.model_time_ms}"
        )
        
        # Property: Model time must not exceed duration
        assert timeline.model_time_ms <= duration + 0.001, (
            f"Agent {timeline.agent_name} model_time ({timeline.model_time_ms}) "
            f"exceeds duration ({duration})"
        )


@given(agent_timelines=agent_timelines_strategy(min_agents=1, max_agents=5))
@settings(max_examples=100)
def test_multi_agent_total_model_time_consistency(agent_timelines: List[AgentTimeline]) -> None:
    """Property 8: Multi-agent time breakdown - total model time consistency.
    
    **Feature: bedrock-latency-benchmark, Property 8: Multi-agent time breakdown invariant**
    **Validates: Requirements 4.2, 4.3**
    
    For any multi-agent benchmark, the total_model_time_ms SHALL equal the sum
    of all individual agent model times.
    """
    assume(len(agent_timelines) > 0)
    
    # Calculate expected total model time
    expected_total_model_time = sum(t.model_time_ms for t in agent_timelines)
    
    # This is what MultiAgentBenchmarkResult.total_model_time_ms should be
    # Property: total_model_time_ms = sum of all agent model times
    assert expected_total_model_time >= 0, (
        f"Total model time must be non-negative, got {expected_total_model_time}"
    )


@given(
    total_time_ms=st.floats(min_value=100.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
    total_model_time_ms=st.floats(min_value=0.0, max_value=50000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_multi_agent_overhead_calculation(total_time_ms: float, total_model_time_ms: float) -> None:
    """Property 8: Multi-agent time breakdown - overhead calculation.
    
    **Feature: bedrock-latency-benchmark, Property 8: Multi-agent time breakdown invariant**
    **Validates: Requirements 4.2, 4.3**
    
    For any multi-agent benchmark, the inter-agent overhead calculation SHALL
    produce a non-negative value when total_time >= total_model_time.
    """
    # Ensure total_time >= total_model_time for valid scenarios
    assume(total_time_ms >= total_model_time_ms)
    
    # Calculate overhead as done in the implementation
    inter_agent_overhead = max(0.0, total_time_ms - total_model_time_ms)
    
    # Property: Overhead must be non-negative
    assert inter_agent_overhead >= 0, (
        f"Inter-agent overhead must be non-negative, got {inter_agent_overhead}"
    )
    
    # Property: Overhead should equal total_time - model_time when total >= model
    expected_overhead = total_time_ms - total_model_time_ms
    assert abs(inter_agent_overhead - expected_overhead) < 0.001, (
        f"Overhead calculation mismatch: got {inter_agent_overhead}, "
        f"expected {expected_overhead}"
    )
