"""Property-based tests for configuration module.

**Feature: bedrock-latency-benchmark, Property 6: Configuration parsing round-trip**
"""

import pytest
from hypothesis import given, strategies as st, settings

from src.config import BenchmarkConfig


# Strategy for generating valid BenchmarkConfig instances
config_strategy = st.builds(
    BenchmarkConfig,
    region=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
    iterations=st.integers(min_value=1, max_value=1000),
    warmup_iterations=st.integers(min_value=0, max_value=100),
    prompt=st.text(min_size=0, max_size=500),
    max_tokens=st.integers(min_value=1, max_value=10000),
    models=st.lists(st.text(min_size=1, max_size=100).filter(lambda x: x.strip()), max_size=10),
    output_dir=st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
)


@given(config=config_strategy)
@settings(max_examples=100)
def test_config_json_round_trip(config: BenchmarkConfig) -> None:
    """Property 6: Configuration parsing round-trip.
    
    **Feature: bedrock-latency-benchmark, Property 6: Configuration parsing round-trip**
    **Validates: Requirements 2.7**
    
    For any valid BenchmarkConfig object, serializing to JSON and parsing back
    SHALL produce an equivalent configuration.
    """
    # Serialize to JSON
    json_str = config.to_json()
    
    # Deserialize back
    restored_config = BenchmarkConfig.from_json(json_str)
    
    # Verify equivalence
    assert restored_config.region == config.region
    assert restored_config.iterations == config.iterations
    assert restored_config.warmup_iterations == config.warmup_iterations
    assert restored_config.prompt == config.prompt
    assert restored_config.max_tokens == config.max_tokens
    assert restored_config.models == config.models
    assert restored_config.output_dir == config.output_dir


@given(config=config_strategy)
@settings(max_examples=100)
def test_config_dict_round_trip(config: BenchmarkConfig) -> None:
    """Property 6 (dict variant): Configuration dict round-trip.
    
    **Feature: bedrock-latency-benchmark, Property 6: Configuration parsing round-trip**
    **Validates: Requirements 2.7**
    
    For any valid BenchmarkConfig object, converting to dict and back
    SHALL produce an equivalent configuration.
    """
    # Convert to dict
    config_dict = config.to_dict()
    
    # Restore from dict
    restored_config = BenchmarkConfig.from_dict(config_dict)
    
    # Verify equivalence
    assert restored_config.region == config.region
    assert restored_config.iterations == config.iterations
    assert restored_config.warmup_iterations == config.warmup_iterations
    assert restored_config.prompt == config.prompt
    assert restored_config.max_tokens == config.max_tokens
    assert restored_config.models == config.models
    assert restored_config.output_dir == config.output_dir
