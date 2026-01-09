"""Property-based tests for report module.

**Feature: bedrock-latency-benchmark, Property 9: Output contains required fields**
"""

import json
import os
import tempfile
from datetime import datetime
from io import StringIO
from typing import List

import pytest
from hypothesis import given, strategies as st, settings
from rich.console import Console

from src.timer import LatencyMetrics
from src.stats import BenchmarkStatistics, calculate_statistics
from src.results import BenchmarkResult, MultiAgentBenchmarkResult, AgentTimeline
from src.report import ReportGenerator


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


# Strategy for generating valid datetime
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
def test_json_output_contains_required_fields(result: BenchmarkResult) -> None:
    """Property 9: Output contains required fields (JSON format).
    
    **Feature: bedrock-latency-benchmark, Property 9: Output contains required fields**
    **Validates: Requirements 1.3, 3.4, 4.4, 5.2**
    
    For any benchmark result output (JSON), the output SHALL contain
    model_id, region, benchmark_type, and all latency statistics.
    """
    # Generate JSON report to a temporary file
    generator = ReportGenerator()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_report.json")
        generator.generate_json_report(result, filepath)
        
        # Read and parse the JSON report
        with open(filepath, "r") as f:
            report_data = json.load(f)
        
        # Verify report structure
        assert "results" in report_data, "JSON report must contain 'results' key"
        assert len(report_data["results"]) > 0, "JSON report must contain at least one result"
        
        result_data = report_data["results"][0]
        
        # Property: model_id is present
        assert "model_id" in result_data, "JSON output must contain 'model_id'"
        assert result_data["model_id"] == result.model_id, (
            f"model_id mismatch: expected {result.model_id}, got {result_data['model_id']}"
        )
        
        # Property: region is present
        assert "region" in result_data, "JSON output must contain 'region'"
        assert result_data["region"] == result.region, (
            f"region mismatch: expected {result.region}, got {result_data['region']}"
        )
        
        # Property: benchmark_type is present
        assert "benchmark_type" in result_data, "JSON output must contain 'benchmark_type'"
        assert result_data["benchmark_type"] == result.benchmark_type, (
            f"benchmark_type mismatch: expected {result.benchmark_type}, got {result_data['benchmark_type']}"
        )
        
        # Property: statistics are present with all required fields
        assert "statistics" in result_data, "JSON output must contain 'statistics'"
        stats = result_data["statistics"]
        
        required_stat_fields = ["min_ms", "max_ms", "avg_ms", "median_ms", "p95_ms", "p99_ms"]
        for field in required_stat_fields:
            assert field in stats, f"JSON output statistics must contain '{field}'"


@given(results=st.lists(benchmark_result_strategy, min_size=1, max_size=5))
@settings(max_examples=100)
def test_json_output_contains_required_fields_multiple_results(results: List[BenchmarkResult]) -> None:
    """Property 9: Output contains required fields for multiple results (JSON format).
    
    **Feature: bedrock-latency-benchmark, Property 9: Output contains required fields**
    **Validates: Requirements 1.3, 3.4, 4.4, 5.2**
    
    For any list of benchmark results, the JSON output SHALL contain
    model_id, region, benchmark_type, and all latency statistics for each result.
    """
    generator = ReportGenerator()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_report.json")
        generator.generate_json_report(results, filepath)
        
        with open(filepath, "r") as f:
            report_data = json.load(f)
        
        assert "results" in report_data, "JSON report must contain 'results' key"
        assert len(report_data["results"]) == len(results), (
            f"Expected {len(results)} results, got {len(report_data['results'])}"
        )
        
        for i, (original, saved) in enumerate(zip(results, report_data["results"])):
            # Property: Each result contains required fields
            assert "model_id" in saved, f"Result {i}: JSON output must contain 'model_id'"
            assert "region" in saved, f"Result {i}: JSON output must contain 'region'"
            assert "benchmark_type" in saved, f"Result {i}: JSON output must contain 'benchmark_type'"
            assert "statistics" in saved, f"Result {i}: JSON output must contain 'statistics'"
            
            # Verify values match
            assert saved["model_id"] == original.model_id
            assert saved["region"] == original.region
            assert saved["benchmark_type"] == original.benchmark_type


@given(result=benchmark_result_strategy)
@settings(max_examples=100)
def test_result_to_dict_contains_required_fields(result: BenchmarkResult) -> None:
    """Property 9: BenchmarkResult.to_dict() contains required fields.
    
    **Feature: bedrock-latency-benchmark, Property 9: Output contains required fields**
    **Validates: Requirements 1.3, 3.4, 4.4, 5.2**
    
    For any BenchmarkResult, the to_dict() output SHALL contain
    model_id, region, benchmark_type, and all latency statistics.
    """
    result_dict = result.to_dict()
    
    # Property: Required fields are present
    assert "model_id" in result_dict, "to_dict() must contain 'model_id'"
    assert "region" in result_dict, "to_dict() must contain 'region'"
    assert "benchmark_type" in result_dict, "to_dict() must contain 'benchmark_type'"
    assert "statistics" in result_dict, "to_dict() must contain 'statistics'"
    
    # Property: Statistics contain all required fields
    stats = result_dict["statistics"]
    required_stat_fields = ["min_ms", "max_ms", "avg_ms", "median_ms", "p95_ms", "p99_ms"]
    for field in required_stat_fields:
        assert field in stats, f"statistics must contain '{field}'"
    
    # Property: Values match original
    assert result_dict["model_id"] == result.model_id
    assert result_dict["region"] == result.region
    assert result_dict["benchmark_type"] == result.benchmark_type
    assert result_dict["statistics"]["min_ms"] == result.statistics.min_ms
    assert result_dict["statistics"]["max_ms"] == result.statistics.max_ms
    assert result_dict["statistics"]["avg_ms"] == result.statistics.avg_ms
    assert result_dict["statistics"]["median_ms"] == result.statistics.median_ms
    assert result_dict["statistics"]["p95_ms"] == result.statistics.p95_ms
    assert result_dict["statistics"]["p99_ms"] == result.statistics.p99_ms


@given(
    stats1=benchmark_statistics_strategy,
    stats2=benchmark_statistics_strategy,
    threshold=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_comparison_report_consistency(
    stats1: BenchmarkStatistics,
    stats2: BenchmarkStatistics,
    threshold: float,
) -> None:
    """Property 10: Comparison report consistency.
    
    **Feature: bedrock-latency-benchmark, Property 10: Comparison report consistency**
    **Validates: Requirements 5.3**
    
    For any two valid benchmark reports, the comparison report SHALL correctly
    identify which metrics improved, degraded, or remained stable.
    """
    generator = ReportGenerator()
    
    # Create two benchmark results with the generated statistics
    result1 = BenchmarkResult(
        model_id="test-model",
        model_name="Test Model",
        provider="Test Provider",
        region="ap-northeast-2",
        benchmark_type="api",
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        iterations=10,
        metrics=[LatencyMetrics(total_time_ms=100.0)],
        statistics=stats1,
        errors=[],
    )
    
    result2 = BenchmarkResult(
        model_id="test-model",
        model_name="Test Model",
        provider="Test Provider",
        region="ap-northeast-2",
        benchmark_type="api",
        timestamp=datetime(2024, 1, 2, 12, 0, 0),
        iterations=10,
        metrics=[LatencyMetrics(total_time_ms=100.0)],
        statistics=stats2,
        errors=[],
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate JSON reports
        report1_path = os.path.join(tmpdir, "report1.json")
        report2_path = os.path.join(tmpdir, "report2.json")
        
        generator.generate_json_report(result1, report1_path)
        generator.generate_json_report(result2, report2_path)
        
        # Compare reports
        comparison = generator.compare_reports(report1_path, report2_path, threshold_percent=threshold)
        
        # Verify comparison correctness for each metric
        metrics_map = {
            "Min": ("min_ms", stats1.min_ms, stats2.min_ms),
            "Max": ("max_ms", stats1.max_ms, stats2.max_ms),
            "Average": ("avg_ms", stats1.avg_ms, stats2.avg_ms),
            "Median": ("median_ms", stats1.median_ms, stats2.median_ms),
            "P95": ("p95_ms", stats1.p95_ms, stats2.p95_ms),
            "P99": ("p99_ms", stats1.p99_ms, stats2.p99_ms),
        }
        
        for comp in comparison.comparisons:
            metric_key, value1, value2 = metrics_map[comp.metric_name]
            
            # Verify values are correctly captured
            assert comp.value1 == value1, (
                f"Metric {comp.metric_name}: expected value1={value1}, got {comp.value1}"
            )
            assert comp.value2 == value2, (
                f"Metric {comp.metric_name}: expected value2={value2}, got {comp.value2}"
            )
            
            # Verify diff calculation
            expected_diff_ms = value2 - value1
            assert abs(comp.diff_ms - expected_diff_ms) < 1e-9, (
                f"Metric {comp.metric_name}: expected diff_ms={expected_diff_ms}, got {comp.diff_ms}"
            )
            
            # Verify diff_percent calculation
            if value1 != 0:
                expected_diff_percent = (expected_diff_ms / value1) * 100
                assert abs(comp.diff_percent - expected_diff_percent) < 1e-9, (
                    f"Metric {comp.metric_name}: expected diff_percent={expected_diff_percent}, got {comp.diff_percent}"
                )
            
            # Verify status classification is correct
            # Lower latency is better, so:
            # - improved: value2 < value1 (negative diff) and |diff_percent| > threshold
            # - degraded: value2 > value1 (positive diff) and |diff_percent| > threshold
            # - stable: |diff_percent| <= threshold
            if abs(comp.diff_percent) <= threshold:
                expected_status = "stable"
            elif comp.diff_percent < 0:
                expected_status = "improved"
            else:
                expected_status = "degraded"
            
            assert comp.status == expected_status, (
                f"Metric {comp.metric_name}: expected status='{expected_status}', got '{comp.status}'. "
                f"diff_percent={comp.diff_percent}, threshold={threshold}"
            )

