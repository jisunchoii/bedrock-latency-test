"""Statistics calculator for benchmark results.

Provides statistical analysis of latency measurements including
min, max, average, median, percentiles, and standard deviation.
"""

import math
from dataclasses import dataclass, asdict
from typing import List
import json

from .timer import LatencyMetrics


@dataclass
class BenchmarkStatistics:
    """Statistical summary of benchmark latency measurements.
    
    Attributes:
        min_ms: Minimum latency in milliseconds
        max_ms: Maximum latency in milliseconds
        avg_ms: Average (mean) latency in milliseconds
        median_ms: Median (50th percentile) latency in milliseconds
        p95_ms: 95th percentile latency in milliseconds
        p99_ms: 99th percentile latency in milliseconds
        std_dev_ms: Standard deviation in milliseconds
    """
    min_ms: float
    max_ms: float
    avg_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    std_dev_ms: float

    def to_dict(self) -> dict:
        """Convert statistics to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize statistics to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkStatistics":
        """Create statistics from dictionary."""
        return cls(
            min_ms=data["min_ms"],
            max_ms=data["max_ms"],
            avg_ms=data["avg_ms"],
            median_ms=data["median_ms"],
            p95_ms=data["p95_ms"],
            p99_ms=data["p99_ms"],
            std_dev_ms=data["std_dev_ms"],
        )

    @classmethod
    def from_json(cls, json_str: str) -> "BenchmarkStatistics":
        """Deserialize statistics from JSON string."""
        return cls.from_dict(json.loads(json_str))


def _percentile(sorted_values: List[float], p: float) -> float:
    """Calculate percentile from sorted values.
    
    Uses linear interpolation method with clamping to handle
    floating-point precision issues.
    
    Args:
        sorted_values: List of values sorted in ascending order
        p: Percentile to calculate (0-100)
        
    Returns:
        Percentile value (clamped to min/max bounds)
    """
    if not sorted_values:
        raise ValueError("Cannot calculate percentile of empty list")
    
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    
    # Calculate the index for the percentile
    k = (p / 100) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    
    if f == c:
        return sorted_values[int(k)]
    
    # Linear interpolation
    result = sorted_values[f] * (c - k) + sorted_values[c] * (k - f)
    
    # Clamp to min/max bounds to handle floating-point precision issues
    return max(sorted_values[0], min(result, sorted_values[-1]))


def calculate_statistics(metrics: List[LatencyMetrics]) -> BenchmarkStatistics:
    """Calculate statistics from a list of latency metrics.
    
    Args:
        metrics: List of LatencyMetrics from benchmark runs
        
    Returns:
        BenchmarkStatistics with calculated values
        
    Raises:
        ValueError: If metrics list is empty
    """
    if not metrics:
        raise ValueError("Cannot calculate statistics from empty metrics list")
    
    # Extract total_time_ms values
    values = [m.total_time_ms for m in metrics]
    n = len(values)
    
    # Sort for percentile calculations
    sorted_values = sorted(values)
    
    # Calculate basic statistics
    min_ms = sorted_values[0]
    max_ms = sorted_values[-1]
    avg_ms = sum(values) / n
    
    # Clamp avg to min/max bounds to handle floating-point precision issues
    avg_ms = max(min_ms, min(avg_ms, max_ms))
    
    # Calculate median (50th percentile)
    median_ms = _percentile(sorted_values, 50)
    
    # Calculate p95 and p99
    p95_ms = _percentile(sorted_values, 95)
    p99_ms = _percentile(sorted_values, 99)
    
    # Calculate standard deviation
    if n == 1:
        std_dev_ms = 0.0
    else:
        variance = sum((x - avg_ms) ** 2 for x in values) / (n - 1)
        std_dev_ms = math.sqrt(variance)
    
    return BenchmarkStatistics(
        min_ms=min_ms,
        max_ms=max_ms,
        avg_ms=avg_ms,
        median_ms=median_ms,
        p95_ms=p95_ms,
        p99_ms=p99_ms,
        std_dev_ms=std_dev_ms,
    )


def calculate_ttfb_statistics(metrics: List[LatencyMetrics]) -> BenchmarkStatistics:
    """Calculate statistics for TTFB values from metrics.
    
    Args:
        metrics: List of LatencyMetrics with ttfb_ms values
        
    Returns:
        BenchmarkStatistics for TTFB values
        
    Raises:
        ValueError: If no metrics have TTFB values
    """
    # Filter metrics that have TTFB values
    ttfb_metrics = [
        LatencyMetrics(total_time_ms=m.ttfb_ms)
        for m in metrics
        if m.ttfb_ms is not None
    ]
    
    if not ttfb_metrics:
        raise ValueError("No metrics have TTFB values")
    
    return calculate_statistics(ttfb_metrics)
