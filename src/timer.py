"""Timer utility for latency measurements.

Provides precise timing for API calls, including TTFB tracking for streaming responses.
"""

import time
from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class LatencyMetrics:
    """Latency metrics for a single benchmark measurement.
    
    Attributes:
        total_time_ms: Total response time in milliseconds
        ttfb_ms: Time to first byte in milliseconds (for streaming)
        model_time_ms: Model inference time in milliseconds
        overhead_ms: Orchestration/framework overhead in milliseconds
    """
    total_time_ms: float
    ttfb_ms: Optional[float] = None
    model_time_ms: Optional[float] = None
    overhead_ms: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize metrics to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "LatencyMetrics":
        """Create metrics from dictionary."""
        return cls(
            total_time_ms=data["total_time_ms"],
            ttfb_ms=data.get("ttfb_ms"),
            model_time_ms=data.get("model_time_ms"),
            overhead_ms=data.get("overhead_ms"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "LatencyMetrics":
        """Deserialize metrics from JSON string."""
        return cls.from_dict(json.loads(json_str))


class Timer:
    """High-precision timer for latency measurements.
    
    Supports tracking total time and time to first byte (TTFB) for streaming.
    
    Usage:
        timer = Timer()
        timer.start()
        # ... perform operation ...
        timer.mark_first_byte()  # Optional, for streaming
        # ... continue operation ...
        metrics = timer.stop()
    """

    def __init__(self) -> None:
        """Initialize timer in stopped state."""
        self._start_time: Optional[float] = None
        self._first_byte_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._model_time_ms: Optional[float] = None
        self._overhead_ms: Optional[float] = None

    def start(self) -> None:
        """Start the timer.
        
        Resets any previous timing data.
        """
        self._start_time = time.perf_counter()
        self._first_byte_time = None
        self._end_time = None
        self._model_time_ms = None
        self._overhead_ms = None

    def mark_first_byte(self) -> None:
        """Mark the time when first byte is received.
        
        Call this when the first chunk of a streaming response arrives.
        Has no effect if timer hasn't been started.
        """
        if self._start_time is not None and self._first_byte_time is None:
            self._first_byte_time = time.perf_counter()

    def set_model_time(self, model_time_ms: float) -> None:
        """Set the model inference time.
        
        Args:
            model_time_ms: Model inference time in milliseconds
        """
        self._model_time_ms = model_time_ms

    def set_overhead(self, overhead_ms: float) -> None:
        """Set the orchestration overhead time.
        
        Args:
            overhead_ms: Overhead time in milliseconds
        """
        self._overhead_ms = overhead_ms

    def stop(self) -> LatencyMetrics:
        """Stop the timer and return metrics.
        
        Returns:
            LatencyMetrics with timing data
            
        Raises:
            RuntimeError: If timer was not started
        """
        if self._start_time is None:
            raise RuntimeError("Timer was not started")

        self._end_time = time.perf_counter()
        
        total_time_ms = (self._end_time - self._start_time) * 1000
        
        ttfb_ms = None
        if self._first_byte_time is not None:
            ttfb_ms = (self._first_byte_time - self._start_time) * 1000

        return LatencyMetrics(
            total_time_ms=total_time_ms,
            ttfb_ms=ttfb_ms,
            model_time_ms=self._model_time_ms,
            overhead_ms=self._overhead_ms,
        )

    @property
    def is_running(self) -> bool:
        """Check if timer is currently running."""
        return self._start_time is not None and self._end_time is None

    @property
    def elapsed_ms(self) -> Optional[float]:
        """Get elapsed time in milliseconds without stopping.
        
        Returns:
            Elapsed time in ms, or None if timer not started
        """
        if self._start_time is None:
            return None
        current = self._end_time if self._end_time else time.perf_counter()
        return (current - self._start_time) * 1000
