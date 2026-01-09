"""Benchmark result dataclasses and serialization.

Provides data structures for storing and serializing benchmark results.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional

from .timer import LatencyMetrics
from .stats import BenchmarkStatistics


@dataclass
class BenchmarkResult:
    """Result of a benchmark run.
    
    Attributes:
        model_id: Full model ID string
        model_name: Short model name
        provider: Model provider (e.g., Anthropic, Amazon)
        region: AWS region where benchmark was run
        benchmark_type: Type of benchmark (api, agent, multi-agent)
        timestamp: When the benchmark was run
        iterations: Number of iterations executed
        metrics: List of latency metrics from each iteration
        statistics: Calculated statistics from metrics
        errors: List of error messages encountered
    """
    model_id: str
    model_name: str
    provider: str
    region: str
    benchmark_type: str
    timestamp: datetime
    iterations: int
    metrics: List[LatencyMetrics]
    statistics: BenchmarkStatistics
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert result to dictionary.
        
        Returns:
            Dictionary representation of the result
        """
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "provider": self.provider,
            "region": self.region,
            "benchmark_type": self.benchmark_type,
            "timestamp": self.timestamp.isoformat(),
            "iterations": self.iterations,
            "metrics": [m.to_dict() for m in self.metrics],
            "statistics": self.statistics.to_dict(),
            "errors": self.errors,
        }


    def to_json(self) -> str:
        """Serialize result to JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkResult":
        """Create result from dictionary.
        
        Args:
            data: Dictionary containing result data
            
        Returns:
            BenchmarkResult instance
        """
        return cls(
            model_id=data["model_id"],
            model_name=data["model_name"],
            provider=data["provider"],
            region=data["region"],
            benchmark_type=data["benchmark_type"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            iterations=data["iterations"],
            metrics=[LatencyMetrics.from_dict(m) for m in data["metrics"]],
            statistics=BenchmarkStatistics.from_dict(data["statistics"]),
            errors=data.get("errors", []),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "BenchmarkResult":
        """Deserialize result from JSON string.
        
        Args:
            json_str: JSON string containing result data
            
        Returns:
            BenchmarkResult instance
        """
        return cls.from_dict(json.loads(json_str))

    def save_to_file(self, filepath: str) -> None:
        """Save result to JSON file.
        
        Args:
            filepath: Path to save the result
        """
        with open(filepath, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load_from_file(cls, filepath: str) -> "BenchmarkResult":
        """Load result from JSON file.
        
        Args:
            filepath: Path to the result file
            
        Returns:
            BenchmarkResult instance
        """
        with open(filepath, "r") as f:
            return cls.from_json(f.read())


@dataclass
class AgentTimeline:
    """Timeline entry for an agent in multi-agent benchmark.
    
    Attributes:
        agent_name: Name of the agent
        start_time_ms: Start time relative to benchmark start
        end_time_ms: End time relative to benchmark start
        model_time_ms: Time spent in model inference
    """
    agent_name: str
    start_time_ms: float
    end_time_ms: float
    model_time_ms: float

    def to_dict(self) -> dict:
        """Convert timeline to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AgentTimeline":
        """Create timeline from dictionary."""
        return cls(
            agent_name=data["agent_name"],
            start_time_ms=data["start_time_ms"],
            end_time_ms=data["end_time_ms"],
            model_time_ms=data["model_time_ms"],
        )



@dataclass
class MultiAgentBenchmarkResult(BenchmarkResult):
    """Result of a multi-agent benchmark run.
    
    Extends BenchmarkResult with multi-agent specific data.
    
    Attributes:
        agent_timelines: Timeline of each agent's execution
        inter_agent_overhead_ms: Overhead from inter-agent communication
        total_model_time_ms: Total time spent in model inference across all agents
    """
    agent_timelines: List[AgentTimeline] = field(default_factory=list)
    inter_agent_overhead_ms: float = 0.0
    total_model_time_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert result to dictionary.
        
        Returns:
            Dictionary representation of the result
        """
        base_dict = super().to_dict()
        base_dict["agent_timelines"] = [t.to_dict() for t in self.agent_timelines]
        base_dict["inter_agent_overhead_ms"] = self.inter_agent_overhead_ms
        base_dict["total_model_time_ms"] = self.total_model_time_ms
        return base_dict

    @classmethod
    def from_dict(cls, data: dict) -> "MultiAgentBenchmarkResult":
        """Create result from dictionary.
        
        Args:
            data: Dictionary containing result data
            
        Returns:
            MultiAgentBenchmarkResult instance
        """
        return cls(
            model_id=data["model_id"],
            model_name=data["model_name"],
            provider=data["provider"],
            region=data["region"],
            benchmark_type=data["benchmark_type"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            iterations=data["iterations"],
            metrics=[LatencyMetrics.from_dict(m) for m in data["metrics"]],
            statistics=BenchmarkStatistics.from_dict(data["statistics"]),
            errors=data.get("errors", []),
            agent_timelines=[
                AgentTimeline.from_dict(t) for t in data.get("agent_timelines", [])
            ],
            inter_agent_overhead_ms=data.get("inter_agent_overhead_ms", 0.0),
            total_model_time_ms=data.get("total_model_time_ms", 0.0),
        )
