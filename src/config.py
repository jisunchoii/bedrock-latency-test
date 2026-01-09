"""Configuration module for Bedrock Latency Benchmark."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution.
    
    Attributes:
        region: AWS region for Bedrock API calls
        iterations: Number of benchmark iterations to run
        warmup_iterations: Number of warmup iterations before measurement
        prompt: Test prompt to send to models
        max_tokens: Maximum tokens in model response
        models: List of model IDs to benchmark
        output_dir: Directory for saving benchmark results
    """
    region: str = "ap-northeast-2"
    iterations: int = 10
    warmup_iterations: int = 2
    prompt: str = "Hello, how are you?"
    max_tokens: int = 100
    models: List[str] = field(default_factory=list)
    output_dir: str = "./benchmark_results"

    def to_json(self) -> str:
        """Serialize configuration to JSON string."""
        return json.dumps(asdict(self), indent=2)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_json(cls, json_str: str) -> "BenchmarkConfig":
        """Deserialize configuration from JSON string.
        
        Args:
            json_str: JSON string containing configuration
            
        Returns:
            BenchmarkConfig instance
            
        Raises:
            json.JSONDecodeError: If JSON is invalid
            TypeError: If required fields are missing or have wrong types
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkConfig":
        """Create configuration from dictionary.
        
        Args:
            data: Dictionary containing configuration values
            
        Returns:
            BenchmarkConfig instance
        """
        return cls(
            region=data.get("region", "ap-northeast-2"),
            iterations=data.get("iterations", 10),
            warmup_iterations=data.get("warmup_iterations", 2),
            prompt=data.get("prompt", "Hello, how are you?"),
            max_tokens=data.get("max_tokens", 100),
            models=data.get("models", []),
            output_dir=data.get("output_dir", "./benchmark_results"),
        )

    @classmethod
    def from_file(cls, filepath: str) -> "BenchmarkConfig":
        """Load configuration from JSON file.
        
        Args:
            filepath: Path to JSON configuration file
            
        Returns:
            BenchmarkConfig instance
        """
        with open(filepath, "r") as f:
            return cls.from_json(f.read())

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file.
        
        Args:
            filepath: Path to save configuration
        """
        with open(filepath, "w") as f:
            f.write(self.to_json())
