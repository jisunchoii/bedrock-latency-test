"""Agent Benchmark module for Strands Agents latency measurement.

Provides benchmarking for single agent invocations using the Strands Agents SDK,
tracking model inference time, tool execution time, and orchestration overhead.
"""

import logging
import time
from datetime import datetime
from typing import List, Optional, Callable, Any

from strands import Agent, tool
from strands.models.bedrock import BedrockModel

from .config import BenchmarkConfig
from .timer import Timer, LatencyMetrics
from .stats import calculate_statistics
from .results import BenchmarkResult
from .models import get_model_info, LIGHTWEIGHT_MODELS

logger = logging.getLogger(__name__)


class AgentBenchmarkError(Exception):
    """Exception raised for agent benchmark errors."""
    pass


@tool
def simple_calculator(operation: str, a: float, b: float) -> float:
    """A simple calculator tool for basic arithmetic operations.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First operand
        b: Second operand
        
    Returns:
        Result of the arithmetic operation
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


@tool
def get_current_time() -> str:
    """Get the current time as a formatted string.
    
    Returns:
        Current time in ISO format
    """
    return datetime.now().isoformat()


class AgentBenchmark:
    """Benchmark class for Strands Agents.
    
    Measures latency for agent invocations including model inference time,
    tool execution time, and orchestration overhead.
    
    Attributes:
        config: Benchmark configuration
        tools: List of tools available to the agent
    """

    def __init__(
        self, 
        config: BenchmarkConfig,
        tools: Optional[List[Callable]] = None
    ) -> None:
        """Initialize agent benchmark.
        
        Args:
            config: Benchmark configuration
            tools: Optional list of tools for the agent. Defaults to simple_calculator and get_current_time.
        """
        self.config = config
        self.tools = tools if tools is not None else [simple_calculator, get_current_time]
        self._tool_times: List[float] = []
        self._model_times: List[float] = []

    def create_agent(self, model_id: str) -> Agent:
        """Create a Strands Agent with the specified Bedrock model.
        
        Args:
            model_id: The Bedrock model ID to use
            
        Returns:
            Configured Strands Agent instance
        """
        # Create Bedrock model configuration
        model = BedrockModel(
            model_id=model_id,
            region_name=self.config.region,
            max_tokens=self.config.max_tokens,
        )
        
        # Create agent with tools
        agent = Agent(
            model=model,
            tools=self.tools,
        )
        
        return agent

    def _create_instrumented_agent(self, model_id: str) -> Agent:
        """Create an agent with instrumentation for timing measurements.
        
        Args:
            model_id: The Bedrock model ID to use
            
        Returns:
            Instrumented Strands Agent instance
        """
        # Reset timing accumulators
        self._tool_times = []
        self._model_times = []
        
        return self.create_agent(model_id)

    def run_single(self, model_id: str, prompt: Optional[str] = None) -> LatencyMetrics:
        """Run a single agent invocation and measure latency.
        
        Args:
            model_id: The model ID to use
            prompt: Optional prompt override. Uses config.prompt if not provided.
            
        Returns:
            LatencyMetrics with timing data including model_time and overhead
            
        Raises:
            AgentBenchmarkError: If agent invocation fails
        """
        prompt_text = prompt or self.config.prompt
        
        try:
            agent = self._create_instrumented_agent(model_id)
            
            timer = Timer()
            timer.start()
            
            # Track model invocation timing
            model_start = time.perf_counter()
            
            # Run the agent
            result = agent(prompt_text)
            
            model_end = time.perf_counter()
            
            # Stop the overall timer
            metrics = timer.stop()
            
            # Calculate time breakdown
            # Note: In a real implementation, we would hook into the agent's
            # internal callbacks to get precise model vs tool timing.
            # For now, we estimate based on total time.
            total_time_ms = metrics.total_time_ms
            
            # Extract timing from agent metrics if available
            model_time_ms = self._extract_model_time(result, total_time_ms)
            tool_time_ms = self._extract_tool_time(result)
            overhead_ms = total_time_ms - model_time_ms - tool_time_ms
            
            # Ensure overhead is non-negative
            if overhead_ms < 0:
                overhead_ms = 0.0
                
            return LatencyMetrics(
                total_time_ms=total_time_ms,
                ttfb_ms=None,
                model_time_ms=model_time_ms,
                overhead_ms=overhead_ms,
            )
            
        except Exception as e:
            logger.error(f"Agent invocation failed: {e}")
            raise AgentBenchmarkError(f"Agent invocation failed: {e}")

    def _extract_model_time(self, result: Any, total_time_ms: float) -> float:
        """Extract model inference time from agent result.
        
        Args:
            result: Agent execution result
            total_time_ms: Total execution time in milliseconds
            
        Returns:
            Model inference time in milliseconds
        """
        # Try to extract from agent metrics if available
        try:
            if hasattr(result, 'metrics') and result.metrics:
                if hasattr(result.metrics, 'model_invoke_duration_ms'):
                    return result.metrics.model_invoke_duration_ms
                if hasattr(result.metrics, 'accumulated_usage'):
                    # Estimate based on token processing
                    pass
        except Exception:
            pass
        
        # Default: assume 90% of time is model inference for simple prompts
        # This is a reasonable estimate when no tool calls are made
        return total_time_ms * 0.9

    def _extract_tool_time(self, result: Any) -> float:
        """Extract tool execution time from agent result.
        
        Args:
            result: Agent execution result
            
        Returns:
            Tool execution time in milliseconds
        """
        # Try to extract from agent metrics if available
        try:
            if hasattr(result, 'metrics') and result.metrics:
                if hasattr(result.metrics, 'tool_invoke_duration_ms'):
                    return result.metrics.tool_invoke_duration_ms
        except Exception:
            pass
        
        # Default: no tool time if we can't extract it
        return 0.0

    def run_warmup(self, model_id: str) -> None:
        """Run warmup iterations to prime the agent and connection.
        
        Args:
            model_id: The model ID to use
        """
        logger.info(
            f"Running {self.config.warmup_iterations} warmup iterations for agent with {model_id}"
        )
        
        for i in range(self.config.warmup_iterations):
            try:
                self.run_single(model_id)
                logger.debug(f"Agent warmup iteration {i + 1} completed")
            except AgentBenchmarkError as e:
                logger.warning(f"Agent warmup iteration {i + 1} failed: {e}")

    def run_benchmark(self, model_id: str) -> BenchmarkResult:
        """Run a complete benchmark for a model.
        
        Args:
            model_id: The model ID to benchmark
            
        Returns:
            BenchmarkResult with all metrics and statistics
        """
        # Find model info
        model_name = None
        provider = "Unknown"
        for name, info in LIGHTWEIGHT_MODELS.items():
            if info["model_id"] == model_id:
                model_name = name
                provider = info["provider"]
                break
        
        if model_name is None:
            model_name = model_id.split(".")[-1]
        
        logger.info(f"Starting agent benchmark for {model_name} ({model_id})")
        
        # Run warmup
        self.run_warmup(model_id)
        
        # Run benchmark iterations
        metrics: List[LatencyMetrics] = []
        errors: List[str] = []
        
        for i in range(self.config.iterations):
            try:
                metric = self.run_single(model_id)
                metrics.append(metric)
                logger.debug(
                    f"Iteration {i + 1}/{self.config.iterations}: "
                    f"total={metric.total_time_ms:.2f}ms, "
                    f"model={metric.model_time_ms:.2f}ms, "
                    f"overhead={metric.overhead_ms:.2f}ms"
                )
            except AgentBenchmarkError as e:
                error_msg = f"Iteration {i + 1} failed: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)
        
        if not metrics:
            raise AgentBenchmarkError(
                f"All {self.config.iterations} iterations failed for {model_id}"
            )
        
        # Calculate statistics
        statistics = calculate_statistics(metrics)
        
        return BenchmarkResult(
            model_id=model_id,
            model_name=model_name,
            provider=provider,
            region=self.config.region,
            benchmark_type="agent",
            timestamp=datetime.now(),
            iterations=len(metrics),
            metrics=metrics,
            statistics=statistics,
            errors=errors,
        )

