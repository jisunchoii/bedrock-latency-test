"""Multi-Agent Benchmark module for Strands Agents latency measurement.

Provides benchmarking for multi-agent patterns using the Agents-as-Tools approach,
tracking individual agent times, inter-agent communication overhead, and total latency.
"""

import logging
import time
from datetime import datetime
from typing import List, Optional, Callable, Any, Dict

from strands import Agent, tool
from strands.models.bedrock import BedrockModel

from .config import BenchmarkConfig
from .timer import Timer, LatencyMetrics
from .stats import calculate_statistics
from .results import MultiAgentBenchmarkResult, AgentTimeline
from .models import get_model_info, LIGHTWEIGHT_MODELS

logger = logging.getLogger(__name__)


class MultiAgentBenchmarkError(Exception):
    """Exception raised for multi-agent benchmark errors."""
    pass


class MultiAgentBenchmark:
    """Benchmark class for multi-agent patterns using Agents-as-Tools.
    
    Measures latency for multi-agent collaborations including individual agent times,
    inter-agent communication overhead, and total end-to-end response time.
    
    The Agents-as-Tools pattern uses a supervisor agent that can delegate tasks
    to worker agents exposed as tools.
    
    Attributes:
        config: Benchmark configuration
        _agent_timelines: List of agent execution timelines for current run
        _benchmark_start_time: Start time of current benchmark iteration
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize multi-agent benchmark.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self._agent_timelines: List[AgentTimeline] = []
        self._benchmark_start_time: float = 0.0
        self._total_model_time_ms: float = 0.0


    def _create_bedrock_model(self, model_id: str) -> BedrockModel:
        """Create a Bedrock model configuration.
        
        Args:
            model_id: The Bedrock model ID to use
            
        Returns:
            Configured BedrockModel instance
        """
        return BedrockModel(
            model_id=model_id,
            region_name=self.config.region,
            max_tokens=self.config.max_tokens,
        )

    def _record_agent_timeline(
        self,
        agent_name: str,
        start_time: float,
        end_time: float,
        model_time_ms: float = 0.0,
    ) -> None:
        """Record an agent's execution timeline.
        
        Args:
            agent_name: Name of the agent
            start_time: Start time (perf_counter value)
            end_time: End time (perf_counter value)
            model_time_ms: Time spent in model inference
        """
        start_ms = (start_time - self._benchmark_start_time) * 1000
        end_ms = (end_time - self._benchmark_start_time) * 1000
        
        timeline = AgentTimeline(
            agent_name=agent_name,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
            model_time_ms=model_time_ms,
        )
        self._agent_timelines.append(timeline)
        self._total_model_time_ms += model_time_ms

    def create_worker_agent(
        self,
        model_id: str,
        role: str,
        system_prompt: Optional[str] = None,
    ) -> Agent:
        """Create a worker agent with the specified role.
        
        Args:
            model_id: The Bedrock model ID to use
            role: Role/name of the worker agent
            system_prompt: Optional system prompt for the agent
            
        Returns:
            Configured worker Agent instance
        """
        model = self._create_bedrock_model(model_id)
        
        default_prompt = f"You are a helpful {role} assistant. Respond concisely."
        
        agent = Agent(
            model=model,
            system_prompt=system_prompt or default_prompt,
        )
        
        return agent

    def _create_worker_tool(
        self,
        model_id: str,
        worker_name: str,
        worker_description: str,
    ) -> Callable:
        """Create a tool that wraps a worker agent.
        
        This implements the Agents-as-Tools pattern where worker agents
        are exposed as tools to the supervisor agent.
        
        Args:
            model_id: The Bedrock model ID for the worker
            worker_name: Name of the worker agent
            worker_description: Description of what the worker does
            
        Returns:
            A tool function that invokes the worker agent
        """
        worker_agent = self.create_worker_agent(model_id, worker_name)
        benchmark_instance = self
        
        @tool(name=worker_name, description=worker_description)
        def worker_tool(task: str) -> str:
            """Execute a task using the worker agent.
            
            Args:
                task: The task description to send to the worker
                
            Returns:
                The worker agent's response
            """
            start_time = time.perf_counter()
            
            try:
                result = worker_agent(task)
                response = str(result)
            except Exception as e:
                logger.error(f"Worker {worker_name} failed: {e}")
                response = f"Error: {e}"
            
            end_time = time.perf_counter()
            
            # Estimate model time (90% of total for simple responses)
            duration_ms = (end_time - start_time) * 1000
            model_time_ms = duration_ms * 0.9
            
            benchmark_instance._record_agent_timeline(
                agent_name=worker_name,
                start_time=start_time,
                end_time=end_time,
                model_time_ms=model_time_ms,
            )
            
            return response
        
        return worker_tool


    def create_supervisor(
        self,
        model_id: str,
        worker_tools: List[Callable],
        system_prompt: Optional[str] = None,
    ) -> Agent:
        """Create a supervisor agent with worker tools.
        
        Args:
            model_id: The Bedrock model ID to use
            worker_tools: List of worker agent tools
            system_prompt: Optional system prompt for the supervisor
            
        Returns:
            Configured supervisor Agent instance
        """
        model = self._create_bedrock_model(model_id)
        
        default_prompt = (
            "You are a supervisor agent that coordinates tasks between worker agents. "
            "Delegate tasks to the appropriate workers and synthesize their responses. "
            "Be concise in your final response."
        )
        
        agent = Agent(
            model=model,
            tools=worker_tools,
            system_prompt=system_prompt or default_prompt,
        )
        
        return agent

    def _reset_iteration_state(self) -> None:
        """Reset state for a new benchmark iteration."""
        self._agent_timelines = []
        self._benchmark_start_time = 0.0
        self._total_model_time_ms = 0.0

    def run_single(
        self,
        model_id: str,
        prompt: Optional[str] = None,
        worker_configs: Optional[List[Dict[str, str]]] = None,
    ) -> LatencyMetrics:
        """Run a single multi-agent invocation and measure latency.
        
        Args:
            model_id: The model ID to use for all agents
            prompt: Optional prompt override. Uses config.prompt if not provided.
            worker_configs: Optional list of worker configurations.
                Each config should have 'name' and 'description' keys.
                Defaults to researcher and writer workers.
            
        Returns:
            LatencyMetrics with timing data
            
        Raises:
            MultiAgentBenchmarkError: If multi-agent invocation fails
        """
        self._reset_iteration_state()
        
        prompt_text = prompt or self.config.prompt
        
        # Default worker configurations
        if worker_configs is None:
            worker_configs = [
                {
                    "name": "researcher",
                    "description": "Research and gather information on a topic",
                },
                {
                    "name": "writer",
                    "description": "Write and compose text based on given information",
                },
            ]
        
        try:
            # Create worker tools
            worker_tools = [
                self._create_worker_tool(
                    model_id=model_id,
                    worker_name=config["name"],
                    worker_description=config["description"],
                )
                for config in worker_configs
            ]
            
            # Create supervisor
            supervisor = self.create_supervisor(model_id, worker_tools)
            
            # Start timing
            timer = Timer()
            timer.start()
            self._benchmark_start_time = time.perf_counter()
            
            # Record supervisor start
            supervisor_start = time.perf_counter()
            
            # Run the supervisor agent
            result = supervisor(prompt_text)
            
            supervisor_end = time.perf_counter()
            
            # Stop the overall timer
            metrics = timer.stop()
            
            # Record supervisor timeline
            # Supervisor model time is total minus worker times
            supervisor_duration_ms = (supervisor_end - supervisor_start) * 1000
            worker_total_ms = sum(t.end_time_ms - t.start_time_ms for t in self._agent_timelines)
            supervisor_model_time_ms = max(0, (supervisor_duration_ms - worker_total_ms) * 0.9)
            
            self._record_agent_timeline(
                agent_name="supervisor",
                start_time=supervisor_start,
                end_time=supervisor_end,
                model_time_ms=supervisor_model_time_ms,
            )
            
            # Calculate overhead
            total_time_ms = metrics.total_time_ms
            total_model_time = self._total_model_time_ms
            overhead_ms = max(0, total_time_ms - total_model_time)
            
            return LatencyMetrics(
                total_time_ms=total_time_ms,
                ttfb_ms=None,
                model_time_ms=total_model_time,
                overhead_ms=overhead_ms,
            )
            
        except Exception as e:
            logger.error(f"Multi-agent invocation failed: {e}")
            raise MultiAgentBenchmarkError(f"Multi-agent invocation failed: {e}")


    def get_agent_timelines(self) -> List[AgentTimeline]:
        """Get the agent timelines from the last run.
        
        Returns:
            List of AgentTimeline objects from the most recent run
        """
        return self._agent_timelines.copy()

    def calculate_inter_agent_overhead(self, total_time_ms: float) -> float:
        """Calculate inter-agent communication overhead.
        
        The overhead is the time not accounted for by individual agent executions.
        
        Args:
            total_time_ms: Total benchmark time in milliseconds
            
        Returns:
            Inter-agent overhead in milliseconds
        """
        if not self._agent_timelines:
            return 0.0
        
        # Sum of all agent execution times
        agent_execution_time = sum(
            t.end_time_ms - t.start_time_ms for t in self._agent_timelines
        )
        
        # Overhead is total time minus agent execution time
        # Note: This can be negative if agents run in parallel, so we take max with 0
        overhead = max(0, total_time_ms - agent_execution_time)
        
        return overhead

    def run_warmup(self, model_id: str) -> None:
        """Run warmup iterations to prime the agents and connections.
        
        Args:
            model_id: The model ID to use
        """
        logger.info(
            f"Running {self.config.warmup_iterations} warmup iterations for multi-agent with {model_id}"
        )
        
        for i in range(self.config.warmup_iterations):
            try:
                self.run_single(model_id)
                logger.debug(f"Multi-agent warmup iteration {i + 1} completed")
            except MultiAgentBenchmarkError as e:
                logger.warning(f"Multi-agent warmup iteration {i + 1} failed: {e}")

    def run_benchmark(
        self,
        model_id: str,
        worker_configs: Optional[List[Dict[str, str]]] = None,
    ) -> MultiAgentBenchmarkResult:
        """Run a complete multi-agent benchmark for a model.
        
        Args:
            model_id: The model ID to benchmark
            worker_configs: Optional list of worker configurations
            
        Returns:
            MultiAgentBenchmarkResult with all metrics and statistics
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
        
        logger.info(f"Starting multi-agent benchmark for {model_name} ({model_id})")
        
        # Run warmup
        self.run_warmup(model_id)
        
        # Run benchmark iterations
        metrics: List[LatencyMetrics] = []
        errors: List[str] = []
        all_timelines: List[List[AgentTimeline]] = []
        total_inter_agent_overhead: float = 0.0
        total_model_time_accumulated: float = 0.0
        
        for i in range(self.config.iterations):
            try:
                metric = self.run_single(model_id, worker_configs=worker_configs)
                metrics.append(metric)
                
                # Capture timelines for this iteration
                iteration_timelines = self.get_agent_timelines()
                all_timelines.append(iteration_timelines)
                
                # Accumulate overhead and model time
                inter_agent_overhead = self.calculate_inter_agent_overhead(metric.total_time_ms)
                total_inter_agent_overhead += inter_agent_overhead
                total_model_time_accumulated += self._total_model_time_ms
                
                logger.debug(
                    f"Iteration {i + 1}/{self.config.iterations}: "
                    f"total={metric.total_time_ms:.2f}ms, "
                    f"model={metric.model_time_ms:.2f}ms, "
                    f"overhead={metric.overhead_ms:.2f}ms, "
                    f"agents={len(iteration_timelines)}"
                )
            except MultiAgentBenchmarkError as e:
                error_msg = f"Iteration {i + 1} failed: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)
        
        if not metrics:
            raise MultiAgentBenchmarkError(
                f"All {self.config.iterations} iterations failed for {model_id}"
            )
        
        # Calculate statistics
        statistics = calculate_statistics(metrics)
        
        # Use timelines from the last successful iteration for the result
        final_timelines = all_timelines[-1] if all_timelines else []
        
        # Calculate average overhead and model time
        num_successful = len(metrics)
        avg_inter_agent_overhead = total_inter_agent_overhead / num_successful
        avg_total_model_time = total_model_time_accumulated / num_successful
        
        return MultiAgentBenchmarkResult(
            model_id=model_id,
            model_name=model_name,
            provider=provider,
            region=self.config.region,
            benchmark_type="multi-agent",
            timestamp=datetime.now(),
            iterations=len(metrics),
            metrics=metrics,
            statistics=statistics,
            errors=errors,
            agent_timelines=final_timelines,
            inter_agent_overhead_ms=avg_inter_agent_overhead,
            total_model_time_ms=avg_total_model_time,
        )
