"""CLI interface for Bedrock Latency Benchmark.

Provides command-line interface for running benchmarks and generating reports.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List, Optional, Dict

from rich.console import Console
from rich.logging import RichHandler

from .config import BenchmarkConfig
from .models import (
    get_available_models,
    get_model_id,
    get_model_names,
    display_models,
    LIGHTWEIGHT_MODELS,
)
from .api_benchmark import APIBenchmark, APIBenchmarkError
from .agent_benchmark import AgentBenchmark, AgentBenchmarkError
from .multi_agent_benchmark import MultiAgentBenchmark, MultiAgentBenchmarkError
from .results import BenchmarkResult, MultiAgentBenchmarkResult
from .report import ReportGenerator

console = Console()
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler.
    
    Args:
        verbose: Enable debug logging if True
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def run_api_benchmark(
    config: BenchmarkConfig,
    model_ids: List[str],
    streaming: bool = False,
) -> List[BenchmarkResult]:
    """Run API benchmarks for specified models.
    
    Args:
        config: Benchmark configuration
        model_ids: List of model IDs to benchmark
        streaming: Use streaming API if True
        
    Returns:
        List of benchmark results
    """
    results = []
    benchmark = APIBenchmark(config)
    
    for model_id in model_ids:
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
        
        console.print(f"[cyan]Running API benchmark for {model_name}...[/cyan]")
        
        try:
            # Run warmup
            benchmark.run_warmup(model_id, streaming=streaming)
            
            # Run benchmark iterations
            from .timer import LatencyMetrics
            from .stats import calculate_statistics
            
            metrics: List[LatencyMetrics] = []
            errors: List[str] = []
            
            for i in range(config.iterations):
                try:
                    if streaming:
                        metric = benchmark.run_streaming(model_id)
                    else:
                        metric = benchmark.run_single(model_id)
                    metrics.append(metric)
                    logger.debug(f"Iteration {i + 1}: {metric.total_time_ms:.2f}ms")
                except APIBenchmarkError as e:
                    error_msg = f"Iteration {i + 1} failed: {e}"
                    errors.append(error_msg)
                    logger.warning(error_msg)
            
            if metrics:
                statistics = calculate_statistics(metrics)
                result = BenchmarkResult(
                    model_id=model_id,
                    model_name=model_name,
                    provider=provider,
                    region=config.region,
                    benchmark_type="api-streaming" if streaming else "api",
                    timestamp=datetime.now(),
                    iterations=len(metrics),
                    metrics=metrics,
                    statistics=statistics,
                    errors=errors,
                )
                results.append(result)
                console.print(f"[green]✓ Completed {model_name}[/green]")
            else:
                console.print(f"[red]✗ All iterations failed for {model_name}[/red]")
                
        except Exception as e:
            console.print(f"[red]✗ Failed to benchmark {model_name}: {e}[/red]")
    
    return results



def run_agent_benchmark(
    config: BenchmarkConfig,
    model_ids: List[str],
) -> List[BenchmarkResult]:
    """Run agent benchmarks for specified models.
    
    Args:
        config: Benchmark configuration
        model_ids: List of model IDs to benchmark
        
    Returns:
        List of benchmark results
    """
    results = []
    benchmark = AgentBenchmark(config)
    
    for model_id in model_ids:
        # Find model name
        model_name = None
        for name, info in LIGHTWEIGHT_MODELS.items():
            if info["model_id"] == model_id:
                model_name = name
                break
        
        if model_name is None:
            model_name = model_id.split(".")[-1]
        
        console.print(f"[cyan]Running agent benchmark for {model_name}...[/cyan]")
        
        try:
            result = benchmark.run_benchmark(model_id)
            results.append(result)
            console.print(f"[green]✓ Completed {model_name}[/green]")
        except AgentBenchmarkError as e:
            console.print(f"[red]✗ Failed to benchmark {model_name}: {e}[/red]")
    
    return results


def run_multi_agent_benchmark(
    config: BenchmarkConfig,
    model_ids: List[str],
) -> List[MultiAgentBenchmarkResult]:
    """Run multi-agent benchmarks for specified models.
    
    Args:
        config: Benchmark configuration
        model_ids: List of model IDs to benchmark
        
    Returns:
        List of multi-agent benchmark results
    """
    results = []
    benchmark = MultiAgentBenchmark(config)
    
    for model_id in model_ids:
        # Find model name
        model_name = None
        for name, info in LIGHTWEIGHT_MODELS.items():
            if info["model_id"] == model_id:
                model_name = name
                break
        
        if model_name is None:
            model_name = model_id.split(".")[-1]
        
        console.print(f"[cyan]Running multi-agent benchmark for {model_name}...[/cyan]")
        
        try:
            result = benchmark.run_benchmark(model_id)
            results.append(result)
            console.print(f"[green]✓ Completed {model_name}[/green]")
        except MultiAgentBenchmarkError as e:
            console.print(f"[red]✗ Failed to benchmark {model_name}: {e}[/red]")
    
    return results


def run_all_benchmarks(
    config: BenchmarkConfig,
    model_ids: List[str],
    benchmark_types: List[str],
) -> Dict[str, List[BenchmarkResult]]:
    """Run all specified benchmark types for all models.
    
    Args:
        config: Benchmark configuration
        model_ids: List of model IDs to benchmark
        benchmark_types: List of benchmark types to run
        
    Returns:
        Dictionary mapping benchmark type to list of results
    """
    all_results: Dict[str, List[BenchmarkResult]] = {}
    
    for benchmark_type in benchmark_types:
        console.print()
        console.print(f"[bold blue]═══ Running {benchmark_type.upper()} Benchmarks ═══[/bold blue]")
        console.print()
        
        if benchmark_type == "api":
            results = run_api_benchmark(config, model_ids, streaming=False)
            all_results["api"] = results
        elif benchmark_type == "api-streaming":
            results = run_api_benchmark(config, model_ids, streaming=True)
            all_results["api-streaming"] = results
        elif benchmark_type == "agent":
            results = run_agent_benchmark(config, model_ids)
            all_results["agent"] = results
        elif benchmark_type == "multi-agent":
            results = run_multi_agent_benchmark(config, model_ids)
            all_results["multi-agent"] = results
        else:
            console.print(f"[yellow]Unknown benchmark type: {benchmark_type}[/yellow]")
    
    return all_results



def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="bedrock-benchmark",
        description="Benchmark latency for AWS Bedrock lightweight models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run API benchmark for all models
  python -m src.cli --type api

  # Run agent benchmark for specific models
  python -m src.cli --type agent --models claude-3-haiku nova-micro

  # Run all benchmark types with custom iterations
  python -m src.cli --type api agent multi-agent --iterations 20

  # Compare two benchmark reports
  python -m src.cli --compare report1.json report2.json
        """,
    )
    
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        help="Model names to benchmark (e.g., claude-3-haiku nova-micro). "
             "If not specified, all available models are used.",
    )
    
    parser.add_argument(
        "--type", "-t",
        nargs="+",
        choices=["api", "api-streaming", "agent", "multi-agent"],
        default=["api"],
        help="Benchmark type(s) to run (default: api)",
    )
    
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=10,
        help="Number of benchmark iterations (default: 10)",
    )
    
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=2,
        help="Number of warmup iterations (default: 2)",
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file path for JSON report",
    )
    
    parser.add_argument(
        "--output-dir",
        default="./benchmark_results",
        help="Output directory for benchmark results (default: ./benchmark_results)",
    )
    
    parser.add_argument(
        "--region", "-r",
        default="ap-northeast-2",
        help="AWS region (default: ap-northeast-2)",
    )
    
    parser.add_argument(
        "--prompt", "-p",
        default="Hello, how are you?",
        help="Prompt to use for benchmarks",
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens in response (default: 100)",
    )
    
    parser.add_argument(
        "--compare", "-c",
        nargs=2,
        metavar=("REPORT1", "REPORT2"),
        help="Compare two benchmark reports",
    )
    
    parser.add_argument(
        "--list-models", "-l",
        action="store_true",
        help="List available models and exit",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.
    
    Args:
        args: Command line arguments (uses sys.argv if None)
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Setup logging
    setup_logging(parsed_args.verbose)
    
    # Handle list models
    if parsed_args.list_models:
        console.print(display_models())
        return 0
    
    # Handle comparison
    if parsed_args.compare:
        report_generator = ReportGenerator()
        try:
            comparison = report_generator.compare_reports(
                parsed_args.compare[0],
                parsed_args.compare[1],
            )
            report_generator.display_comparison(comparison)
            return 0
        except Exception as e:
            console.print(f"[red]Error comparing reports: {e}[/red]")
            return 1
    
    # Resolve model IDs
    if parsed_args.models:
        model_ids = []
        for model_name in parsed_args.models:
            model_id = get_model_id(model_name)
            if model_id:
                model_ids.append(model_id)
            else:
                console.print(f"[yellow]Warning: Unknown model '{model_name}', skipping[/yellow]")
        
        if not model_ids:
            console.print("[red]Error: No valid models specified[/red]")
            return 1
    else:
        # Use all available models
        model_ids = [info["model_id"] for info in get_available_models().values()]
    
    # Create configuration
    config = BenchmarkConfig(
        region=parsed_args.region,
        iterations=parsed_args.iterations,
        warmup_iterations=parsed_args.warmup,
        prompt=parsed_args.prompt,
        max_tokens=parsed_args.max_tokens,
        models=model_ids,
        output_dir=parsed_args.output_dir,
    )
    
    # Display configuration
    console.print()
    console.print("[bold]Benchmark Configuration:[/bold]")
    console.print(f"  Region: {config.region}")
    console.print(f"  Iterations: {config.iterations}")
    console.print(f"  Warmup: {config.warmup_iterations}")
    console.print(f"  Models: {len(model_ids)}")
    console.print(f"  Types: {', '.join(parsed_args.type)}")
    console.print()
    
    # Run benchmarks
    try:
        all_results = run_all_benchmarks(config, model_ids, parsed_args.type)
    except Exception as e:
        console.print(f"[red]Error running benchmarks: {e}[/red]")
        return 1
    
    # Flatten results for reporting
    flat_results = []
    for results in all_results.values():
        flat_results.extend(results)
    
    if not flat_results:
        console.print("[yellow]No benchmark results to report[/yellow]")
        return 1
    
    # Generate reports
    report_generator = ReportGenerator()
    
    # Console report
    console.print()
    console.print("[bold blue]═══ Benchmark Results ═══[/bold blue]")
    report_generator.generate_console_report(flat_results)
    
    # JSON report
    if parsed_args.output:
        output_path = parsed_args.output
    else:
        # Generate default output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(config.output_dir, exist_ok=True)
        output_path = os.path.join(config.output_dir, f"benchmark_{timestamp}.json")
    
    report_generator.generate_json_report(flat_results, output_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
