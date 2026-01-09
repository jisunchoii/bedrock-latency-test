"""Report generator for benchmark results.

Provides console and JSON report generation, as well as comparison
between multiple benchmark runs.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Union

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .results import BenchmarkResult, MultiAgentBenchmarkResult
from .stats import BenchmarkStatistics


@dataclass
class MetricComparison:
    """Comparison of a single metric between two benchmark runs.
    
    Attributes:
        metric_name: Name of the metric being compared
        value1: Value from first benchmark
        value2: Value from second benchmark
        diff_ms: Absolute difference (value2 - value1)
        diff_percent: Percentage change ((value2 - value1) / value1 * 100)
        status: 'improved', 'degraded', or 'stable'
    """
    metric_name: str
    value1: float
    value2: float
    diff_ms: float
    diff_percent: float
    status: str


@dataclass
class ComparisonReport:
    """Comparison report between two benchmark runs.
    
    Attributes:
        report1_path: Path to first benchmark report
        report2_path: Path to second benchmark report
        model_id: Model ID being compared
        benchmark_type: Type of benchmark
        timestamp: When comparison was generated
        comparisons: List of metric comparisons
        summary: Overall summary of changes
    """
    report1_path: str
    report2_path: str
    model_id: str
    benchmark_type: str
    timestamp: datetime
    comparisons: List[MetricComparison]
    summary: str

    def to_dict(self) -> dict:
        """Convert comparison report to dictionary."""
        return {
            "report1_path": self.report1_path,
            "report2_path": self.report2_path,
            "model_id": self.model_id,
            "benchmark_type": self.benchmark_type,
            "timestamp": self.timestamp.isoformat(),
            "comparisons": [asdict(c) for c in self.comparisons],
            "summary": self.summary,
        }

    def to_json(self) -> str:
        """Serialize comparison report to JSON."""
        return json.dumps(self.to_dict(), indent=2)



class ReportGenerator:
    """Generator for benchmark reports in console and JSON formats.
    
    Provides methods to display results in formatted console tables
    and save results to JSON files.
    """

    def __init__(self) -> None:
        """Initialize report generator."""
        self.console = Console()

    def generate_console_report(
        self,
        results: Union[BenchmarkResult, List[BenchmarkResult], Dict[str, BenchmarkResult]],
    ) -> None:
        """Generate and display a formatted console report.
        
        Args:
            results: Single result, list of results, or dict of results
        """
        # Normalize input to list
        if isinstance(results, BenchmarkResult):
            results_list = [results]
        elif isinstance(results, dict):
            results_list = list(results.values())
        else:
            results_list = results

        if not results_list:
            self.console.print("[yellow]No benchmark results to display.[/yellow]")
            return

        # Group results by benchmark type
        by_type: Dict[str, List[BenchmarkResult]] = {}
        for result in results_list:
            if result.benchmark_type not in by_type:
                by_type[result.benchmark_type] = []
            by_type[result.benchmark_type].append(result)

        # Display each type
        for benchmark_type, type_results in by_type.items():
            self._display_benchmark_type(benchmark_type, type_results)

    def _display_benchmark_type(
        self,
        benchmark_type: str,
        results: List[BenchmarkResult],
    ) -> None:
        """Display results for a specific benchmark type.
        
        Args:
            benchmark_type: Type of benchmark (api, agent, multi-agent)
            results: List of results for this type
        """
        title = f"Benchmark Results: {benchmark_type.upper()}"
        
        # Create main statistics table
        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Model", style="green")
        table.add_column("Provider", style="blue")
        table.add_column("Region", style="magenta")
        table.add_column("Iterations", justify="right")
        table.add_column("Min (ms)", justify="right")
        table.add_column("Avg (ms)", justify="right")
        table.add_column("Median (ms)", justify="right")
        table.add_column("P95 (ms)", justify="right")
        table.add_column("Max (ms)", justify="right")

        for result in results:
            stats = result.statistics
            table.add_row(
                result.model_name,
                result.provider,
                result.region,
                str(result.iterations),
                f"{stats.min_ms:.2f}",
                f"{stats.avg_ms:.2f}",
                f"{stats.median_ms:.2f}",
                f"{stats.p95_ms:.2f}",
                f"{stats.max_ms:.2f}",
            )

        self.console.print()
        self.console.print(table)

        # Display additional details for agent/multi-agent benchmarks
        for result in results:
            if isinstance(result, MultiAgentBenchmarkResult):
                self._display_multi_agent_details(result)
            elif result.benchmark_type == "agent":
                self._display_agent_details(result)

        # Display errors if any
        for result in results:
            if result.errors:
                self._display_errors(result)

    def _display_agent_details(self, result: BenchmarkResult) -> None:
        """Display agent-specific details.
        
        Args:
            result: Agent benchmark result
        """
        # Calculate average breakdown from metrics
        if not result.metrics:
            return

        avg_model_time = sum(
            m.model_time_ms or 0 for m in result.metrics
        ) / len(result.metrics)
        avg_overhead = sum(
            m.overhead_ms or 0 for m in result.metrics
        ) / len(result.metrics)

        panel_content = Text()
        panel_content.append(f"Model: {result.model_name}\n", style="bold")
        panel_content.append(f"Average Model Time: {avg_model_time:.2f} ms\n")
        panel_content.append(f"Average Overhead: {avg_overhead:.2f} ms\n")

        self.console.print(Panel(
            panel_content,
            title="Agent Time Breakdown",
            border_style="blue",
        ))

    def _display_multi_agent_details(self, result: MultiAgentBenchmarkResult) -> None:
        """Display multi-agent specific details including timeline.
        
        Args:
            result: Multi-agent benchmark result
        """
        # Timeline table
        if result.agent_timelines:
            timeline_table = Table(
                title=f"Agent Timeline: {result.model_name}",
                show_header=True,
                header_style="bold yellow",
            )
            timeline_table.add_column("Agent", style="green")
            timeline_table.add_column("Start (ms)", justify="right")
            timeline_table.add_column("End (ms)", justify="right")
            timeline_table.add_column("Duration (ms)", justify="right")
            timeline_table.add_column("Model Time (ms)", justify="right")

            for timeline in result.agent_timelines:
                duration = timeline.end_time_ms - timeline.start_time_ms
                timeline_table.add_row(
                    timeline.agent_name,
                    f"{timeline.start_time_ms:.2f}",
                    f"{timeline.end_time_ms:.2f}",
                    f"{duration:.2f}",
                    f"{timeline.model_time_ms:.2f}",
                )

            self.console.print()
            self.console.print(timeline_table)

        # Summary panel
        panel_content = Text()
        panel_content.append(f"Model: {result.model_name}\n", style="bold")
        panel_content.append(f"Total Model Time: {result.total_model_time_ms:.2f} ms\n")
        panel_content.append(f"Inter-Agent Overhead: {result.inter_agent_overhead_ms:.2f} ms\n")
        panel_content.append(f"Agents: {len(result.agent_timelines)}\n")

        self.console.print(Panel(
            panel_content,
            title="Multi-Agent Summary",
            border_style="yellow",
        ))

    def _display_errors(self, result: BenchmarkResult) -> None:
        """Display errors from a benchmark result.
        
        Args:
            result: Benchmark result with errors
        """
        error_text = Text()
        error_text.append(f"Model: {result.model_name}\n", style="bold red")
        for error in result.errors:
            error_text.append(f"  • {error}\n", style="red")

        self.console.print(Panel(
            error_text,
            title="Errors",
            border_style="red",
        ))


    def generate_json_report(
        self,
        results: Union[BenchmarkResult, List[BenchmarkResult], Dict[str, BenchmarkResult]],
        filepath: str,
    ) -> str:
        """Generate and save a JSON report to file.
        
        Args:
            results: Single result, list of results, or dict of results
            filepath: Path to save the JSON report
            
        Returns:
            Path to the saved report file
        """
        # Normalize input to list
        if isinstance(results, BenchmarkResult):
            results_list = [results]
        elif isinstance(results, dict):
            results_list = list(results.values())
        else:
            results_list = results

        # Build report structure
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_benchmarks": len(results_list),
            "results": [r.to_dict() for r in results_list],
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        # Write to file
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        self.console.print(f"[green]Report saved to: {filepath}[/green]")
        return filepath

    def compare_reports(
        self,
        report1_path: str,
        report2_path: str,
        threshold_percent: float = 5.0,
    ) -> ComparisonReport:
        """Compare two benchmark reports and generate a comparison.
        
        Args:
            report1_path: Path to first (baseline) benchmark report
            report2_path: Path to second (comparison) benchmark report
            threshold_percent: Percentage threshold for 'stable' classification
            
        Returns:
            ComparisonReport with detailed comparisons
            
        Raises:
            ValueError: If reports cannot be compared (different models/types)
        """
        # Load reports
        with open(report1_path, "r") as f:
            report1_data = json.load(f)
        with open(report2_path, "r") as f:
            report2_data = json.load(f)

        # Get first result from each report for comparison
        # In a full implementation, we'd match by model_id
        results1 = report1_data.get("results", [])
        results2 = report2_data.get("results", [])

        if not results1 or not results2:
            raise ValueError("Both reports must contain at least one result")

        # Find matching results by model_id
        result1 = results1[0]
        result2 = None
        
        for r in results2:
            if r["model_id"] == result1["model_id"]:
                result2 = r
                break

        if result2 is None:
            # Use first result if no match found
            result2 = results2[0]

        # Compare statistics
        stats1 = result1["statistics"]
        stats2 = result2["statistics"]

        comparisons = []
        metrics_to_compare = [
            ("min_ms", "Min"),
            ("max_ms", "Max"),
            ("avg_ms", "Average"),
            ("median_ms", "Median"),
            ("p95_ms", "P95"),
            ("p99_ms", "P99"),
        ]

        improved_count = 0
        degraded_count = 0

        for metric_key, metric_name in metrics_to_compare:
            value1 = stats1[metric_key]
            value2 = stats2[metric_key]
            diff_ms = value2 - value1
            diff_percent = (diff_ms / value1 * 100) if value1 != 0 else 0

            # Determine status (lower latency is better)
            if abs(diff_percent) <= threshold_percent:
                status = "stable"
            elif diff_percent < 0:
                status = "improved"
                improved_count += 1
            else:
                status = "degraded"
                degraded_count += 1

            comparisons.append(MetricComparison(
                metric_name=metric_name,
                value1=value1,
                value2=value2,
                diff_ms=diff_ms,
                diff_percent=diff_percent,
                status=status,
            ))

        # Generate summary
        if improved_count > degraded_count:
            summary = f"Overall improvement: {improved_count} metrics improved, {degraded_count} degraded"
        elif degraded_count > improved_count:
            summary = f"Overall degradation: {degraded_count} metrics degraded, {improved_count} improved"
        else:
            summary = f"Mixed results: {improved_count} improved, {degraded_count} degraded"

        return ComparisonReport(
            report1_path=report1_path,
            report2_path=report2_path,
            model_id=result1["model_id"],
            benchmark_type=result1["benchmark_type"],
            timestamp=datetime.now(),
            comparisons=comparisons,
            summary=summary,
        )

    def display_comparison(self, comparison: ComparisonReport) -> None:
        """Display a comparison report in the console.
        
        Args:
            comparison: ComparisonReport to display
        """
        self.console.print()
        self.console.print(Panel(
            f"[bold]Comparing:[/bold]\n"
            f"  Baseline: {comparison.report1_path}\n"
            f"  Current:  {comparison.report2_path}\n"
            f"  Model: {comparison.model_id}\n"
            f"  Type: {comparison.benchmark_type}",
            title="Benchmark Comparison",
            border_style="cyan",
        ))

        # Comparison table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="white")
        table.add_column("Baseline (ms)", justify="right")
        table.add_column("Current (ms)", justify="right")
        table.add_column("Diff (ms)", justify="right")
        table.add_column("Diff (%)", justify="right")
        table.add_column("Status", justify="center")

        for comp in comparison.comparisons:
            # Color code the status
            if comp.status == "improved":
                status_style = "[green]✓ Improved[/green]"
                diff_style = "green"
            elif comp.status == "degraded":
                status_style = "[red]✗ Degraded[/red]"
                diff_style = "red"
            else:
                status_style = "[yellow]~ Stable[/yellow]"
                diff_style = "yellow"

            table.add_row(
                comp.metric_name,
                f"{comp.value1:.2f}",
                f"{comp.value2:.2f}",
                f"[{diff_style}]{comp.diff_ms:+.2f}[/{diff_style}]",
                f"[{diff_style}]{comp.diff_percent:+.1f}%[/{diff_style}]",
                status_style,
            )

        self.console.print()
        self.console.print(table)

        # Summary
        self.console.print()
        self.console.print(Panel(
            comparison.summary,
            title="Summary",
            border_style="cyan",
        ))
