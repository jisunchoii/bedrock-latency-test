# Implementation Plan

- [x] 1. Set up project structure and core utilities
  - [x] 1.1 Create project structure with pyproject.toml and requirements.txt
    - Initialize Python project with dependencies (boto3, strands-agents, hypothesis, pytest, rich)
    - Create src/ and tests/ directories
    - _Requirements: 1.1_
  - [x] 1.2 Implement configuration module (config.py)
    - Create BenchmarkConfig dataclass with region, iterations, warmup, prompt, max_tokens, models, output_dir
    - Implement JSON serialization/deserialization
    - _Requirements: 2.7_
  - [x] 1.3 Write property test for configuration round-trip
    - **Property 6: Configuration parsing round-trip**
    - **Validates: Requirements 2.7**
  - [x] 1.4 Implement models registry (models.py)
    - Define LIGHTWEIGHT_MODELS dictionary with model IDs for Claude Haiku, Nova Micro, Nova Lite
    - Implement get_available_models() function
    - _Requirements: 1.1, 1.2, 1.3_

- [x] 2. Implement timer and statistics utilities
  - [x] 2.1 Implement timer utility (timer.py)
    - Create LatencyMetrics dataclass with total_time_ms, ttfb_ms, model_time_ms, overhead_ms
    - Implement Timer class with start(), mark_first_byte(), stop() methods
    - _Requirements: 2.1, 2.2_
  - [x] 2.2 Write property test for latency positivity
    - **Property 1: Latency measurements are positive**
    - **Validates: Requirements 2.1, 3.1, 4.1**
  - [x] 2.3 Write property test for TTFB constraint
    - **Property 2: TTFB is less than or equal to total time**
    - **Validates: Requirements 2.2**
  - [x] 2.4 Implement statistics calculator (stats.py)
    - Create BenchmarkStatistics dataclass with min, max, avg, median, p95, p99, std_dev
    - Implement calculate_statistics() function
    - _Requirements: 2.3_
  - [x] 2.5 Write property test for statistics bounds
    - **Property 3: Statistics bounds invariant**
    - **Validates: Requirements 2.3**

- [x] 3. Checkpoint - Make sure all tests are passing
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement API benchmark
  - [x] 4.1 Implement API benchmark class (api_benchmark.py)
    - Create APIBenchmark class with run_single() and run_streaming() methods
    - Implement retry logic with exponential backoff (3 retries)
    - Use boto3 bedrock-runtime client for invoke_model and invoke_model_with_response_stream
    - _Requirements: 2.1, 2.2, 2.4, 2.5_
  - [x] 4.2 Write property test for iteration count
    - **Property 4: Iteration count consistency**
    - **Validates: Requirements 2.4**
  - [x] 4.3 Implement BenchmarkResult dataclass and serialization
    - Create BenchmarkResult with model_id, region, benchmark_type, timestamp, metrics, statistics
    - Implement to_json() and from_json() methods
    - _Requirements: 2.6, 5.2_
  - [x] 4.4 Write property test for result serialization round-trip
    - **Property 5: Benchmark result serialization round-trip**
    - **Validates: Requirements 2.6**

- [x] 5. Checkpoint - Make sure all tests are passing
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement single agent benchmark
  - [x] 6.1 Implement agent benchmark class (agent_benchmark.py)
    - Create AgentBenchmark class using strands-agents SDK
    - Implement create_agent() with Bedrock model configuration
    - Track model_time, tool_time, orchestration_overhead separately
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  - [x] 6.2 Write property test for agent time breakdown
    - **Property 7: Agent time breakdown invariant**
    - **Validates: Requirements 3.2, 3.3**

- [x] 7. Implement multi-agent benchmark
  - [x] 7.1 Implement multi-agent benchmark class (multi_agent_benchmark.py)
    - Create MultiAgentBenchmark class using Agents-as-Tools pattern
    - Implement supervisor and worker agent creation
    - Track individual agent times and inter-agent overhead
    - Create AgentTimeline and MultiAgentBenchmarkResult dataclasses
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  - [x] 7.2 Write property test for multi-agent time breakdown
    - **Property 8: Multi-agent time breakdown invariant**
    - **Validates: Requirements 4.2, 4.3**

- [x] 8. Checkpoint - Make sure all tests are passing
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement report generation and CLI
  - [x] 9.1 Implement report generator (report.py)
    - Create ReportGenerator class with generate_console_report() using rich library
    - Implement generate_json_report() for file output
    - Implement compare_reports() for comparing two benchmark runs
    - _Requirements: 5.1, 5.3, 5.4_
  - [x] 9.2 Write property test for output fields
    - **Property 9: Output contains required fields**
    - **Validates: Requirements 1.3, 3.4, 4.4, 5.2**
  - [x] 9.3 Write property test for comparison consistency
    - **Property 10: Comparison report consistency**
    - **Validates: Requirements 5.3**
  - [x] 9.4 Implement CLI interface (cli.py)
    - Create main() function with argparse for command-line arguments
    - Support --models, --iterations, --type (api/agent/multi-agent), --output options
    - Implement run_all_benchmarks() orchestration function
    - _Requirements: 1.1, 2.4, 5.1, 5.4_

- [x] 10. Final Checkpoint - Make sure all tests are passing
  - Ensure all tests pass, ask the user if questions arise.
