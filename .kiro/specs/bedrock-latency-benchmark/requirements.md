# Requirements Document

## Introduction

이 문서는 AWS Bedrock 경량 모델의 latency를 측정하는 벤치마크 스크립트에 대한 요구사항을 정의합니다. 서울 리전(ap-northeast-2)에서 사용 가능한 경량 모델을 대상으로 API 직접 호출, Strands Agents 단일 에이전트, 멀티 에이전트 패턴에서의 latency를 측정합니다.

## Glossary

- **Bedrock**: AWS의 완전 관리형 생성형 AI 서비스
- **경량 모델 (Lightweight Model)**: 빠른 응답 속도와 낮은 비용을 제공하는 소형 모델 (Haiku, Nova Lite, Nova Micro 등)
- **Latency**: API 요청부터 응답 완료까지의 시간
- **TTFB (Time To First Byte)**: 첫 번째 응답 바이트까지의 시간
- **Strands Agents**: AWS의 오픈소스 에이전트 프레임워크
- **멀티 에이전트 패턴**: 여러 에이전트가 협력하여 작업을 수행하는 아키텍처

## 서울 리전 (ap-northeast-2) 사용 가능 경량 모델

| 모델 | Provider | Model ID | 특징 |
|------|----------|----------|------|
| Claude 3 Haiku | Anthropic | anthropic.claude-3-haiku-20240307-v1:0 | 빠른 응답, 저비용 |
| Claude 3.5 Haiku | Anthropic | anthropic.claude-3-5-haiku-20241022-v1:0 | Haiku 업그레이드 버전 |
| Amazon Nova Micro | Amazon | amazon.nova-micro-v1:0 | 텍스트 전용, 최저 latency |
| Amazon Nova Lite | Amazon | amazon.nova-lite-v1:0 | 멀티모달, 저비용 |

## Requirements

### Requirement 1

**User Story:** As a 개발자, I want to 서울 리전에서 사용 가능한 Bedrock 경량 모델 목록을 확인하고 싶다, so that 벤치마크 대상 모델을 선정할 수 있다.

#### Acceptance Criteria

1. WHEN the script initializes THEN the Latency_Benchmark_System SHALL retrieve and display the list of available lightweight models in ap-northeast-2 region
2. WHEN a model is not available in the region THEN the Latency_Benchmark_System SHALL skip that model and log a warning message
3. WHEN displaying model information THEN the Latency_Benchmark_System SHALL show model ID, provider name, and availability status

### Requirement 2

**User Story:** As a 개발자, I want to Bedrock API를 직접 호출하여 latency를 측정하고 싶다, so that 기본 모델 응답 시간을 파악할 수 있다.

#### Acceptance Criteria

1. WHEN measuring API latency THEN the Latency_Benchmark_System SHALL record total response time in milliseconds
2. WHEN measuring API latency THEN the Latency_Benchmark_System SHALL record TTFB (Time To First Byte) for streaming responses
3. WHEN a benchmark run completes THEN the Latency_Benchmark_System SHALL calculate and display min, max, average, and p95 latency values
4. WHEN running benchmarks THEN the Latency_Benchmark_System SHALL execute a configurable number of iterations (default: 10)
5. WHEN an API call fails THEN the Latency_Benchmark_System SHALL retry up to 3 times with exponential backoff and log the failure
6. WHEN serializing benchmark results THEN the Latency_Benchmark_System SHALL encode them using JSON format
7. WHEN parsing benchmark configuration THEN the Latency_Benchmark_System SHALL validate it against the expected schema

### Requirement 3

**User Story:** As a 개발자, I want to Strands Agents를 사용하여 에이전트 내에서 모델 호출 latency를 측정하고 싶다, so that 에이전트 오버헤드를 포함한 실제 응답 시간을 파악할 수 있다.

#### Acceptance Criteria

1. WHEN measuring agent latency THEN the Latency_Benchmark_System SHALL record the time from agent invocation to response completion
2. WHEN measuring agent latency THEN the Latency_Benchmark_System SHALL separately track model inference time and agent orchestration overhead
3. WHEN an agent uses tools THEN the Latency_Benchmark_System SHALL record tool execution time separately from model latency
4. WHEN displaying agent benchmark results THEN the Latency_Benchmark_System SHALL show breakdown of model time, tool time, and orchestration overhead

### Requirement 4

**User Story:** As a 개발자, I want to 멀티 에이전트 패턴에서 모델 호출 latency를 측정하고 싶다, so that 복잡한 에이전트 협업 시나리오의 성능을 파악할 수 있다.

#### Acceptance Criteria

1. WHEN measuring multi-agent latency THEN the Latency_Benchmark_System SHALL record total end-to-end response time
2. WHEN measuring multi-agent latency THEN the Latency_Benchmark_System SHALL track individual agent response times within the collaboration
3. WHEN agents communicate THEN the Latency_Benchmark_System SHALL record inter-agent communication overhead
4. WHEN displaying multi-agent results THEN the Latency_Benchmark_System SHALL show a timeline breakdown of each agent's contribution

### Requirement 5

**User Story:** As a 개발자, I want to 벤치마크 결과를 저장하고 비교하고 싶다, so that 시간에 따른 성능 변화를 추적할 수 있다.

#### Acceptance Criteria

1. WHEN a benchmark completes THEN the Latency_Benchmark_System SHALL save results to a JSON file with timestamp
2. WHEN saving results THEN the Latency_Benchmark_System SHALL include model ID, region, test type, and all latency metrics
3. WHEN requested THEN the Latency_Benchmark_System SHALL generate a comparison report between multiple benchmark runs
4. WHEN generating reports THEN the Latency_Benchmark_System SHALL output results in both console table format and JSON format
