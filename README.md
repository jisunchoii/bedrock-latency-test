# Bedrock Latency Benchmark

AWS Bedrock 경량 모델의 latency를 측정하는 벤치마크 도구입니다.

## 설치

```bash
# 의존성 설치
pip install -r requirements.txt
```

## 사용 가능한 모델

서울 리전(ap-northeast-2)에서 사용 가능한 경량 모델:

| 모델명 | Provider | 설명 |
|--------|----------|------|
| claude-3-haiku | Anthropic | 빠른 응답, 저비용 |
| claude-3.5-haiku | Anthropic | Haiku 업그레이드 버전 |
| nova-micro | Amazon | 텍스트 전용, 최저 latency |
| nova-lite | Amazon | 멀티모달, 저비용 |

## 실행 방법

### 모델 목록 확인

```bash
python -m src.cli --list-models
```

### API 벤치마크 실행

```bash
# 모든 모델에 대해 API 벤치마크 실행
python -m src.cli --type api

# 특정 모델만 벤치마크
python -m src.cli --type api --models claude-3-haiku nova-micro

# 스트리밍 API 벤치마크
python -m src.cli --type api-streaming
```

### 에이전트 벤치마크 실행

```bash
# 단일 에이전트 벤치마크
python -m src.cli --type agent

# 멀티 에이전트 벤치마크
python -m src.cli --type multi-agent
```

### 여러 벤치마크 타입 동시 실행

```bash
python -m src.cli --type api agent multi-agent
```

### 옵션 설정

```bash
# 반복 횟수 설정 (기본값: 10)
python -m src.cli --type api --iterations 20

# 워밍업 횟수 설정 (기본값: 2)
python -m src.cli --type api --warmup 3

# 결과 파일 경로 지정
python -m src.cli --type api --output ./results/my_benchmark.json

# 리전 변경
python -m src.cli --type api --region us-east-1

# 프롬프트 변경
python -m src.cli --type api --prompt "안녕하세요, 오늘 날씨가 어때요?"

# 최대 토큰 수 설정
python -m src.cli --type api --max-tokens 200

# 상세 로그 출력
python -m src.cli --type api --verbose
```

### 벤치마크 결과 비교

```bash
python -m src.cli --compare report1.json report2.json
```

## 전체 옵션 목록

| 옵션 | 단축 | 설명 | 기본값 |
|------|------|------|--------|
| `--models` | `-m` | 벤치마크할 모델 이름 | 전체 모델 |
| `--type` | `-t` | 벤치마크 타입 (api, api-streaming, agent, multi-agent) | api |
| `--iterations` | `-i` | 벤치마크 반복 횟수 | 10 |
| `--warmup` | `-w` | 워밍업 반복 횟수 | 2 |
| `--output` | `-o` | JSON 결과 파일 경로 | 자동 생성 |
| `--output-dir` | | 결과 저장 디렉토리 | ./benchmark_results |
| `--region` | `-r` | AWS 리전 | ap-northeast-2 |
| `--prompt` | `-p` | 테스트 프롬프트 | "Hello, how are you?" |
| `--max-tokens` | | 최대 응답 토큰 수 | 100 |
| `--compare` | `-c` | 두 벤치마크 결과 비교 | - |
| `--list-models` | `-l` | 사용 가능한 모델 목록 출력 | - |
| `--verbose` | `-v` | 상세 로그 출력 | - |

## 출력 예시

벤치마크 실행 후 콘솔에 다음과 같은 결과가 표시됩니다:

```
Benchmark Results: API
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ Model          ┃ Provider ┃ Region         ┃ Iterations ┃ Min (ms) ┃ Avg (ms) ┃ Median (ms)┃ P95 (ms) ┃ Max (ms) ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ claude-3-haiku │ Anthropic│ ap-northeast-2 │         10 │   150.23 │   180.45 │     175.32 │   210.56 │   225.89 │
│ nova-micro     │ Amazon   │ ap-northeast-2 │         10 │    95.12 │   120.34 │     115.67 │   145.23 │   160.45 │
└────────────────┴──────────┴────────────────┴────────────┴──────────┴──────────┴────────────┴──────────┴──────────┘
```

## AWS 자격 증명

벤치마크를 실행하려면 AWS 자격 증명이 필요합니다:

```bash
# 환경 변수 설정
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=ap-northeast-2

# 또는 AWS CLI 프로필 사용
export AWS_PROFILE=your_profile
```

## 결과 파일

벤치마크 결과는 `./benchmark_results/` 디렉토리에 JSON 형식으로 저장됩니다.
