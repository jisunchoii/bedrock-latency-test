"""API Benchmark module for direct Bedrock API latency measurement.

Provides benchmarking for direct boto3 calls to Bedrock Runtime API,
supporting both single (invoke_model) and streaming (invoke_model_with_response_stream) modes.
"""

import json
import logging
import random
import time
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from .config import BenchmarkConfig
from .timer import Timer, LatencyMetrics

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0


class APIBenchmarkError(Exception):
    """Exception raised for API benchmark errors."""
    pass


class APIBenchmark:
    """Benchmark class for direct Bedrock API calls.
    
    Measures latency for both synchronous and streaming API calls
    to Bedrock Runtime.
    
    Attributes:
        config: Benchmark configuration
        client: Boto3 Bedrock Runtime client
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize API benchmark.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=config.region
        )

    def _build_request_body(self, model_id: str) -> str:
        """Build the request body for a model invocation.
        
        Args:
            model_id: The model ID to invoke
            
        Returns:
            JSON string request body
        """
        # Handle different model formats
        if "anthropic.claude" in model_id:
            # Direct Anthropic models use Messages API format
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.config.max_tokens,
                "messages": [
                    {"role": "user", "content": self.config.prompt}
                ]
            }
        elif "global.anthropic.claude" in model_id:
            # Global inference profile Claude models use legacy format
            body = {
                "prompt": f"\n\nHuman: {self.config.prompt}\n\nAssistant:",
                "max_tokens_to_sample": self.config.max_tokens
            }
        elif "amazon.nova" in model_id or "apac.amazon.nova" in model_id:
            # Nova models use their own format
            body = {
                "inferenceConfig": {
                    "max_new_tokens": self.config.max_tokens
                },
                "messages": [
                    {"role": "user", "content": [{"text": self.config.prompt}]}
                ]
            }
        else:
            # Generic fallback
            body = {
                "prompt": self.config.prompt,
                "max_tokens": self.config.max_tokens
            }
        
        return json.dumps(body)

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter.
        
        Args:
            attempt: Current retry attempt (0-indexed)
            
        Returns:
            Backoff time in seconds
        """
        backoff = min(
            BASE_BACKOFF_SECONDS * (2 ** attempt),
            MAX_BACKOFF_SECONDS
        )
        # Add jitter (0-25% of backoff)
        jitter = random.uniform(0, backoff * 0.25)
        return backoff + jitter

    def _should_retry(self, error: ClientError) -> bool:
        """Determine if an error should trigger a retry.
        
        Args:
            error: The ClientError from boto3
            
        Returns:
            True if the error is retryable
        """
        error_code = error.response.get("Error", {}).get("Code", "")
        retryable_codes = [
            "ThrottlingException",
            "ServiceUnavailableException",
            "InternalServerException",
            "ModelStreamErrorException",
        ]
        return error_code in retryable_codes

    def run_single(self, model_id: str) -> LatencyMetrics:
        """Run a single synchronous API call and measure latency.
        
        Args:
            model_id: The model ID to invoke
            
        Returns:
            LatencyMetrics with timing data
            
        Raises:
            APIBenchmarkError: If all retries fail
        """
        body = self._build_request_body(model_id)
        last_error: Optional[Exception] = None
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                timer = Timer()
                timer.start()
                
                self.client.invoke_model(
                    modelId=model_id,
                    body=body,
                    contentType="application/json",
                    accept="application/json"
                )
                
                return timer.stop()
                
            except ClientError as e:
                last_error = e
                if attempt < MAX_RETRIES and self._should_retry(e):
                    backoff = self._calculate_backoff(attempt)
                    logger.warning(
                        f"API call failed (attempt {attempt + 1}/{MAX_RETRIES + 1}), "
                        f"retrying in {backoff:.2f}s: {e}"
                    )
                    time.sleep(backoff)
                else:
                    break
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error during API call: {e}")
                break
        
        raise APIBenchmarkError(
            f"API call failed after {MAX_RETRIES + 1} attempts: {last_error}"
        )


    def run_streaming(self, model_id: str) -> LatencyMetrics:
        """Run a streaming API call and measure latency including TTFB.
        
        Args:
            model_id: The model ID to invoke
            
        Returns:
            LatencyMetrics with timing data including TTFB
            
        Raises:
            APIBenchmarkError: If all retries fail
        """
        body = self._build_request_body(model_id)
        last_error: Optional[Exception] = None
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                timer = Timer()
                timer.start()
                
                response = self.client.invoke_model_with_response_stream(
                    modelId=model_id,
                    body=body,
                    contentType="application/json",
                    accept="application/json"
                )
                
                # Process the stream to measure TTFB and total time
                stream = response.get("body")
                first_chunk = True
                
                for event in stream:
                    if first_chunk:
                        timer.mark_first_byte()
                        first_chunk = False
                    # Continue consuming the stream
                
                return timer.stop()
                
            except ClientError as e:
                last_error = e
                if attempt < MAX_RETRIES and self._should_retry(e):
                    backoff = self._calculate_backoff(attempt)
                    logger.warning(
                        f"Streaming API call failed (attempt {attempt + 1}/{MAX_RETRIES + 1}), "
                        f"retrying in {backoff:.2f}s: {e}"
                    )
                    time.sleep(backoff)
                else:
                    break
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error during streaming API call: {e}")
                break
        
        raise APIBenchmarkError(
            f"Streaming API call failed after {MAX_RETRIES + 1} attempts: {last_error}"
        )

    def run_warmup(self, model_id: str, streaming: bool = False) -> None:
        """Run warmup iterations to prime the connection.
        
        Args:
            model_id: The model ID to invoke
            streaming: Whether to use streaming API
        """
        logger.info(
            f"Running {self.config.warmup_iterations} warmup iterations for {model_id}"
        )
        
        for i in range(self.config.warmup_iterations):
            try:
                if streaming:
                    self.run_streaming(model_id)
                else:
                    self.run_single(model_id)
                logger.debug(f"Warmup iteration {i + 1} completed")
            except APIBenchmarkError as e:
                logger.warning(f"Warmup iteration {i + 1} failed: {e}")
