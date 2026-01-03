# =============================================================================
# Prometheus Metrics Collection
# =============================================================================
import time
import asyncio
import functools
from typing import Callable, Optional, Any
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CollectorRegistry,
    REGISTRY,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Request Metrics
# =============================================================================

REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total number of requests",
    ["endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Request size metrics
REQUEST_SIZE = Histogram(
    "rag_request_size_bytes",
    "Request size in bytes",
    ["endpoint"],
    buckets=[100, 1000, 10000, 100000, 1000000]
)

# Response size metrics
RESPONSE_SIZE = Histogram(
    "rag_response_size_bytes",
    "Response size in bytes",
    ["endpoint"],
    buckets=[100, 1000, 10000, 100000, 1000000]
)


# =============================================================================
# RAG Pipeline Metrics
# =============================================================================

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Retrieval step latency",
    ["retrieval_type"],  # vector, hybrid, keyword
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

SYNTHESIS_LATENCY = Histogram(
    "rag_synthesis_latency_seconds",
    "Answer synthesis latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

VERIFICATION_LATENCY = Histogram(
    "rag_verification_latency_seconds",
    "Answer verification latency",
    ["verification_tier"],  # tier1, tier2
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)

CHUNKS_RETRIEVED = Histogram(
    "rag_chunks_retrieved",
    "Number of chunks retrieved per query",
    buckets=[1, 5, 10, 20, 50, 100]
)

SECTIONS_RETRIEVED = Histogram(
    "rag_sections_retrieved",
    "Number of sections retrieved per query",
    buckets=[1, 5, 10, 20, 50]
)

DOCUMENTS_RETRIEVED = Histogram(
    "rag_documents_retrieved",
    "Number of documents retrieved per query",
    buckets=[1, 5, 10, 20]
)

# RAG quality metrics
ANSWER_GROUNDEDNESS = Histogram(
    "rag_answer_groundedness_score",
    "Answer groundedness confidence score",
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
)

QUERY_COMPLEXITY = Counter(
    "rag_query_complexity_total",
    "Query complexity classification",
    ["complexity"]  # simple, medium, complex
)


# =============================================================================
# Cache Metrics
# =============================================================================

CACHE_HIT = Counter(
    "rag_cache_hits_total",
    "Cache hit count",
    ["cache_type", "level"]  # embedding, semantic, retrieval / l1, l2
)

CACHE_MISS = Counter(
    "rag_cache_misses_total",
    "Cache miss count",
    ["cache_type", "level"]
)

CACHE_EVICTION = Counter(
    "rag_cache_evictions_total",
    "Cache eviction count",
    ["cache_type", "level"]
)

CACHE_SIZE = Gauge(
    "rag_cache_size",
    "Current cache size",
    ["cache_type", "level"]
)

CACHE_LATENCY = Histogram(
    "rag_cache_latency_seconds",
    "Cache operation latency",
    ["cache_type", "operation"],  # get, set, delete
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25]
)


# =============================================================================
# System Metrics
# =============================================================================

ACTIVE_CONNECTIONS = Gauge(
    "rag_active_connections",
    "Number of active database connections",
    ["db_type"]  # postgres, qdrant, redis
)

MEMORY_USAGE = Gauge(
    "rag_memory_usage_bytes",
    "Memory usage in bytes",
    ["component"]  # api, embedder, cache
)

QUEUE_SIZE = Gauge(
    "rag_queue_size",
    "Current queue size",
    ["queue_name"]
)

THREAD_POOL_SIZE = Gauge(
    "rag_thread_pool_size",
    "Thread pool size",
    ["pool_name"]
)

THREAD_POOL_ACTIVE = Gauge(
    "rag_thread_pool_active",
    "Active threads in pool",
    ["pool_name"]
)


# =============================================================================
# Agent Metrics
# =============================================================================

AGENT_EXECUTION_LATENCY = Histogram(
    "rag_agent_latency_seconds",
    "Agent execution latency",
    ["agent_name"],  # router, planner, retriever, synthesizer, verifier, grader
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

AGENT_EXECUTION_COUNT = Counter(
    "rag_agent_executions_total",
    "Agent execution count",
    ["agent_name", "status"]  # success, failure, error
)

ORCHESTRATOR_STEPS = Histogram(
    "rag_orchestrator_steps_total",
    "Number of steps in orchestrator workflow",
    buckets=[1, 2, 3, 5, 7, 10, 15]
)


# =============================================================================
# Document Processing Metrics
# =============================================================================

DOCUMENTS_INDEXED = Counter(
    "rag_documents_indexed_total",
    "Number of documents indexed",
    ["status"]  # success, failed
)

DOCUMENT_INDEXING_LATENCY = Histogram(
    "rag_document_indexing_latency_seconds",
    "Document indexing latency",
    ["file_type"],  # md, html, pdf, docx
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0]
)

CHUNKS_GENERATED = Counter(
    "rag_chunks_generated_total",
    "Number of chunks generated",
    ["document_type"]
)

EMBEDDING_GENERATION = Histogram(
    "rag_embedding_generation_latency_seconds",
    "Embedding generation latency",
    ["embedder_type"],  # openai, local
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

EMBEDDING_BATCH_SIZE = Histogram(
    "rag_embedding_batch_size",
    "Embedding batch size",
    buckets=[1, 5, 10, 20, 50, 100]
)


# =============================================================================
# LLM Metrics
# =============================================================================

LLM_REQUEST_COUNT = Counter(
    "rag_llm_requests_total",
    "Total LLM requests",
    ["provider", "model", "status"]  # anthropic, openai / success, failure
)

LLM_REQUEST_LATENCY = Histogram(
    "rag_llm_request_latency_seconds",
    "LLM request latency",
    ["provider", "model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0]
)

LLM_TOKEN_USAGE = Histogram(
    "rag_llm_tokens_used",
    "LLM token usage per request",
    ["provider", "model", "token_type"],  # prompt, completion, total
    buckets=[100, 500, 1000, 2000, 4000, 8000, 16000]
)

LLM_TOTAL_COST = Counter(
    "rag_llm_cost_total",
    "Total LLM cost in USD",
    ["provider", "model"]
)


# =============================================================================
# Error Metrics
# =============================================================================

ERROR_COUNT = Counter(
    "rag_errors_total",
    "Total error count",
    ["error_type", "component"]  # timeout, connection, validation / api, storage, llm
)

ERROR_RATE = Gauge(
    "rag_error_rate",
    "Current error rate (errors per second)",
    ["component"]
)

RETRY_COUNT = Counter(
    "rag_retries_total",
    "Total retry count",
    ["operation", "max_attempts"]
)


# =============================================================================
# Decorators and Utilities
# =============================================================================

def track_latency(metric: Histogram, labels: Optional[dict] = None):
    """
    Decorator to track function latency in a histogram metric.

    Args:
        metric: The Histogram metric to record to
        labels: Optional dict of label names to extract from function args/kwargs

    Example:
        @track_latency(REQUEST_LATENCY, {"endpoint": "query"})
        async def process_query(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                latency = time.time() - start
                # Extract labels if provided
                label_values = {}
                if labels:
                    for label_name, label_source in labels.items():
                        if isinstance(label_source, str):
                            label_values[label_name] = label_source
                        elif callable(label_source):
                            label_values[label_name] = label_source(*args, **kwargs)

                # Record with labels
                if label_values:
                    metric.labels(**label_values).observe(latency)
                else:
                    metric.observe(latency)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                latency = time.time() - start
                if labels:
                    label_values = {}
                    for label_name, label_source in labels.items():
                        if isinstance(label_source, str):
                            label_values[label_name] = label_source
                        elif callable(label_source):
                            label_values[label_name] = label_source(*args, **kwargs)
                    metric.labels(**label_values).observe(latency)
                else:
                    metric.observe(latency)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def track_count(metric: Counter, labels: Optional[dict] = None, status: str = "success"):
    """
    Decorator to track function call count.

    Args:
        metric: The Counter metric to increment
        labels: Optional dict of label names to values
        status: Status label value (success, failure, error)

    Example:
        @track_count(AGENT_EXECUTION_COUNT, {"agent_name": "retriever"})
        async def retrieve(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                label_values = {"status": status}
                if labels:
                    label_values.update(labels)
                metric.labels(**label_values).inc()
                return result
            except Exception as e:
                label_values = {"status": "error"}
                if labels:
                    label_values.update(labels)
                metric.labels(**label_values).inc()
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                label_values = {"status": status}
                if labels:
                    label_values.update(labels)
                metric.labels(**label_values).inc()
                return result
            except Exception as e:
                label_values = {"status": "error"}
                if labels:
                    label_values.update(labels)
                metric.labels(**label_values).inc()
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


async def get_metrics() -> bytes:
    """
    Generate Prometheus metrics in text format.

    Returns:
        Metrics in Prometheus exposition format
    """
    try:
        return generate_latest(REGISTRY)
    except Exception as e:
        logger.error("Failed to generate metrics", error=str(e))
        # Return empty metrics on error
        return b""


def get_metrics_registry() -> CollectorRegistry:
    """Get the Prometheus metrics registry."""
    return REGISTRY


# =============================================================================
# Convenience Functions for Common Metrics
# =============================================================================

def track_request(endpoint: str):
    """
    Decorator to track HTTP request metrics.

    Args:
        endpoint: The endpoint name (e.g., "query", "health")

    Example:
        @track_request("query")
        async def query_endpoint(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            try:
                result = await func(*args, **kwargs)
                REQUEST_COUNT.labels(endpoint=endpoint, status="200").inc()
                return result
            except Exception as e:
                status = "error"
                REQUEST_COUNT.labels(endpoint="500", status="error").inc()
                ERROR_COUNT.labels(
                    error_type=type(e).__name__,
                    component="api"
                ).inc()
                raise
            finally:
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)

        return wrapper
    return decorator


def increment_cache_hit(cache_type: str, level: str = "l2"):
    """Increment cache hit counter."""
    CACHE_HIT.labels(cache_type=cache_type, level=level).inc()
    logger.debug("Cache hit", cache_type=cache_type, level=level)


def increment_cache_miss(cache_type: str, level: str = "l2"):
    """Increment cache miss counter."""
    CACHE_MISS.labels(cache_type=cache_type, level=level).inc()
    logger.debug("Cache miss", cache_type=cache_type, level=level)


def increment_cache_eviction(cache_type: str, level: str = "l1"):
    """Increment cache eviction counter."""
    CACHE_EVICTION.labels(cache_type=cache_type, level=level).inc()


def set_cache_size(cache_type: str, level: str, size: int):
    """Set current cache size gauge."""
    CACHE_SIZE.labels(cache_type=cache_type, level=level).set(size)


def set_active_connections(db_type: str, count: int):
    """Set active database connections gauge."""
    ACTIVE_CONNECTIONS.labels(db_type=db_type).set(count)


def increment_llm_request(provider: str, model: str, status: str = "success"):
    """Increment LLM request counter."""
    LLM_REQUEST_COUNT.labels(provider=provider, model=model, status=status).inc()


def observe_llm_latency(provider: str, model: str, latency: float):
    """Record LLM request latency."""
    LLM_REQUEST_LATENCY.labels(provider=provider, model=model).observe(latency)


def observe_llm_tokens(provider: str, model: str, token_type: str, count: int):
    """Record LLM token usage."""
    LLM_TOKEN_USAGE.labels(provider=provider, model=model, token_type=token_type).observe(count)


def increment_agent_execution(agent_name: str, status: str = "success"):
    """Increment agent execution counter."""
    AGENT_EXECUTION_COUNT.labels(agent_name=agent_name, status=status).inc()


def observe_agent_latency(agent_name: str, latency: float):
    """Record agent execution latency."""
    AGENT_EXECUTION_LATENCY.labels(agent_name=agent_name).observe(latency)


def observe_retrieval_latency(retrieval_type: str, latency: float):
    """Record retrieval latency."""
    RETRIEVAL_LATENCY.labels(retrieval_type=retrieval_type).observe(latency)


def observe_synthesis_latency(latency: float):
    """Record synthesis latency."""
    SYNTHESIS_LATENCY.observe(latency)


def observe_verification_latency(tier: str, latency: float):
    """Record verification latency."""
    VERIFICATION_LATENCY.labels(verification_tier=tier).observe(latency)


def observe_chunks_retrieved(count: int):
    """Record number of chunks retrieved."""
    CHUNKS_RETRIEVED.observe(count)


def observe_groundedness_score(score: float):
    """Record answer groundedness score."""
    ANSWER_GROUNDEDNESS.observe(score)
