"""
Load testing suite for EpisodicMemory production validation.

Targets:
- Ingestion: 100 episodes/second
- Search: 500 queries/second, P99 < 50ms
- Gating: 200 requests/second

Run with: pytest tests/load/test_load.py -v --log-cli-level=INFO
"""

import asyncio
import os
import random
import statistics
import time
from dataclasses import dataclass

import pytest

from src.models.episode import EpisodeCreate
from src.models.gating import ReflectRequest
from src.models.search import SearchRequest

pytestmark = pytest.mark.load


def _strict_mode_enabled() -> bool:
    value = os.getenv("LOAD_TEST_STRICT", "")
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class LoadTestMetrics:
    """Metrics collected during load test."""

    total_requests: int
    successful: int
    failed: int
    latencies_ms: list[float]
    throughput_rps: float
    duration_seconds: float

    @property
    def p50_ms(self) -> float:
        if not self.latencies_ms:
            return float("nan")
        return statistics.median(self.latencies_ms)

    @property
    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return float("nan")
        if len(self.latencies_ms) < 20:
            return max(self.latencies_ms)  # Return max as fallback
        return statistics.quantiles(self.latencies_ms, n=20)[18]

    @property
    def p99_ms(self) -> float:
        if not self.latencies_ms:
            return float("nan")
        if len(self.latencies_ms) < 100:
            return max(self.latencies_ms)  # Return max as fallback
        return statistics.quantiles(self.latencies_ms, n=100)[98]

    @property
    def success_rate(self) -> float:
        return (self.successful / self.total_requests * 100) if self.total_requests > 0 else 0

    def summary(self) -> str:
        return f"""
Load Test Results:
==================
Total Requests: {self.total_requests}
Successful: {self.successful} ({self.success_rate:.1f}%)
Failed: {self.failed}
Duration: {self.duration_seconds:.2f}s
Throughput: {self.throughput_rps:.1f} req/s

Latency:
  P50: {self.p50_ms:.1f}ms
  P95: {self.p95_ms:.1f}ms
  P99: {self.p99_ms:.1f}ms
"""


class LoadTester:
    """Base class for load testing operations."""

    async def _execute_concurrent_requests(
        self,
        request_func,
        num_requests: int,
        concurrency: int,
    ) -> LoadTestMetrics:
        """
        Execute requests concurrently and collect metrics.

        Args:
            request_func: Async function to call for each request
            num_requests: Total number of requests to make
            concurrency: Number of concurrent requests
        """
        latencies = []
        successful = 0
        failed = 0

        start_time = time.perf_counter()

        # Create batches of concurrent requests
        for batch_start in range(0, num_requests, concurrency):
            batch_size = min(concurrency, num_requests - batch_start)
            tasks = [request_func() for _ in range(batch_size)]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    failed += 1
                else:
                    latency_ms = result
                    latencies.append(latency_ms)
                    successful += 1

        end_time = time.perf_counter()
        duration = end_time - start_time

        return LoadTestMetrics(
            total_requests=num_requests,
            successful=successful,
            failed=failed,
            latencies_ms=latencies,
            throughput_rps=num_requests / duration,
            duration_seconds=duration,
        )


@pytest.mark.load
@pytest.mark.asyncio
class TestIngestionLoad(LoadTester):
    """Load tests for episode ingestion."""

    async def test_ingestion_100_eps(
        self,
        ingestion_pipeline,
        sample_episode_create: EpisodeCreate,
    ):
        """
        Test ingestion at 100 episodes/second.

        Target: 1000 episodes in ~10 seconds
        """
        num_episodes = 1000
        concurrency = 100  # 100 concurrent requests

        async def ingest_episode():
            start = time.perf_counter()
            episode = sample_episode_create.model_copy(deep=True)
            await ingestion_pipeline.capture_episode(
                episode_data=episode, generate_reflection=False  # Disable for load test
            )
            return (time.perf_counter() - start) * 1000

        metrics = await self._execute_concurrent_requests(
            request_func=ingest_episode, num_requests=num_episodes, concurrency=concurrency
        )

        print(metrics.summary())

        # Assertions
        assert metrics.success_rate >= 99.0, f"Success rate {metrics.success_rate}% < 99%"
        if _strict_mode_enabled():
            assert metrics.throughput_rps >= 90, f"Throughput {metrics.throughput_rps} < 90 eps/s"
            assert metrics.p99_ms < 1000, f"P99 latency {metrics.p99_ms}ms >= 1000ms"


@pytest.mark.load
@pytest.mark.asyncio
class TestSearchLoad(LoadTester):
    """Load tests for search operations."""

    async def test_search_500_qps(
        self,
        search_pipeline,
        load_customer_id: str,
    ):
        """
        Test search at 500 queries/second with P99 < 50ms.

        Target: 5000 queries in ~10 seconds, P99 < 50ms
        """
        num_queries = 5000
        concurrency = 50  # 50 concurrent requests
        customer_id = load_customer_id

        search_queries = [
            "deploy kubernetes production",
            "docker build failed",
            "database connection timeout",
            "api rate limit exceeded",
            "authentication error 401",
        ]

        async def execute_search():
            query = random.choice(search_queries)

            start = time.perf_counter()

            request = SearchRequest(customer_id=customer_id, goal=query, k=5, min_similarity=0.6)

            await search_pipeline.search(request)
            return (time.perf_counter() - start) * 1000

        metrics = await self._execute_concurrent_requests(
            request_func=execute_search, num_requests=num_queries, concurrency=concurrency
        )

        print(metrics.summary())

        # Assertions
        assert metrics.success_rate >= 99.5, f"Success rate {metrics.success_rate}% < 99.5%"
        if _strict_mode_enabled():
            assert metrics.throughput_rps >= 450, f"Throughput {metrics.throughput_rps} < 450 qps"
            assert metrics.p99_ms < 50, f"P99 latency {metrics.p99_ms}ms >= 50ms (CRITICAL)"
            assert metrics.p95_ms < 30, f"P95 latency {metrics.p95_ms}ms >= 30ms"


@pytest.mark.load
@pytest.mark.asyncio
class TestGatingLoad(LoadTester):
    """Load tests for pre-action gating."""

    async def test_gating_200_rps(
        self,
        gating_service,
        load_customer_id: str,
    ):
        """
        Test gating at 200 requests/second.

        Target: 2000 requests in ~10 seconds
        """
        num_requests = 2000
        concurrency = 40  # 40 concurrent requests
        customer_id = load_customer_id

        actions = [
            ("kubectl delete namespace production", "kubectl"),
            ("rm -rf /var/data", "bash"),
            ("DROP DATABASE users", "psql"),
            ("docker system prune -a --volumes", "docker"),
        ]

        async def execute_gating():
            action, tool = random.choice(actions)

            start = time.perf_counter()

            request = ReflectRequest(
                goal="Execute maintenance task",
                proposed_action=action,
                tool=tool,
                current_state={"env": "production"},
            )

            await gating_service.reflect_before_action(request, customer_id)
            return (time.perf_counter() - start) * 1000

        metrics = await self._execute_concurrent_requests(
            request_func=execute_gating, num_requests=num_requests, concurrency=concurrency
        )

        print(metrics.summary())

        # Assertions
        assert metrics.success_rate >= 99.0, f"Success rate {metrics.success_rate}% < 99%"
        if _strict_mode_enabled():
            assert metrics.throughput_rps >= 180, f"Throughput {metrics.throughput_rps} < 180 rps"
            assert metrics.p99_ms < 100, f"P99 latency {metrics.p99_ms}ms >= 100ms"


@pytest.mark.load
@pytest.mark.asyncio
class TestEndToEndLoad(LoadTester):
    """End-to-end load test simulating realistic usage."""

    async def test_realistic_workload(
        self,
        ingestion_pipeline,
        search_pipeline,
        gating_service,
        sample_episode_create: EpisodeCreate,
        load_customer_id: str,
    ):
        """
        Simulate realistic mixed workload:
        - 60% search
        - 30% ingestion
        - 10% gating

        Total: 1000 operations in ~10 seconds
        """
        num_operations = 1000
        concurrency = 50
        customer_id = load_customer_id

        # operation_counts mutation is safe: asyncio coroutines run on a single-threaded event loop
        # Counts are approximate under concurrent execution
        operation_counts = {"search": 0, "ingest": 0, "gate": 0}

        async def mixed_operation():
            op_type = random.choices(["search", "ingest", "gate"], weights=[60, 30, 10])[0]

            operation_counts[op_type] += 1

            start = time.perf_counter()

            if op_type == "search":
                request = SearchRequest(
                    customer_id=customer_id, goal="kubernetes deployment failed", k=5
                )
                await search_pipeline.search(request)

            elif op_type == "ingest":
                episode = sample_episode_create.model_copy(deep=True)
                await ingestion_pipeline.capture_episode(
                    episode_data=episode, generate_reflection=False
                )

            else:  # gate
                request = ReflectRequest(
                    goal="Deploy to production", proposed_action="kubectl apply", tool="kubectl"
                )
                await gating_service.reflect_before_action(request, customer_id)

            return (time.perf_counter() - start) * 1000

        metrics = await self._execute_concurrent_requests(
            request_func=mixed_operation, num_requests=num_operations, concurrency=concurrency
        )

        print("\nOperation Distribution:")
        print(f"  Search: {operation_counts['search']}")
        print(f"  Ingest: {operation_counts['ingest']}")
        print(f"  Gate: {operation_counts['gate']}")
        print(metrics.summary())

        # Assertions
        assert metrics.success_rate >= 99.0, f"Success rate {metrics.success_rate}% < 99%"
        if _strict_mode_enabled():
            assert metrics.p99_ms < 100, f"P99 latency {metrics.p99_ms}ms >= 100ms"
