"""
Sustained load test for Vritti (KyroDB + embeddings + pipelines).

This is an opt-in stress test intended for local benchmarking, not CI.

Why it's structured this way:
- Fixed-duration runs are more meaningful than "fixed N requests" runs (which can
  silently extend duration when the system falls behind).
- A worker pool with bounded concurrency avoids unbounded backlog and produces
  stable, reproducible throughput numbers.

Run:
  RUN_LOAD_TESTS=1 ./.venv/bin/pytest tests/load/test_load_1000_rps.py -v -s

Tuning (optional):
  LOAD_DURATION_SECONDS=30
  LOAD_CONCURRENCY=200
  LOAD_TARGET_RPS=1000          (only enforced when LOAD_TEST_STRICT=1)
  LOAD_TEST_STRICT=1            (enforce target RPS + latency SLOs)
"""

from __future__ import annotations

import asyncio
import os
import random
import statistics
import time
from dataclasses import dataclass

import pytest

from src.models.episode import EpisodeCreate, EpisodeType, ErrorClass
from src.models.gating import ReflectRequest
from src.models.search import SearchRequest


@dataclass
class LoadMetrics:
    total_requests: int
    successful: int
    failed: int
    latencies_ms: list[float]
    duration_seconds: float

    @property
    def throughput_rps(self) -> float:
        return self.total_requests / self.duration_seconds if self.duration_seconds > 0 else 0.0

    @property
    def success_rate(self) -> float:
        return (self.successful / self.total_requests * 100.0) if self.total_requests > 0 else 0.0

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p99_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        if len(self.latencies_ms) < 100:
            return max(self.latencies_ms)
        return statistics.quantiles(self.latencies_ms, n=100)[98]


def _summarize(op_type: str, metrics: LoadMetrics) -> None:
    print(f"\n{op_type.upper()} ({metrics.total_requests} requests):")
    print(f"  Success: {metrics.successful}/{metrics.total_requests} ({metrics.success_rate:.1f}%)")
    print(f"  P50 latency: {metrics.p50_ms:.1f}ms")
    print(f"  P99 latency: {metrics.p99_ms:.1f}ms")


@pytest.mark.load
@pytest.mark.asyncio
@pytest.mark.slow
async def test_sustained_load_worker_pool(
    skip_if_no_kyrodb,
    ingestion_pipeline,
    search_pipeline,
    gating_service,
    load_customer_id,
):
    duration_seconds = int(os.getenv("LOAD_DURATION_SECONDS", "30"))
    concurrency = int(os.getenv("LOAD_CONCURRENCY", "200"))
    strict_env = os.getenv("LOAD_TEST_STRICT", "")
    strict = strict_env.strip().lower() in {"1", "true", "yes", "y", "on"}
    target_rps = int(os.getenv("LOAD_TARGET_RPS", "1000"))

    # Mix of operations (realistic workload)
    # 50% search, 30% gating, 20% ingestion
    operation_mix = {
        "search": 0.50,
        "gating": 0.30,
        "ingestion": 0.20,
    }

    print("\n" + "=" * 80)
    print("SUSTAINED LOAD TEST (WORKER POOL)")
    print("=" * 80)
    print("Configuration:")
    print(f"  Duration: {duration_seconds}s")
    print(f"  Concurrency: {concurrency}")
    if strict:
        print(f"  Strict mode: ON (target_rps={target_rps})")
    else:
        print("  Strict mode: OFF (reports throughput; enforces only reliability floor)")

    stop_at = time.perf_counter() + duration_seconds

    # Each worker keeps local buffers (reduces contention); we merge at the end.
    async def worker(worker_id: int):
        local_results = {
            "search": [],
            "gating": [],
            "ingestion": [],
        }
        local_success = {
            "search": 0,
            "gating": 0,
            "ingestion": 0,
        }
        local_fail = {
            "search": 0,
            "gating": 0,
            "ingestion": 0,
        }

        while time.perf_counter() < stop_at:
            rand = random.random()
            if rand < operation_mix["search"]:
                op_type = "search"
            elif rand < operation_mix["search"] + operation_mix["gating"]:
                op_type = "gating"
            else:
                op_type = "ingestion"

            op_start = time.perf_counter()
            try:
                if op_type == "search":
                    request = SearchRequest(
                        customer_id=load_customer_id,
                        goal=f"test query {random.randint(1, 10_000)}",
                        k=5,
                        min_similarity=0.0,
                    )
                    await search_pipeline.search(request)
                elif op_type == "gating":
                    request = ReflectRequest(
                        goal=f"test goal {random.randint(1, 1000)}",
                        proposed_action=f"action {random.randint(1, 10_000)}",
                        tool="kubectl",
                        current_state={"env": "test"},
                        context="load-test",
                    )
                    await gating_service.reflect_before_action(request, load_customer_id)
                else:
                    episode = EpisodeCreate(
                        episode_type=EpisodeType.FAILURE,
                        goal=f"test goal {random.randint(1, 1000)}",
                        actions_taken=["test action"],
                        error_class=ErrorClass.NETWORK_ERROR,
                        error_trace=f"test error {random.randint(1, 10_000)}",
                        tool_chain=["test"],
                        environment_info={"test": "true"},
                        customer_id=load_customer_id,
                    )
                    await ingestion_pipeline.capture_episode(
                        episode_data=episode, generate_reflection=False
                    )

                local_success[op_type] += 1
            except Exception:
                local_fail[op_type] += 1
            finally:
                latency_ms = (time.perf_counter() - op_start) * 1000.0
                local_results[op_type].append(latency_ms)

        return local_results, local_success, local_fail

    start_time = time.perf_counter()
    tasks = [worker(i) for i in range(concurrency)]
    worker_outcomes = await asyncio.gather(*tasks)
    total_duration = time.perf_counter() - start_time

    # Merge worker results
    results = {"search": [], "gating": [], "ingestion": []}
    success = {"search": 0, "gating": 0, "ingestion": 0}
    fail = {"search": 0, "gating": 0, "ingestion": 0}

    for local_results, local_success, local_fail in worker_outcomes:
        for op_type in results:
            results[op_type].extend(local_results[op_type])
            success[op_type] += local_success[op_type]
            fail[op_type] += local_fail[op_type]

    overall_latencies = results["search"] + results["gating"] + results["ingestion"]
    overall_successful = sum(success.values())
    overall_failed = sum(fail.values())
    overall_total = overall_successful + overall_failed

    overall = LoadMetrics(
        total_requests=overall_total,
        successful=overall_successful,
        failed=overall_failed,
        latencies_ms=overall_latencies,
        duration_seconds=total_duration,
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for op_type in ("search", "gating", "ingestion"):
        m = LoadMetrics(
            total_requests=success[op_type] + fail[op_type],
            successful=success[op_type],
            failed=fail[op_type],
            latencies_ms=results[op_type],
            duration_seconds=total_duration,
        )
        _summarize(op_type, m)

    print(f"\nOVERALL ({overall.total_requests} total requests):")
    print(f"  Duration: {total_duration:.1f}s")
    print(f"  Throughput: {overall.throughput_rps:.1f} RPS")
    print(f"  Success rate: {overall.success_rate:.1f}%")
    print(f"  P50 latency: {overall.p50_ms:.1f}ms")
    print(f"  P99 latency: {overall.p99_ms:.1f}ms")
    print("=" * 80 + "\n")

    # Reliability floor: even in non-strict mode, we require the system to remain mostly healthy.
    min_success_rate = 99.0 if strict else 98.0
    assert (
        overall.success_rate >= min_success_rate
    ), f"Success rate {overall.success_rate:.2f}% < {min_success_rate:.2f}%"

    if strict:
        assert (
            overall.throughput_rps >= target_rps
        ), f"Throughput {overall.throughput_rps:.0f} RPS < {target_rps} RPS"
        assert overall.p99_ms < 200.0, f"P99 latency {overall.p99_ms:.1f}ms > 200ms"
