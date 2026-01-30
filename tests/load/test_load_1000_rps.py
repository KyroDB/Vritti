"""
1000 RPS Load Test for Vritti Production Readiness

Tests system under sustained 1000 requests/second load.

Run with: pytest tests/load/test_load_1000_rps.py -v -s
"""

import asyncio
import os
import time
import statistics
import random
from typing import List
from dataclasses import dataclass
import pytest

from src.models.episode import EpisodeCreate, EpisodeType, ErrorClass
from src.models.search import SearchRequest
from src.models.gating import ReflectRequest


@dataclass
class LoadMetrics:
    total_requests: int
    successful: int
    failed: int
    latencies_ms: List[float]
    duration_seconds: float

    @property
    def throughput_rps(self) -> float:
        return self.total_requests / self.duration_seconds if self.duration_seconds > 0 else 0

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0

    @property
    def p99_ms(self) -> float:
        if not self.latencies_ms or len(self.latencies_ms) < 100:
            return max(self.latencies_ms) if self.latencies_ms else 0
        return statistics.quantiles(self.latencies_ms, n=100)[98]

    @property
    def success_rate(self) -> float:
        return (self.successful / self.total_requests * 100) if self.total_requests > 0 else 0


class Test1000RPSLoad:
    """1000 RPS sustained load test."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.getenv("RUN_LOAD_TESTS"),
        reason="Load test is disabled by default. Set RUN_LOAD_TESTS=1 to run."
    )
    async def test_sustained_1000_rps(
        self, ingestion_pipeline, search_pipeline, gating_service
    ):
        """
        Test sustained 1000 RPS load for 30 seconds.

        Target: 30,000 requests in 30 seconds
        SLO: P99 latency <200ms, success rate >99%
        """
        print("\n" + "="*80)
        print("1000 RPS SUSTAINED LOAD TEST")
        print("="*80)

        duration_seconds = 30
        target_rps = 1000
        total_requests = duration_seconds * target_rps

        print(f"\nConfiguration:")
        print(f"  Duration: {duration_seconds}s")
        print(f"  Target RPS: {target_rps}")
        print(f"  Total requests: {total_requests}")

        # Mix of operations (realistic workload)
        # 50% search, 30% gating, 20% ingestion
        operation_mix = {
            'search': 0.50,
            'gating': 0.30,
            'ingestion': 0.20,
        }

        results = {
            'search': [],
            'gating': [],
            'ingestion': [],
        }

        start_time = time.perf_counter()

        # Generate requests in batches of 100 to maintain 1000 RPS
        batch_size = 100
        batch_interval = batch_size / target_rps  # 0.1 seconds

        completed = 0

        async def execute_operation(op_type: str):
            """Execute single operation and measure latency."""
            op_start = time.perf_counter()
            success = False

            try:
                if op_type == 'search':
                    request = SearchRequest(
                        customer_id="test_customer",
                        goal=f"test query {random.randint(1, 1000)}",
                        k=5,
                        min_similarity=0.7,
                    )
                    await search_pipeline.search(request)
                    success = True

                elif op_type == 'gating':
                    request = ReflectRequest(
                        proposed_action=f"action {random.randint(1, 1000)}",
                        goal="test goal " + str(random.randint(1, 100)),
                        tool="kubectl",
                        context="test context",
                        current_state={"env": "test"},
                    )
                    await gating_service.reflect_before_action(request, "test_customer")
                    success = True

                else:  # ingestion
                    episode = EpisodeCreate(
                        episode_type=EpisodeType.FAILURE,
                        goal="test goal " + str(random.randint(1, 100)),
                        actions_taken=["test action"],
                        error_class=ErrorClass.NETWORK_ERROR,
                        error_trace=f"test error {random.randint(1, 1000)}",
                        tool_chain=["test"],
                        environment_info={"test": "true"},
                        customer_id="test_customer",
                    )
                    await ingestion_pipeline.capture_episode(episode, generate_reflection=False)
                    success = True

            except Exception as e:
                print(f"\n  Error in {op_type}: {str(e)[:50]}")

            latency_ms = (time.perf_counter() - op_start) * 1000
            return op_type, success, latency_ms

        # Execute load test
        batch_num = 0
        while completed < total_requests:
            batch_start = time.perf_counter()

            # Create batch of mixed operations
            tasks = []
            for _ in range(batch_size):
                if completed >= total_requests:
                    break

                # Select operation type based on mix
                rand = random.random()
                if rand < operation_mix['search']:
                    op_type = 'search'
                elif rand < operation_mix['search'] + operation_mix['gating']:
                    op_type = 'gating'
                else:
                    op_type = 'ingestion'

                tasks.append(execute_operation(op_type))
                completed += 1

            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Record results
            for result in batch_results:
                if isinstance(result, tuple):
                    op_type, success, latency_ms = result
                    results[op_type].append({
                        'success': success,
                        'latency_ms': latency_ms,
                    })
                elif isinstance(result, Exception):
                    # Log unexpected exceptions that weren't caught in execute_operation
                    print(f"\n  ⚠️  Unexpected exception: {type(result).__name__}: {str(result)[:50]}")
                    # Count as failure for the operation type (we don't know which, so skip)

            batch_num += 1

            # Progress update every 5 seconds
            elapsed = time.perf_counter() - start_time
            if batch_num % 50 == 0:
                current_rps = completed / elapsed if elapsed > 0 else 0
                print(f"  Progress: {completed}/{total_requests} ({completed/total_requests*100:.1f}%) | "
                      f"Elapsed: {elapsed:.1f}s | Current RPS: {current_rps:.0f}")

            # Rate limiting: wait for next batch interval
            batch_elapsed = time.perf_counter() - batch_start
            sleep_time = max(0, batch_interval - batch_elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        total_duration = time.perf_counter() - start_time

        # Compile metrics for each operation type
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)

        overall_successful = 0
        overall_failed = 0
        all_latencies = []

        for op_type, op_results in results.items():
            if not op_results:
                continue

            successful = sum(1 for r in op_results if r['success'])
            failed = len(op_results) - successful
            latencies = [r['latency_ms'] for r in op_results]

            overall_successful += successful
            overall_failed += failed
            all_latencies.extend(latencies)

            metrics = LoadMetrics(
                total_requests=len(op_results),
                successful=successful,
                failed=failed,
                latencies_ms=latencies,
                duration_seconds=total_duration,
            )

            print(f"\n{op_type.upper()} ({len(op_results)} requests):")
            print(f"  Success: {successful}/{len(op_results)} ({metrics.success_rate:.1f}%)")
            print(f"  P50 latency: {metrics.p50_ms:.1f}ms")
            print(f"  P99 latency: {metrics.p99_ms:.1f}ms")

        # Overall metrics
        overall_metrics = LoadMetrics(
            total_requests=completed,
            successful=overall_successful,
            failed=overall_failed,
            latencies_ms=all_latencies,
            duration_seconds=total_duration,
        )

        print(f"\nOVERALL ({completed} total requests):")
        print(f"  Duration: {total_duration:.1f}s")
        print(f"  Throughput: {overall_metrics.throughput_rps:.1f} RPS")
        print(f"  Success rate: {overall_metrics.success_rate:.1f}%")
        print(f"  P50 latency: {overall_metrics.p50_ms:.1f}ms")
        print(f"  P99 latency: {overall_metrics.p99_ms:.1f}ms")

        print("\n" + "="*80)

        # Assertions
        assert overall_metrics.throughput_rps >= 900, f"Throughput {overall_metrics.throughput_rps:.0f} RPS < 900 RPS"
        assert overall_metrics.success_rate >= 99.0, f"Success rate {overall_metrics.success_rate:.1f}% < 99%"
        assert overall_metrics.p99_ms < 200, f"P99 latency {overall_metrics.p99_ms:.1f}ms > 200ms"

        print("\n✅ LOAD TEST PASSED")
        print(f"   Sustained {overall_metrics.throughput_rps:.0f} RPS for {duration_seconds}s")
        print(f"   Success rate: {overall_metrics.success_rate:.1f}%")
        print(f"   P99 latency: {overall_metrics.p99_ms:.1f}ms")
        print("="*80 + "\n")
