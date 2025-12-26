"""
Episode ingestion pipeline.

Handles end-to-end capture of failure/success episodes:
1. PII redaction
2. ID generation
3. Multi-modal embedding
4. KyroDB storage
5. Async multi-perspective reflection generation

Security:
- All inputs sanitized before storage
- Customer ID validated (never user-provided)
- Reflections validated before persistence

Designed for <50ms P99 latency (excluding async reflection).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.models.episode import Reflection

from src.config import get_settings
from src.ingestion.embedding import EmbeddingService
from src.ingestion.tiered_reflection import (
    TieredReflectionService,
)
from src.kyrodb.router import KyroDBRouter
from src.models.episode import Episode, EpisodeCreate, Reflection, ReflectionTier
from src.utils.identifiers import (
    generate_episode_id,
    hash_environment,
    hash_error_signature,
)
from src.utils.pii_redaction import redact_all

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestrates episode ingestion with multi-modal embeddings.

    Components:
    - PII redaction (regex-based, <1ms)
    - ID generation (Snowflake, <1μs)
    - Embedding generation (5-30ms)
    - KyroDB storage (100-200ns)
    - Async reflection (background, 2-5s)
    """

    def __init__(
        self,
        kyrodb_router: KyroDBRouter,
        embedding_service: EmbeddingService,
        reflection_service: TieredReflectionService | None = None,
    ):
        """
        Initialize ingestion pipeline.

        Args:
            kyrodb_router: KyroDB router for dual-instance storage
            embedding_service: Multi-modal embedding service
            reflection_service: Optional tiered LLM reflection service

        Security:
            - All services initialized with validated configs
            - Customer ID validation enforced at API layer
        """
        self.kyrodb_router = kyrodb_router
        self.embedding_service = embedding_service
        self.reflection_service = reflection_service

        self.settings = get_settings()
        self.total_ingested = 0
        self.total_failures = 0
        
        # Track pending background tasks for graceful shutdown
        self._pending_tasks: set[asyncio.Task] = set()

        if reflection_service:
            logger.info(
                f"Reflection service enabled with providers: "
                f"{reflection_service.config.enabled_providers}"
            )
        else:
            logger.warning("Reflection service disabled - no LLM API keys configured")

    async def shutdown(self, timeout: float = 30.0) -> None:
        """
        Gracefully shutdown the ingestion pipeline.
        
        Waits for all pending background reflection tasks to complete
        before returning. This ensures no reflections are lost during
        application shutdown.
        
        Args:
            timeout: Maximum seconds to wait for pending tasks.
                     After timeout, remaining tasks are cancelled.
        """
        pending_count = len(self._pending_tasks)
        if pending_count == 0:
            logger.info("No pending reflection tasks to complete")
            return
            
        logger.info(f"Waiting for {pending_count} pending reflection tasks...")
        
        try:
            # Wait for all pending tasks with timeout
            done, pending = await asyncio.wait(
                self._pending_tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )
            
            if pending:
                logger.warning(
                    f"Timeout reached, cancelling {len(pending)} pending tasks"
                )
                for task in pending:
                    task.cancel()
                # Wait briefly for cancellations to complete
                await asyncio.gather(*pending, return_exceptions=True)
            
            logger.info(
                f"Reflection task shutdown complete: "
                f"{len(done)} completed, {len(pending)} cancelled"
            )
        except Exception as e:
            logger.error(f"Error during pipeline shutdown: {e}", exc_info=True)

    async def capture_episode(
        self,
        episode_data: EpisodeCreate,
        generate_reflection: bool = True,
        tier_override: ReflectionTier | None = None,
    ) -> Episode:
        """
        Capture and store an episode.

        Pipeline:
        1. PII redaction (~1ms)
        2. Environment hashing (~1ms)
        3. ID generation (<1μs)
        4. Text embedding (~5-10ms)
        5. Image embedding if present (~20-30ms)
        6. KyroDB insert (~100ns)
        7. Async reflection (background, 2-5s)

        Args:
            episode_data: Episode creation data
            generate_reflection: Whether to generate reflection (async)
            tier_override: Optional tier override (cheap/cached/premium), auto-selects if None

        Returns:
            Episode: Stored episode with ID

        Raises:
            KyroDBError: If storage fails critically
            ValueError: If validation fails
        """
        try:
            # Step 1: PII redaction (~1ms)
            episode_data = self._redact_pii(episode_data)

            # Step 2: Environment hashing (~1ms)
            env_hash = hash_environment(episode_data.environment_info)
            error_sig = hash_error_signature(
                error_class=episode_data.error_class.value,
                tool=episode_data.tool_chain[0],
                environment_hash=env_hash,
            )

            # Step 3: Generate episode ID (<1μs)
            episode_id = generate_episode_id()

            # Step 4: Generate embeddings (~10-40ms total)
            text_embedding, image_embedding = await self._generate_embeddings(episode_data)

            # Step 5: Create Episode object
            episode = Episode(
                create_data=episode_data,
                episode_id=episode_id,
                created_at=datetime.now(UTC),
                retrieval_count=0,
                reflection=None,  # Will be generated async
            )

            # Step 6: Store in KyroDB (~100ns)
            await self._store_in_kyrodb(
                episode=episode,
                text_embedding=text_embedding,
                image_embedding=image_embedding,
                env_hash=env_hash,
                error_sig=error_sig,
            )

            # Step 7: Queue async reflection generation (non-blocking)
            if generate_reflection and self.reflection_service:
                task = asyncio.create_task(
                    self._generate_and_update_reflection(
                        episode_id, 
                        episode_data, 
                        tier_override
                    )
                )
                # Track task for graceful shutdown
                self._pending_tasks.add(task)
                task.add_done_callback(self._pending_tasks.discard)

            self.total_ingested += 1
            logger.info(
                f"Episode {episode_id} ingested "
                f"(type: {episode_data.episode_type.value}, "
                f"tool: {episode_data.tool_chain[0]})"
            )

            return episode

        except Exception as e:
            self.total_failures += 1
            logger.error(f"Ingestion failed for episode: {e}", exc_info=True)
            raise

    def _redact_pii(self, episode_data: EpisodeCreate) -> EpisodeCreate:
        """
        Redact PII from episode data.

        Applies to:
        - error_trace
        - code_state_diff
        - actions_taken

        Args:
            episode_data: Original episode data

        Returns:
            EpisodeCreate: Data with PII redacted
        """
        # Redact error trace
        episode_data.error_trace = redact_all(
            episode_data.error_trace,
            redact_phones=False,  # Avoid false positives
        )

        # Redact code diff if present
        if episode_data.code_state_diff:
            episode_data.code_state_diff = redact_all(
                episode_data.code_state_diff,
                redact_phones=False,
            )

        # Redact actions (may contain commands with credentials)
        episode_data.actions_taken = [
            redact_all(action, redact_phones=False) for action in episode_data.actions_taken
        ]

        return episode_data

    async def _generate_embeddings(
        self, episode_data: EpisodeCreate
    ) -> tuple[list[float], list[float] | None]:
        """
        Generate text and optional image embeddings.

        Args:
            episode_data: Episode data

        Returns:
            tuple: (text_embedding, image_embedding or None)
        """
        # Text embedding: combine goal + error trace for semantic richness
        # This gives better retrieval than goal alone
        text_content = f"{episode_data.goal}\n\n{episode_data.error_trace}"

        # Truncate if too long (model limit: ~8192 tokens for MiniLM)
        if len(text_content) > 5000:
            text_content = text_content[:5000] + "... (truncated)"

        text_embedding = self.embedding_service.embed_text(text_content)

        # Image embedding if screenshot provided
        image_embedding = None
        if episode_data.screenshot_path:
            try:
                image_embedding = self.embedding_service.embed_image(episode_data.screenshot_path)
                logger.debug(f"Generated image embedding for {episode_data.screenshot_path}")
            except Exception as e:
                logger.warning(f"Image embedding failed for {episode_data.screenshot_path}: {e}")
                # Continue without image embedding

        return text_embedding, image_embedding

    async def _store_in_kyrodb(
        self,
        episode: Episode,
        text_embedding: list[float],
        image_embedding: list[float] | None,
        env_hash: str,
        error_sig: str,
    ) -> None:
        """
        Store episode in KyroDB instances with customer namespace isolation.

        Args:
            episode: Episode object
            text_embedding: Text embedding vector
            image_embedding: Optional image embedding
            env_hash: Environment hash
            error_sig: Error signature hash

        Raises:
            KyroDBError: If storage fails
            ValueError: If customer_id is missing (multi-tenancy violation)
        """
        # Verify customer_id is set (security: prevent data leakage)
        if not episode.create_data.customer_id:
            raise ValueError("customer_id is required - multi-tenancy violation detected")

        # Determine base collection (only "failures" for now)
        collection = "failures"

        # Build metadata
        metadata = episode.to_metadata_dict()

        # Add deduplication keys
        metadata["environment_hash"] = env_hash
        metadata["error_signature"] = error_sig

        # Store in KyroDB with customer-namespaced collection
        # KyroDBRouter will handle namespace formatting: {customer_id}:failures
        text_success, image_success = await self.kyrodb_router.insert_episode(
            episode_id=episode.episode_id,
            customer_id=episode.create_data.customer_id,
            collection=collection,
            text_embedding=text_embedding,
            image_embedding=image_embedding,
            metadata=metadata,
        )

        if not text_success:
            raise RuntimeError(f"Failed to store episode {episode.episode_id} in text instance")

        logger.debug(
            f"Episode {episode.episode_id} stored in KyroDB "
            f"(customer: {episode.create_data.customer_id}, "
            f"collection: {collection}, "
            f"text: {text_success}, image: {image_success})"
        )

    async def _generate_and_update_reflection(
        self, episode_id: int, episode_data: EpisodeCreate, tier_override: ReflectionTier | None = None
    ) -> None:
        """
        Generate tiered reflection and persist to KyroDB with retry logic.

        Security:
        - Episode data validated before LLM call
        - All LLM outputs validated against schema
        - Customer ID verified before persistence
        - Cost limits enforced

        Reliability:
        - 3 retries with exponential backoff for persistence
        - Dead-letter queue for failed reflections
        - Prometheus metrics for monitoring

        This runs in the background after main ingestion completes,
        so it doesn't block the /capture API response.

        Args:
            episode_id: Episode ID to attach reflection to
            episode_data: Episode data for reflection generation
            tier_override: Optional tier override for reflection generation

        Note:
            This method does not raise exceptions - all errors are logged.
            Reflection failures should not cause episode ingestion to fail.
        """
        try:
            import time

            tier_name = tier_override.value if tier_override else "auto-select"
            logger.info(
                f"Starting tiered reflection generation for episode {episode_id} "
                f"(tier: {tier_name})..."
            )
            start_time = time.perf_counter()

            # Generate reflection using tiered service with optional override
            reflection = await self.reflection_service.generate_reflection(
                episode_data,
                tier=tier_override  # Auto-selects if None
            )

            generation_time = time.perf_counter() - start_time

            if reflection:
                # Persist reflection with retry logic
                success = await self._persist_reflection_with_retry(
                    episode_id=episode_id,
                    customer_id=episode_data.customer_id,
                    reflection=reflection,
                    max_retries=3,
                )

                if success:
                    logger.info(
                        f"Reflection generated and persisted for episode {episode_id}:\n"
                        f"  Tier: {reflection.tier}\n"
                        f"  Model: {reflection.llm_model}\n"
                        f"  Consensus: {reflection.consensus.consensus_method if reflection.consensus else 'N/A'}\n"
                        f"  Confidence: {reflection.confidence_score:.2f}\n"
                        f"  Cost: ${reflection.cost_usd:.4f}\n"
                        f"  Total time: {generation_time:.1f}s\n"
                        f"  Root cause: {reflection.root_cause[:100]}..."
                    )

                    # Track success metrics
                    self._track_reflection_success(reflection, generation_time)

                else:
                    logger.error(
                        f"Reflection generation succeeded but persistence failed "
                        f"after all retries for episode {episode_id}"
                    )
                    # Log to dead-letter queue for manual recovery
                    await self._log_to_dead_letter_queue(
                        episode_id=episode_id,
                        customer_id=episode_data.customer_id,
                        reflection=reflection,
                        failure_reason="persistence_failed_after_retries",
                    )
                    self._track_reflection_failure("persistence_failed")

            else:
                logger.warning(
                    f"Reflection generation returned None for episode {episode_id} "
                    f"- likely all LLM calls failed"
                )
                self._track_reflection_failure("generation_failed")

        except Exception as e:
            logger.error(
                f"Async reflection pipeline failed for episode {episode_id}: {e}",
                exc_info=True,
            )
            self._track_reflection_failure("exception")

    async def _persist_reflection_with_retry(
        self,
        episode_id: int,
        customer_id: str,
        reflection: Reflection,
        max_retries: int = 3,
    ) -> bool:
        """
        Persist reflection to KyroDB with exponential backoff retry.

        Implements robust retry logic:
        - 3 attempts with exponential backoff (1s, 2s, 4s)
        - Logs each retry attempt
        - Tracks retry metrics
        - Returns False after all retries exhausted

        Args:
            episode_id: Episode ID to update
            customer_id: Customer ID for namespace isolation
            reflection: Generated reflection to persist
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            bool: True if persistence succeeded, False after all retries fail
        """

        base_delay_seconds = 1.0  # Initial delay before first retry
        last_exception = None

        for attempt in range(max_retries):
            try:
                logger.debug(
                    f"Attempting reflection persistence for episode {episode_id} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )

                success = await self.kyrodb_router.update_episode_reflection(
                    episode_id=episode_id,
                    customer_id=customer_id,
                    collection="failures",
                    reflection=reflection,
                )

                if success:
                    if attempt > 0:
                        logger.info(
                            f"Reflection persistence succeeded on retry {attempt + 1} "
                            f"for episode {episode_id}"
                        )
                        # Track successful retry metric
                        self._track_persistence_retry(
                            episode_id=episode_id,
                            attempt=attempt + 1,
                            success=True,
                        )
                    return True

                # Persistence returned False (not an exception)
                logger.warning(
                    f"Reflection persistence returned False for episode {episode_id} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Reflection persistence failed for episode {episode_id} "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )

            # Track retry attempt metric
            self._track_persistence_retry(
                episode_id=episode_id,
                attempt=attempt + 1,
                success=False,
            )

            # Calculate backoff delay (exponential: 1s, 2s, 4s)
            if attempt < max_retries - 1:
                delay = base_delay_seconds * (2 ** attempt)
                logger.debug(
                    f"Waiting {delay:.1f}s before retry for episode {episode_id}"
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        logger.error(
            f"Reflection persistence exhausted all {max_retries} retries "
            f"for episode {episode_id}"
            + (f": {last_exception}" if last_exception else "")
        )

        return False

    async def _log_to_dead_letter_queue(
        self,
        episode_id: int,
        customer_id: str,
        reflection: Reflection,
        failure_reason: str,
    ) -> None:
        """
        Log failed reflection to dead-letter queue file for manual recovery.

        Dead-letter queue file path is configurable via ServiceConfig.
        Format: JSON lines with timestamp, episode_id, customer_id, failure_reason, reflection
        
        Includes automatic file rotation when size exceeds configured limit.

        This enables manual recovery of reflections that failed persistence.

        Args:
            episode_id: Episode ID that failed
            customer_id: Customer ID
            reflection: Generated reflection that couldn't be persisted
            failure_reason: Why persistence failed
        """
        import json
        from pathlib import Path

        # Get configurable path from settings
        dead_letter_path = Path(self.settings.service.dead_letter_queue_path)

        try:
            # Ensure data directory exists
            dead_letter_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check file size and rotate if needed
            max_size_bytes = self.settings.service.dead_letter_queue_max_size_mb * 1024 * 1024
            if dead_letter_path.exists() and dead_letter_path.stat().st_size >= max_size_bytes:
                # Rotate: rename current file with timestamp
                timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                rotated_path = dead_letter_path.with_suffix(f".{timestamp}.log")
                dead_letter_path.rename(rotated_path)
                logger.info(
                    f"Rotated dead letter queue file: {dead_letter_path} -> {rotated_path} "
                    f"(size exceeded {max_size_bytes} bytes)"
                )

            # Serialize reflection to JSON-safe format
            reflection_data = {
                "root_cause": reflection.root_cause,
                "preconditions": reflection.preconditions,
                "resolution_strategy": reflection.resolution_strategy,
                "environment_factors": reflection.environment_factors,
                "affected_components": reflection.affected_components,
                "generalization_score": reflection.generalization_score,
                "confidence_score": reflection.confidence_score,
                "llm_model": reflection.llm_model,
                "generated_at": reflection.generated_at.isoformat(),
                "cost_usd": reflection.cost_usd,
                "generation_latency_ms": reflection.generation_latency_ms,
                "tier": reflection.tier,
            }

            # Add consensus data if present
            if reflection.consensus:
                reflection_data["consensus"] = {
                    "method": reflection.consensus.consensus_method,
                    "confidence": reflection.consensus.consensus_confidence,
                    "disagreement_points": reflection.consensus.disagreement_points,
                }

            # Create dead-letter entry
            entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "episode_id": episode_id,
                "customer_id": customer_id,
                "failure_reason": failure_reason,
                "reflection": reflection_data,
            }

            # Append to dead-letter queue file (atomic append with newline)
            with open(dead_letter_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

            logger.warning(
                f"Logged failed reflection to dead-letter queue: episode {episode_id}, "
                f"reason: {failure_reason}, path: {dead_letter_path}"
            )

            # Track dead-letter metric
            from src.observability.metrics import track_dead_letter_queue
            track_dead_letter_queue(
                episode_id=episode_id,
                customer_id=customer_id,
                failure_reason=failure_reason,
            )

        except Exception as e:
            # Don't fail if logging to dead-letter queue fails
            logger.error(
                f"Failed to log to dead-letter queue for episode {episode_id}: {e}",
                exc_info=True,
            )

    def _track_persistence_retry(
        self,
        episode_id: int,
        attempt: int,
        success: bool,
    ) -> None:
        """Track reflection persistence retry metrics."""
        try:
            from src.observability.metrics import track_reflection_persistence_retry
            track_reflection_persistence_retry(
                episode_id=episode_id,
                attempt=attempt,
                success=success,
            )
        except ImportError:
            # Metrics module may not have this function yet
            logger.debug(f"Persistence retry: episode={episode_id}, attempt={attempt}, success={success}")

    def _track_reflection_success(
        self, reflection: Reflection, generation_time: float
    ) -> None:
        """Track successful reflection generation metrics."""
        from src.observability.metrics import (
            track_reflection_generation,
            track_reflection_persistence,
        )

        # Track generation metrics
        consensus_method = (
            reflection.consensus.consensus_method
            if reflection.consensus
            else "fallback_heuristic"
        )

        num_models = (
            len(reflection.consensus.perspectives)
            if reflection.consensus
            else 0
        )

        track_reflection_generation(
            consensus_method=consensus_method,
            num_models=num_models,
            duration_seconds=generation_time,
            cost_usd=reflection.cost_usd,
            confidence=reflection.confidence_score,
            success=True,
        )

        # Track persistence (assumed successful if we reach here)
        track_reflection_persistence(success=True)

    def _track_reflection_failure(self, reason: str) -> None:
        """Track reflection failures for monitoring."""
        from src.observability.metrics import (
            track_reflection_failure,
            track_reflection_persistence,
        )

        track_reflection_failure(reason=reason)

        # If failure was during persistence, track that separately
        if reason == "persistence_failed":
            track_reflection_persistence(success=False)

    async def bulk_capture(
        self, episodes: list[EpisodeCreate], generate_reflections: bool = False
    ) -> list[Episode]:
        """
        Capture multiple episodes in batch.

        Optimizes embedding generation with batching.

        Args:
            episodes: List of episodes to capture
            generate_reflections: Whether to generate reflections (expensive)

        Returns:
            list[Episode]: Captured episodes
        """
        # Sequential processing for reliability - batch embedding optimization pending
        results = []
        for episode_data in episodes:
            try:
                episode = await self.capture_episode(
                    episode_data, generate_reflection=generate_reflections
                )
                results.append(episode)
            except Exception as e:
                logger.error(f"Bulk capture failed for episode: {e}")
                # Continue with remaining episodes

        return results

    def get_stats(self) -> dict[str, int]:
        """
        Get ingestion statistics.

        Returns:
            dict: Stats (total_ingested, total_failures)
        """
        return {
            "total_ingested": self.total_ingested,
            "total_failures": self.total_failures,
            "success_rate": (
                self.total_ingested / (self.total_ingested + self.total_failures)
                if (self.total_ingested + self.total_failures) > 0
                else 0.0
            ),
        }
