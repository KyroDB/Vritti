"""
Episode ingestion pipeline.

Handles end-to-end capture of failure/success episodes:
1. PII redaction
2. ID generation
3. Multi-modal embedding
4. KyroDB storage
5. Async reflection generation

Designed for <50ms P99 latency (excluding async reflection).
"""

import asyncio
import logging
from datetime import UTC, datetime

from src.config import get_settings
from src.ingestion.embedding import EmbeddingService
from src.ingestion.reflection import ReflectionService
from src.kyrodb.router import KyroDBRouter
from src.models.episode import Episode, EpisodeCreate
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
        reflection_service: ReflectionService | None = None,
    ):
        """
        Initialize ingestion pipeline.

        Args:
            kyrodb_router: KyroDB router for dual-instance storage
            embedding_service: Multi-modal embedding service
            reflection_service: Optional LLM reflection service
        """
        self.kyrodb_router = kyrodb_router
        self.embedding_service = embedding_service
        self.reflection_service = reflection_service

        self.settings = get_settings()
        self.total_ingested = 0
        self.total_failures = 0

    async def capture_episode(
        self,
        episode_data: EpisodeCreate,
        generate_reflection: bool = True,
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
                asyncio.create_task(self._generate_and_update_reflection(episode_id, episode_data))

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
        self, episode_id: int, episode_data: EpisodeCreate
    ) -> None:
        """
        Generate reflection asynchronously and update KyroDB.

        This runs in the background after the main ingestion completes.

        Args:
            episode_id: Episode ID
            episode_data: Episode data for reflection
        """
        try:
            logger.info(f"Generating reflection for episode {episode_id}...")

            reflection = await self.reflection_service.generate_reflection(episode_data)

            if reflection:
                # TODO: Update episode in KyroDB with reflection
                # For now, reflection is generated but not persisted
                # Phase 2 will add reflection storage and retrieval
                logger.info(
                    f"Reflection generated for episode {episode_id} "
                    f"(confidence: {reflection.confidence_score:.2f})"
                )
            else:
                logger.warning(f"Reflection generation failed for episode {episode_id}")

        except Exception as e:
            logger.error(
                f"Async reflection generation failed for episode {episode_id}: {e}",
                exc_info=True,
            )

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
        # TODO: Implement batch optimization
        # For now, process sequentially
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
