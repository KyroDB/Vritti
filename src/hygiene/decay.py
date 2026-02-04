"""
Memory decay and archival policy for episodic memory hygiene.

Implements intelligent memory consolidation with permanent protection
for critical failures.

Security:
- Customer isolation enforced
- Permanent protection for critical errors
- Audit logging for all delete operations
- Dry-run mode for testing

Performance:
- Batch operations
- Async I/O throughout
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Protocol, cast

from src.kyrodb.router import KyroDBRouter, get_namespaced_collection
from src.models.episode import Episode, ErrorClass

logger = logging.getLogger(__name__)


class EpisodeIndex(Protocol):
    async def list_episode_ids(
        self,
        *,
        customer_id: str,
        collection: str,
        include_archived: bool = False,
        include_deleted: bool = False,
        limit: int | None = None,
    ) -> list[int]: ...

    async def mark_episode_archived(
        self,
        *,
        customer_id: str,
        episode_id: int,
        archived: bool,
    ) -> bool: ...

    async def mark_episode_deleted(
        self,
        *,
        customer_id: str,
        episode_id: int,
    ) -> bool: ...


class MemoryDecayPolicy:
    """
    Intelligent memory decay with permanent protection.

    Rules:
    1. Archive episodes >6 months old (hidden from search, not deleted)
    2. Delete episodes unused for 3 months (retrieval_count=0)
    3. NEVER delete/archive critical failures (permanent memory)
    4. NEVER delete high-performing skills (success_rate >75%)

    Security:
    - All deletions logged for audit
    - Dry-run mode available
    - Customer isolation
    - Permanent protection checks
    """

    # Critical error classes that trigger permanent protection
    PERMANENT_ERROR_CLASSES = {
        ErrorClass.CONFIGURATION_ERROR.value,  # Keep config failures
        # Note: Add custom critical classes when enum is extended
        # "data_loss", "security_breach", "production_outage", "corruption"
    }

    def __init__(
        self,
        kyrodb_router: KyroDBRouter,
        episode_index: EpisodeIndex,
        archive_age_days: int = 180,
        delete_unused_days: int = 90,
    ):
        """
        Initialize decay policy.

        Args:
            kyrodb_router: KyroDB router for data access
            episode_index: Local index for enumerating episodes (authoritative for scan/list)
            archive_age_days: Archive episodes older than N days
            delete_unused_days: Delete unused episodes after N days
        """
        if archive_age_days < 30:
            raise ValueError("archive_age_days must be >= 30 for safety")
        if delete_unused_days < 30:
            raise ValueError("delete_unused_days must be >= 30 for safety")

        self.kyrodb_router = kyrodb_router
        self.episode_index = episode_index
        self.archive_age_days = archive_age_days
        self.delete_unused_days = delete_unused_days

        logger.info(
            f"MemoryDecayPolicy initialized "
            f"(archive: {archive_age_days}d, delete unused: {delete_unused_days}d)"
        )

    async def apply_decay_policy(
        self, customer_id: str, collection: str = "failures", dry_run: bool = False
    ) -> dict[str, int]:
        """
        Apply decay policy to all episodes for a customer.

        Args:
            customer_id: Customer ID
            collection: Episode collection
            dry_run: If True, only log what would be done (don't modify)

        Returns:
            Stats dict: {archived: int, deleted: int, protected: int, total: int}

        Security:
            - Validates customer_id
            - Permanent protection enforced
            - All actions logged
        """
        logger.info(
            f"Applying decay policy for {customer_id} "
            f"(dry_run={dry_run}, collection={collection})"
        )

        # Validate customer_id (type-safe)
        if not customer_id or not isinstance(customer_id, str) or len(customer_id) > 100:
            raise ValueError(f"Invalid customer_id: {customer_id}")

        # Calculate cutoff dates
        now = datetime.now(UTC)
        archive_cutoff = now - timedelta(days=self.archive_age_days)
        delete_cutoff = now - timedelta(days=self.delete_unused_days)

        # Fetch all episodes
        episodes = await self._fetch_all_episodes(customer_id, collection)

        stats = {"total": len(episodes), "archived": 0, "deleted": 0, "protected": 0, "errors": 0}

        for episode in episodes:
            try:
                # Check for permanent protection
                if self._is_permanent(episode):
                    stats["protected"] += 1
                    logger.debug(f"Episode {episode.episode_id} protected (permanent memory)")
                    continue

                # Rule 1: Archive old episodes
                if episode.created_at < archive_cutoff:
                    if not dry_run:
                        await self._archive_episode(episode, customer_id, collection)
                    stats["archived"] += 1
                    logger.info(
                        f"{'[DRY-RUN] Would archive' if dry_run else 'Archived'} "
                        f"episode {episode.episode_id} "
                        f"(age: {(now - episode.created_at).days}d)"
                    )
                    continue

                # Rule 2: Delete unused episodes
                if episode.usage_stats.total_retrievals == 0 and episode.created_at < delete_cutoff:
                    if not dry_run:
                        await self._delete_episode(episode, customer_id, collection)
                    stats["deleted"] += 1
                    logger.warning(
                        f"{'[DRY-RUN] Would delete' if dry_run else 'Deleted'} "
                        f"episode {episode.episode_id} "
                        f"(unused for {(now - episode.created_at).days}d)"
                    )
                    continue

            except Exception as e:
                logger.error(f"Error processing episode {episode.episode_id}: {e}", exc_info=True)
                stats["errors"] += 1

        logger.info(
            f"Decay policy complete for {customer_id}: "
            f"{stats['archived']} archived, "
            f"{stats['deleted']} deleted, "
            f"{stats['protected']} protected, "
            f"{stats['errors']} errors "
            f"(total: {stats['total']}, dry_run={dry_run})"
        )

        return stats

    def _is_permanent(self, episode: Episode) -> bool:
        """
        Check if episode should be permanently protected.

        Protection criteria:
        1. Critical error class (security, data loss, etc.)
        2. Manually marked as permanent
        3. High-performing skill (success rate >75%, applied >5 times)

        Args:
            episode: Episode to check

        Returns:
            True if episode should never be deleted/archived
        """
        # Check 1: Explicit permanent flag (stored in environment_info)
        permanent_flag = episode.create_data.environment_info.get("permanent")
        if permanent_flag in (True, "true", "True", "1", 1):
            return True

        # Check 2: Critical error class
        if episode.create_data.error_class.value in self.PERMANENT_ERROR_CLASSES:
            return True

        # Check 3: Critical tags
        critical_tags = {"data_loss", "security_breach", "production_outage"}
        if {tag.lower() for tag in episode.create_data.tags} & critical_tags:
            return True

        # Check 4: High-performing fix (applied >5 times, success rate >75%)
        if (
            episode.usage_stats.fix_success_count > 5
            and episode.usage_stats.fix_success_rate > 0.75
        ):
            return True

        return False

    async def _archive_episode(self, episode: Episode, customer_id: str, collection: str) -> None:
        """
        Archive episode (mark as archived, don't delete).

        Archived episodes:
        - Hidden from search by default
        - Still retrievable if needed
        - Can be un-archived
        """
        episode.archived = True
        episode.archived_at = datetime.now(UTC)

        # Re-insert episode metadata so episode_json stays consistent with archived flag.
        await self._reinsert_episode_metadata(
            episode=episode,
            customer_id=customer_id,
            collection=collection,
        )

        # Best-effort: keep local index in sync for enumeration.
        try:
            await self.episode_index.mark_episode_archived(
                customer_id=customer_id,
                episode_id=episode.episode_id,
                archived=True,
            )
        except Exception as e:
            logger.warning(
                f"Failed to mark episode archived in local index: {e}",
                exc_info=True,
            )

    async def _delete_episode(self, episode: Episode, customer_id: str, collection: str) -> None:
        """
        Permanently delete episode.

        CAUTION: This is irreversible!
        Only called after permanent protection checks.
        """
        text_deleted, _ = await self.kyrodb_router.delete_episode(
            episode_id=episode.episode_id,
            customer_id=customer_id,
            collection=collection,
            delete_images=True,
        )
        if not text_deleted:
            raise RuntimeError(f"Failed to delete episode {episode.episode_id} from KyroDB")

        # Best-effort: keep local index in sync for enumeration.
        try:
            await self.episode_index.mark_episode_deleted(
                customer_id=customer_id,
                episode_id=episode.episode_id,
            )
        except Exception as e:
            logger.warning(
                f"Failed to mark episode deleted in local index: {e}",
                exc_info=True,
            )

    async def _fetch_all_episodes(self, customer_id: str, collection: str) -> list[Episode]:
        """
        Fetch all episodes for decay processing.

        Uses the local episode index for enumeration, then bulk-fetches metadata from KyroDB.
        """
        episode_ids = await self.episode_index.list_episode_ids(
            customer_id=customer_id,
            collection=collection,
            include_archived=True,
            include_deleted=False,
        )
        if not episode_ids:
            return []

        # KyroDB bulk query is bounded, but we still chunk to keep responses small.
        chunk_size = 256
        max_in_flight = 8
        semaphore = asyncio.Semaphore(max_in_flight)

        async def _fetch_chunk(ids: list[int]) -> list[Episode]:
            async with semaphore:
                return await self.kyrodb_router.bulk_fetch_episodes(
                    episode_ids=ids,
                    customer_id=customer_id,
                    collection=collection,
                )

        chunks: list[list[int]] = []
        tasks: list[asyncio.Task[list[Episode]]] = []
        for i in range(0, len(episode_ids), chunk_size):
            chunk = episode_ids[i : i + chunk_size]
            chunks.append(chunk)
            tasks.append(asyncio.create_task(_fetch_chunk(chunk)))

        episodes: list[Episode] = []
        chunk_errors: list[tuple[list[int], Exception]] = []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for chunk_ids, item in zip(chunks, results, strict=True):
            if isinstance(item, Exception):
                chunk_errors.append((chunk_ids, item))
                logger.error(
                    "Failed to fetch episode chunk for decay",
                    extra={
                        "customer_id": customer_id,
                        "collection": collection,
                        "chunk_size": len(chunk_ids),
                        "chunk_first_id": chunk_ids[0] if chunk_ids else None,
                        "chunk_last_id": chunk_ids[-1] if chunk_ids else None,
                        "error": str(item),
                    },
                    exc_info=True,
                )
                continue
            episodes.extend(cast(list[Episode], item))

        if chunk_errors and not episodes:
            raise RuntimeError(
                f"Failed to fetch all episode chunks for decay "
                f"(errors={len(chunk_errors)}, customer_id={customer_id}, collection={collection})"
            ) from chunk_errors[0][1]

        if chunk_errors:
            logger.warning(
                "Decay fetch completed with partial chunk failures",
                extra={
                    "customer_id": customer_id,
                    "collection": collection,
                    "chunks_failed": len(chunk_errors),
                    "chunks_total": len(chunks),
                    "episodes_returned": len(episodes),
                },
            )

        return episodes

    async def _reinsert_episode_metadata(
        self,
        *,
        episode: Episode,
        customer_id: str,
        collection: str,
    ) -> None:
        """
        Update an episode's metadata in KyroDB (text + image if present).

        We use `UpdateMetadata` with merge semantics so that extra system metadata stored
        alongside the episode payload (e.g., `reflection_*`, `cluster_id`) is preserved.
        """
        namespaced_collection = get_namespaced_collection(customer_id, collection)

        updated_metadata = episode.to_metadata_dict()
        response = await self.kyrodb_router.text_client.update_metadata(
            doc_id=episode.episode_id,
            namespace=namespaced_collection,
            metadata=updated_metadata,
            merge=True,
        )
        if not response.success:
            raise RuntimeError(
                f"Failed to update metadata for episode {episode.episode_id}: {response.error}"
            )

        # Keep image namespace metadata consistent so image-only retrieval behaves correctly.
        image_namespace = f"{namespaced_collection}_images"
        try:
            image_existing = await self.kyrodb_router.image_client.query(
                doc_id=episode.episode_id,
                namespace=image_namespace,
                include_embedding=False,
            )
            if image_existing.found:
                image_response = await self.kyrodb_router.image_client.update_metadata(
                    doc_id=episode.episode_id,
                    namespace=image_namespace,
                    metadata=updated_metadata,
                    merge=True,
                )
                if not image_response.success:
                    logger.warning(
                        f"Failed to update image metadata for archived episode {episode.episode_id}: "
                        f"{image_response.error}"
                    )
        except Exception as e:
            logger.warning(
                f"Image metadata update failed for archived episode {episode.episode_id}: {e}",
                exc_info=True,
            )
