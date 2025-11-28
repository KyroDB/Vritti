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

import logging
from datetime import timezone, datetime, timedelta

from src.kyrodb.router import KyroDBRouter
from src.models.episode import Episode, ErrorClass

logger = logging.getLogger(__name__)


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
        archive_age_days: int = 180,
        delete_unused_days: int = 90
    ):
        """
        Initialize decay policy.
        
        Args:
            kyrodb_router: KyroDB router for data access
            archive_age_days: Archive episodes older than N days
            delete_unused_days: Delete unused episodes after N days
        """
        if archive_age_days < 30:
            raise ValueError("archive_age_days must be >= 30 for safety")
        if delete_unused_days < 30:
            raise ValueError("delete_unused_days must be >= 30 for safety")
        
        self.kyrodb_router = kyrodb_router
        self.archive_age_days = archive_age_days
        self.delete_unused_days = delete_unused_days
        
        logger.info(
            f"MemoryDecayPolicy initialized "
            f"(archive: {archive_age_days}d, delete unused: {delete_unused_days}d)"
        )
    
    async def apply_decay_policy(
        self,
        customer_id: str,
        collection: str = "failures",
        dry_run: bool = False
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
        now = datetime.now(timezone.utc)
        archive_cutoff = now - timedelta(days=self.archive_age_days)
        delete_cutoff = now - timedelta(days=self.delete_unused_days)
        
        # Fetch all episodes
        episodes = await self._fetch_all_episodes(customer_id, collection)
       
        stats = {
            "total": len(episodes),
            "archived": 0,
            "deleted": 0,
            "protected": 0,
            "errors": 0
        }
        
        for episode in episodes:
            try:
                # Check for permanent protection
                if self._is_permanent(episode):
                    stats["protected"] += 1
                    logger.debug(
                        f"Episode {episode.episode_id} protected (permanent memory)"
                    )
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
                if (episode.usage_stats.total_retrievals == 0 and
                    episode.created_at < delete_cutoff):
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
                logger.error(
                    f"Error processing episode {episode.episode_id}: {e}",
                    exc_info=True
                )
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
        # Check 1: Permanent flag (check both new and legacy locations for migration compatibility)
        if hasattr(episode, 'create_data') and episode.create_data:
            if hasattr(episode.create_data, 'environment_info') and episode.create_data.environment_info:
                # Accept boolean or string representations
                if episode.create_data.environment_info.get("permanent") in (True, "true", "True", "1"):
                    return True

        # MIGRATION FALLBACK: Check legacy metadata location for permanent flag
        # This ensures episodes marked permanent before migration are still protected
        if hasattr(episode, 'metadata') and episode.metadata:
            if episode.metadata.get("permanent") in (True, "true", "True", "1"):
                logger.info(
                    f"Episode {episode.episode_id} has permanent flag in legacy metadata location "
                    f"- consider migrating to environment_info"
                )
                return True
        
        # Check 2: Critical error class (defensive nested access)
        if hasattr(episode, 'create_data') and episode.create_data:
            if hasattr(episode.create_data, 'error_class') and episode.create_data.error_class:
                try:
                    error_class_value = episode.create_data.error_class.value
                    if error_class_value in self.PERMANENT_ERROR_CLASSES:
                        return True
                except AttributeError:
                    logger.warning(f"Episode {episode.episode_id} has invalid error_class")
            
            # Check for custom critical error tags (defensive access)
            if hasattr(episode.create_data, 'tags') and episode.create_data.tags:
                critical_tags = {"data_loss", "security_breach", "production_outage"}
                try:
                    episode_tags = {tag.lower() for tag in episode.create_data.tags}
                    if episode_tags & critical_tags:
                        return True
                except (AttributeError, TypeError):
                    logger.warning(f"Episode {episode.episode_id} has invalid tags")
        
        # Check 3: High-performing skill (defensive access)
        if hasattr(episode, 'usage_stats') and episode.usage_stats:
            try:
                if (episode.usage_stats.fix_success_count > 5 and
                    episode.usage_stats.fix_success_rate > 0.75):
                    return True
            except (AttributeError, TypeError):
                logger.warning(f"Episode {episode.episode_id} has invalid usage_stats")
        
        return False
    
    async def _archive_episode(
        self,
        episode: Episode,
        customer_id: str,
        collection: str
    ):
        """
        Archive episode (mark as archived, don't delete).
        
        Archived episodes:
        - Hidden from search by default
        - Still retrievable if needed
        - Can be un-archived
        """
        # TODO: Implement update_episode_metadata in KyroDBRouter
        logger.debug(
            f"Would update episode {episode.episode_id} metadata: archived=true "
            f"(update_episode_metadata not yet implemented)"
        )
    
    async def _delete_episode(
        self,
        episode: Episode,
        customer_id: str,
        collection: str
    ):
        """
        Permanently delete episode.
        
        CAUTION: This is irreversible!
        Only called after permanent protection checks.
        """
        # TODO: Implement delete_episode in KyroDBRouter
        logger.warning(
            f"Would DELETE episode {episode.episode_id} "
            f"(delete_episode not yet implemented)"
        )
    
    async def _fetch_all_episodes(
        self,
        customer_id: str,
        collection: str
    ) -> list[Episode]:
        """
        Fetch all episodes for decay processing.
        
        TODO: Implement bulk_fetch_episodes in KyroDBRouter
        """
        logger.warning(
            f"bulk_fetch_episodes not yet implemented - "
            f"returning empty list for {customer_id}"
        )
        return []
