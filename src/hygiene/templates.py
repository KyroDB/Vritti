"""
Cluster template generation and management.

Handles creation of reflection templates from episode clusters
and template-based reflection generation for cached tier.

Security:
- Customer isolation
- Template validation
- Usage tracking to prevent abuse

Performance:
- Template caching
- Batch operations
"""

import logging
from datetime import UTC, datetime

from src.kyrodb.router import KyroDBRouter
from src.models.clustering import ClusterInfo, ClusterTemplate
from src.models.episode import Episode, Reflection, ReflectionTier

logger = logging.getLogger(__name__)


class TemplateGenerator:
    """
    Generate and manage cluster reflection templates.
    
    Strategy:
    1. Find best episode in cluster (highest confidence reflection)
    2. If no reflections exist, generate one using PREMIUM tier
    3. Store as template for future reuse
    """
    
    def __init__(
        self,
        kyrodb_router: KyroDBRouter,
        reflection_service: object | None = None  # TieredReflectionService
    ):
        """
        Initialize template generator.
        
        Args:
            kyrodb_router: KyroDB router for data access
            reflection_service: Optional reflection service for generating templates
        """
        self.kyrodb_router = kyrodb_router
        self.reflection_service = reflection_service
    
    async def generate_cluster_template(
        self,
        customer_id: str,
        cluster_info: ClusterInfo
    ) -> ClusterTemplate:
        """
        Generate template reflection for a cluster.
        
        Strategy:
        1. Fetch all episodes in cluster
        2. Find episode with best reflection (highest confidence)
        3. If no reflections, generate one using premium tier
        4. Create and persist template
        
        Args:
            customer_id: Customer ID
            cluster_info: Cluster information
            
        Returns:
            ClusterTemplate
            
        Security:
            - Validates customer_id matches cluster
            - Ensures template quality (confidence >= 0.7)
        """
        if customer_id != cluster_info.customer_id:
            raise ValueError(
                f"Customer ID mismatch: {customer_id} != {cluster_info.customer_id}"
            )
        
        logger.info(
            f"Generating template for cluster {cluster_info.cluster_id} "
            f"({len(cluster_info.episode_ids)} episodes)"
        )
        
        # Fetch episodes in cluster using bulk operation
        episodes = await self._fetch_cluster_episodes(cluster_info)
        
        if not episodes:
            raise ValueError(
                f"No episodes found for cluster {cluster_info.cluster_id}. "
                f"Cluster has {len(cluster_info.episode_ids)} episode IDs."
            )
        
        # Find episode with best reflection (highest confidence)
        best_reflection: Reflection | None = None
        source_episode_id: int | None = None
        
        for episode in episodes:
            if (
                episode.reflection
                and episode.reflection.confidence_score
                and (
                    best_reflection is None
                    or episode.reflection.confidence_score > best_reflection.confidence_score
                )
            ):
                best_reflection = episode.reflection
                source_episode_id = episode.episode_id
        
        # If no good reflection exists, generate one with premium tier
        if best_reflection is None:
            logger.info(
                f"No existing reflections in cluster {cluster_info.cluster_id}, "
                f"generating premium reflection for template"
            )
            best_reflection, source_episode_id = await self._generate_premium_reflection(
                episodes[0]
            )
        
        # Validate template quality
        if best_reflection.confidence_score < 0.7:
            logger.warning(
                f"Template quality low (confidence: {best_reflection.confidence_score:.2f}), "
                f"regenerating with premium tier"
            )
            best_reflection, source_episode_id = await self._generate_premium_reflection(
                episodes[0]
            )
            
            # Validate regenerated reflection
            if best_reflection.confidence_score < 0.7:
                raise ValueError(
                    f"Failed to generate high-quality template for cluster {cluster_info.cluster_id}: "
                    f"confidence {best_reflection.confidence_score:.2f} < 0.7 after premium regeneration"
                )
        
        # Create template
        template = ClusterTemplate(
            cluster_id=cluster_info.cluster_id,
            customer_id=customer_id,
            template_reflection=best_reflection.model_dump(),  # Serialize to dict
            source_episode_id=source_episode_id,
            episode_count=len(cluster_info.episode_ids),
            avg_similarity=cluster_info.avg_intra_cluster_similarity,
            usage_count=0
        )
        
        # Persist template using the cluster centroid embedding for matching.
        persisted = await self._persist_template(
            template, template_embedding=cluster_info.centroid_embedding
        )
        if not persisted:
            raise RuntimeError(
                f"Failed to persist template for cluster {cluster_info.cluster_id} "
                f"(customer: {customer_id})"
            )
        
        logger.info(
            f"Template generated for cluster {cluster_info.cluster_id} "
            f"(source: episode {source_episode_id}, "
            f"confidence: {best_reflection.confidence_score:.2f})"
        )
        
        return template
    
    async def get_cached_reflection(
        self,
        cluster_template: ClusterTemplate,
        episode_id: int
    ) -> Reflection:
        """
        Generate cached reflection from cluster template.
        
        Clones template reflection and updates episode-specific metadata.
        
        Args:
            cluster_template: Cluster template to use
            episode_id: New episode ID
            
        Returns:
            Reflection with tier=cached, cost=0
        """
        # Deserialize template reflection
        template_refl_dict = cluster_template.template_reflection
        template_refl = Reflection.model_validate(template_refl_dict)
        
        # Clone and update
        cached_reflection = template_refl.model_copy(deep=True)
        
        # Mark as cached
        cached_reflection.tier = ReflectionTier.CACHED.value
        cached_reflection.cost_usd = 0.0  # $0 for cached reflections
        cached_reflection.llm_model = f"cluster-template-{cluster_template.cluster_id}"
        cached_reflection.generated_at = datetime.now(UTC)
        
        # Track template usage
        await self._track_template_usage(
            cluster_template.cluster_id,
            cluster_template.customer_id
        )
        
        logger.debug(
            f"Generated cached reflection for episode {episode_id} "
            f"using template {cluster_template.cluster_id}"
        )
        
        return cached_reflection
    
    async def _generate_premium_reflection(
        self,
        episode: Episode
    ) -> tuple[Reflection, int]:
        """
        Generate premium reflection for episode (for template).
        
        Args:
            episode: Episode to generate reflection for
            
        Returns:
            (reflection, episode_id)
        """
        if self.reflection_service is None:
            raise RuntimeError("Reflection service not available for template generation")
        
        # Generate premium reflection
        reflection = await self.reflection_service.generate_reflection(
            episode.create_data,
            episode_id=episode.episode_id,
            tier=ReflectionTier.PREMIUM
        )
        
        return reflection, episode.episode_id
    
    async def _fetch_cluster_episodes(self, cluster_info: ClusterInfo) -> list[Episode]:
        """
        Fetch all episodes for a cluster.
        
        Args:
            cluster_info: Cluster information
        
        Returns:
            list[Episode]: Episodes in cluster
        """
        # Use bulk fetch for efficiency
        try:
            episodes = await self.kyrodb_router.bulk_fetch_episodes(
                episode_ids=cluster_info.episode_ids,
                customer_id=cluster_info.customer_id,
                collection="failures",
            )
            
            logger.info(
                f"Bulk fetched {len(episodes)}/{len(cluster_info.episode_ids)} episodes "
                f"for cluster {cluster_info.cluster_id}"
            )
            
            return episodes
            
        except Exception as e:
            logger.error(
                f"Failed to bulk fetch episodes for cluster {cluster_info.cluster_id}: {e}",
                exc_info=True,
            )
            raise  # Re-raise so caller knows fetch failed vs no episodes found
    
    async def _persist_template(
        self, cluster_template: ClusterTemplate, template_embedding: list[float]
    ) -> bool:
        """
        Persist cluster template to KyroDB for reuse across restarts.
        
        Critical: Without persistence, templates are lost on service restart,
        defeating the purpose of template caching.
        
        Args:
            cluster_template: Template to persist
        
        Returns:
            bool: Success status
        """
        try:
            # Serialize template to metadata
            template_metadata = cluster_template.to_metadata_dict()

            # Persist to KyroDB cluster_templates namespace
            success = await self.kyrodb_router.insert_cluster_template(
                customer_id=cluster_template.customer_id,
                cluster_id=cluster_template.cluster_id,
                template_embedding=template_embedding,
                template_metadata=template_metadata
            )
            
            if success:
                logger.info(
                    f"Persisted template for cluster {cluster_template.cluster_id} "
                    f"(customer: {cluster_template.customer_id}, episodes: {cluster_template.episode_count})"
                )
            else:
                logger.error(
                    f"Failed to persist template for cluster {cluster_template.cluster_id}"
                )
            
            return success
            
        except Exception as e:
            logger.error(
                f"Error persisting template for cluster {cluster_template.cluster_id}: {e}",
                exc_info=True
            )
            return False
    
    async def _track_template_usage(self, cluster_id: int, customer_id: str):
        """
        Increment template usage counter.
        """
        try:
            await self.kyrodb_router.increment_cluster_template_usage(
                customer_id=customer_id,
                cluster_id=cluster_id,
            )
        except Exception as e:
            logger.warning(
                f"Failed to track template usage for cluster {cluster_id} (customer: {customer_id}): {e}",
                exc_info=True,
            )
