"""
Tiered reflection generation for cost optimization.

Implements three-tier system:
- CHEAP: Gemini Flash only (~$0.0003/reflection)
- CACHED: Cluster templates (~$0, Phase 6)
- PREMIUM: Multi-perspective consensus (~$0.096/reflection)

Security:
- Input sanitization inherited from base services
- Cost tracking and circuit breakers
- Tier selection logging for audit
- Graceful fallback on failures

Performance:
- Auto-tier selection based on error characteristics
- 90% cost reduction vs all-premium baseline
- Quality gates ensure minimum standards
"""

import asyncio
import json
import logging
import time
import threading
from collections import Counter
from datetime import timezone, datetime
from enum import Enum
from typing import Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

from src.config import LLMConfig
from src.models.episode import EpisodeCreate, Reflection, LLMPerspective, ReflectionTier
from src.ingestion.multi_perspective_reflection import (
    MultiPerspectiveReflectionService,
    PromptInjectionDefense
)
from src.hygiene.clustering import EpisodeClusterer
from src.hygiene.templates import TemplateGenerator
from src.models.clustering import ClusterTemplate

logger = logging.getLogger(__name__)


class CheapReflectionService:
    """
    Single-model reflection using Gemini Flash for cost optimization.
    
    Cost: ~$0.0003 per reflection (320x cheaper than premium)
    Quality: Confidence ~0.6-0.7 (acceptable for non-critical errors)
    
    Security:
    - Input sanitization via PromptInjectionDefense
    - Output validation with strict schema
    - Timeout enforcement
    """
    
    # Same system prompt as multi-perspective for consistency
    SYSTEM_PROMPT = """You are an expert AI assistant analyzing software development failures.

Extract the following in valid JSON format:
{
  "root_cause": "Fundamental reason for failure (not symptoms) - be concise",
  "preconditions": ["Specific condition 1", "Specific condition 2", ...],
  "resolution_strategy": "Step-by-step resolution (be specific and actionable)",
  "environment_factors": ["OS/version/tool that matters"],
  "affected_components": ["Component 1", "Component 2"],
  "generalization_score": 0.5,  // 0.0 to 1.0
  "confidence_score": 0.8,      // 0.0 to 1.0
  "reasoning": "Brief explanation of your analysis"
}

IMPORTANT RULES:
- Be concise and actionable
- Focus on root cause, not symptoms
- Resolution should be step-by-step
- Generalization: 0.0 = very specific, 1.0 = universal pattern
- Confidence: how certain you are about this analysis
- Keep reasoning under 200 words

Return ONLY valid JSON, no markdown."""
    
  
    COST_PER_REFLECTION_USD = 0.0003
    
    def __init__(self, config: LLMConfig):
        """
        Initialize cheap reflection service.
        
        Args:
            config: LLM configuration with Google API key
        """
        self.config = config
        
        # Initialize Gemini Flash client
        if config.google_api_key and GEMINI_AVAILABLE:
            genai.configure(api_key=config.google_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Cheap reflection service initialized with Gemini Flash")
        else:
            self.gemini_model = None
            if not GEMINI_AVAILABLE:
                logger.warning("google-generativeai not installed - cheap reflections disabled")
            else:
                logger.warning("Google API key not configured - cheap reflections disabled")
        
        # Cost tracking (thread-safe)
        self._stats_lock = threading.Lock()
        self.total_cost_usd = 0.0
        self.total_requests = 0
    
    async def generate_reflection(self, episode: EpisodeCreate) -> Reflection:
        """
        Generate reflection using only Gemini Flash.
        
        Args:
            episode: Episode data (will be sanitized)
        
        Returns:
            Reflection with single-model analysis
            
        Raises:
            No exceptions - falls back to heuristic on failure
        """
        start_time = time.perf_counter()
        
        # Security: Sanitize inputs
        sanitized_episode = self._sanitize_episode(episode)
        
        # Check if Gemini available
        if not self.gemini_model:
            logger.warning("Gemini Flash not available, using fallback")
            return self._create_fallback_reflection(sanitized_episode)
        
        # Build prompt
        user_prompt = self._build_user_prompt(sanitized_episode)
        
        # Call Gemini Flash with retry
        try:
            max_retries = self.config.max_retries
            
            for attempt in range(max_retries):
                try:
                    full_prompt = f"{self.SYSTEM_PROMPT}\n\n{user_prompt}"
                    
                    response = await asyncio.to_thread(
                        self.gemini_model.generate_content,
                        full_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=self.config.temperature,
                            max_output_tokens=1000,  # Shorter than premium (cheaper)
                        )
                    )
                    
                    content = response.text
                    
                    # Extract JSON from markdown if present
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]
                    
                    # Security: Validate JSON
                    data = json.loads(content.strip())
                    
                    # Create reflection
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    
                    reflection = Reflection(
                        root_cause=data.get("root_cause", "Unknown"),
                        preconditions=data.get("preconditions", []),
                        resolution_strategy=data.get("resolution_strategy", ""),
                        environment_factors=data.get("environment_factors", []),
                        affected_components=data.get("affected_components", []),
                        generalization_score=float(data.get("generalization_score", 0.5)),
                        confidence_score=float(data.get("confidence_score", 0.6)),
                        llm_model="gemini-1.5-flash",
                        generated_at=datetime.now(timezone.utc),
                        cost_usd=self.COST_PER_REFLECTION_USD,
                        generation_latency_ms=latency_ms,
                        tier=ReflectionTier.CHEAP.value  # Mark tier
                    )
                    
                    # Track metrics (thread-safe)
                    with self._stats_lock:
                        self.total_cost_usd += self.COST_PER_REFLECTION_USD
                        self.total_requests += 1
                    
                    logger.info(
                        f"Cheap reflection generated: confidence={reflection.confidence_score:.2f}, "
                        f"cost=${self.COST_PER_REFLECTION_USD:.4f}, latency={latency_ms:.0f}ms"
                    )
                    
                    return reflection
                
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.error(f"Cheap reflection parsing failed (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    # Fall through to fallback
                
                except Exception as e:
                    logger.error(f"Cheap reflection call failed (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    # Fall through to fallback
            
            # All retries failed
            logger.warning("All Gemini Flash attempts failed, using fallback")
            return self._create_fallback_reflection(sanitized_episode)
        
        except Exception as e:
            logger.error(f"Cheap reflection generation failed: {e}", exc_info=True)
            return self._create_fallback_reflection(sanitized_episode)
    
    def _sanitize_episode(self, episode: EpisodeCreate) -> EpisodeCreate:
        """Security: Sanitize all episode fields."""
        return EpisodeCreate(
            customer_id=episode.customer_id,
            episode_type=episode.episode_type,
            goal=PromptInjectionDefense.sanitize_text(episode.goal, "goal"),
            tool_chain=PromptInjectionDefense.sanitize_list(episode.tool_chain, "tool_chain"),
            actions_taken=PromptInjectionDefense.sanitize_list(episode.actions_taken, "actions_taken"),
            error_trace=PromptInjectionDefense.sanitize_text(episode.error_trace, "error_trace"),
            error_class=episode.error_class,
            code_state_diff=PromptInjectionDefense.sanitize_text(episode.code_state_diff or "", "code_state_diff") if episode.code_state_diff else None,
            environment_info=episode.environment_info,
            screenshot_path=episode.screenshot_path,
            resolution=PromptInjectionDefense.sanitize_text(episode.resolution or "", "resolution") if episode.resolution else None,
            time_to_resolve_seconds=episode.time_to_resolve_seconds,
            tags=PromptInjectionDefense.sanitize_list(episode.tags, "tags"),
            severity=episode.severity,
        )
    
    def _build_user_prompt(self, episode: EpisodeCreate) -> str:
        """Build user prompt from sanitized episode."""
        prompt_parts = [
            f"Goal: {episode.goal}",
            f"\nTool Chain: {' → '.join(episode.tool_chain)}",
            f"\nError Class: {episode.error_class.value}",
            "\nActions Taken:",
        ]
        
        for i, action in enumerate(episode.actions_taken[:20], 1):
            prompt_parts.append(f"  {i}. {action}")
        
        # Error trace (defensive check for None)
        if episode.error_trace:
            prompt_parts.append(f"\nError Trace:\n{episode.error_trace[:2000]}")
        
        if episode.code_state_diff:
            diff = episode.code_state_diff[:2000]
            prompt_parts.append(f"\nCode Diff:\n{diff}")
        
        if episode.environment_info:
            safe_env = {
                str(k)[:100]: str(v)[:500]
                for k, v in list(episode.environment_info.items())[:20]
            }
            env_str = json.dumps(safe_env, indent=2)
            prompt_parts.append(f"\nEnvironment:\n{env_str}")
        
        if episode.resolution:
            prompt_parts.append(f"\nResolution: {episode.resolution[:1000]}")
        
        return "\n".join(prompt_parts)
    
    def _create_fallback_reflection(self, episode: EpisodeCreate) -> Reflection:
        """Create heuristic reflection when LLM fails."""
        logger.warning("Creating fallback reflection for cheap tier")
        
        root_cause = f"{episode.error_class.value} in {episode.tool_chain[0] if episode.tool_chain else 'unknown tool'}"
        
        preconditions = [
            f"Using tool: {episode.tool_chain[0]}" if episode.tool_chain else "Tool unknown",
            f"Error class: {episode.error_class.value}",
        ]
        
        resolution_strategy = (
            episode.resolution
            if episode.resolution
            else "Manual investigation required - check error trace and environment"
        )
        
        return Reflection(
            root_cause=root_cause,
            preconditions=preconditions,
            resolution_strategy=resolution_strategy,
            environment_factors=list(episode.environment_info.keys())[:10] if episode.environment_info else [],
            affected_components=episode.tool_chain[:5] if episode.tool_chain else [],
            generalization_score=0.3,
            confidence_score=0.4,  # Low confidence for fallback
            llm_model="fallback_heuristic",
            generated_at=datetime.now(timezone.utc),
            cost_usd=0.0,
            generation_latency_ms=0.0,
            tier=ReflectionTier.CHEAP.value
        )
    
    def get_stats(self) -> dict:
        """Get service statistics."""
        return {
            "total_cost_usd": self.total_cost_usd,
            "total_requests": self.total_requests,
            "average_cost_per_request": (
                self.total_cost_usd / self.total_requests
                if self.total_requests > 0
                else 0.0
            ),
        }


class TieredReflectionService:
    """
    Orchestrates reflection generation with cost optimization via tiering.
    
    Auto-selects appropriate tier based on episode characteristics:
    - PREMIUM: Critical errors, novel signatures, retries
    - CHEAP: Everything else (90% of episodes)
    - CACHED:  Cluster templates (Phase 6)
    
    Security:
    - Cost tracking per tier
    - Circuit breakers for budget limits
    - Tier selection audit logging
    - Graceful fallback on failures
    """
    
    # Critical error classes that require premium tier
    PREMIUM_ERROR_CLASSES = {
        "data_loss",
        "security_breach",
        "production_outage",
        "corruption",
    }
    
    def __init__(
        self,
        config: LLMConfig,
        kyrodb_router: Optional[object] = None,  # KyroDBRouter
        embedding_service: Optional[object] = None  # EmbeddingService
    ):
        """
        Initialize tiered reflection service.
        
        Args:
            config: LLM configuration
            kyrodb_router: Optional KyroDB router (for Phase 6 clustering)
            embedding_service: Optional embedding service (for cluster matching)
        """
        self.config = config
        
        # Initialize services
        self.cheap_service = CheapReflectionService(config)
        self.premium_service = MultiPerspectiveReflectionService(config)
        
        # Phase 6: Clustering services (optional)
        self.clusterer: Optional[EpisodeClusterer] = None
        self.template_generator: Optional[TemplateGenerator] = None
        self.embedding_service = embedding_service
        
        # Thread-local storage for cached cluster templates (prevents race conditions)
        self._cached_template_local = threading.local()
        
        if kyrodb_router is not None:
            try:
                from src.config import settings
                clustering_config = settings.clustering if hasattr(settings, 'clustering') else None
                
                if clustering_config:
                    self.clusterer = EpisodeClusterer(
                        kyrodb_router=kyrodb_router,
                        min_cluster_size=clustering_config.min_cluster_size,
                        min_samples=clustering_config.min_samples,
                        metric='cosine'
                    )
                    self.template_generator = TemplateGenerator(
                        kyrodb_router=kyrodb_router,
                        reflection_service=self
                    )
                    logger.info("Clustering services initialized for cached tier")
                else:
                    logger.info("Clustering config not found, cached tier disabled")
            except Exception as e:
                logger.warning(f"Failed to initialize clustering services: {e}")
        
        # Cost tracking (thread-safe)
        self._stats_lock = threading.Lock()
        self.total_cost = 0.0
        self.cost_by_tier = {
            ReflectionTier.CHEAP: 0.0,
            ReflectionTier.CACHED: 0.0,
            ReflectionTier.PREMIUM: 0.0,
        }
        self.count_by_tier = {
            ReflectionTier.CHEAP: 0,
            ReflectionTier.CACHED: 0,
            ReflectionTier.PREMIUM: 0,
        }
        
        logger.info("Tiered reflection service initialized")
    
    async def generate_reflection(
        self,
        episode: EpisodeCreate,
        tier: Optional[ReflectionTier] = None
    ) -> Reflection:
        """
        Generate reflection using appropriate tier.
        
        Tier selection priority:
        1. Explicit tier override (for testing/debugging)
        2. Auto-selection based on error characteristics
        3. Fallback to cheap on errors
        
        Args:
            episode: Episode data
            tier: Optional tier override (auto-select if None)
        
        Returns:
            Reflection from selected tier
        """
        # Step 1: Select tier if not specified
        if tier is None:
            tier = await self._select_tier(episode)
        
        logger.info(f"Generating reflection with {tier.value} tier")
        
        # Step 2: Generate based on tier
        try:
            if tier == ReflectionTier.PREMIUM:
                reflection = await self.premium_service.generate_multi_perspective_reflection(episode)
                reflection.tier = ReflectionTier.PREMIUM.value
                logger.info(
                    f"Generated PREMIUM reflection (cost: ${reflection.cost_usd:.4f}, "
                    f"confidence: {reflection.confidence_score:.2f})"
                )
            
            elif tier == ReflectionTier.CHEAP:
                reflection = await self.cheap_service.generate_reflection(episode)
                
                # Quality gate: fall back to premium if confidence too low
                if not self._validate_cheap_quality(reflection):
                    logger.warning(
                        f"Cheap reflection failed quality check (confidence={reflection.confidence_score:.2f}), "
                        f"upgrading to premium"
                    )
                    reflection = await self.premium_service.generate_multi_perspective_reflection(episode)
                    reflection.tier = ReflectionTier.PREMIUM.value
                    tier = ReflectionTier.PREMIUM  # Track as premium for metrics
                else:
                    logger.info(
                        f"Generated CHEAP reflection (cost: ${reflection.cost_usd:.4f}, "
                        f"confidence: {reflection.confidence_score:.2f})"
                    )
            
            elif tier == ReflectionTier.CACHED:
                # Phase 6: Cluster-based cached reflections
                # Use thread-local storage to prevent race conditions
                if self.template_generator and hasattr(self._cached_template_local, 'cluster_template'):
                    cluster_template = self._cached_template_local.cluster_template
                    
                    # Generate episode-specific ID (temporary until persistence)
                    # TODO: Replace with actual episode_id after database persistence
                    import uuid
                    temp_episode_id = abs(hash(str(uuid.uuid4()))) % (10 ** 8)
                    
                    reflection = await self.template_generator.get_cached_reflection(
                        cluster_template=cluster_template,
                        episode_id=temp_episode_id
                    )
                    
                    # Clean up thread-local cache after use
                    delattr(self._cached_template_local, 'cluster_template')
                    
                    # Quality validation for cached reflection
                    if reflection.confidence_score < 0.6:
                        logger.warning(
                            f"Cached reflection quality too low (confidence={reflection.confidence_score:.2f}), "
                            f"falling back to CHEAP tier"
                        )
                        reflection = await self.cheap_service.generate_reflection(episode)
                        tier = ReflectionTier.CHEAP
                    else:
                        logger.info(
                            f"Generated CACHED reflection from cluster {cluster_template.cluster_id} "
                            f"(cost: $0, confidence: {reflection.confidence_score:.2f})"
                        )
                else:
                    logger.warning("CACHED tier selected but no template available, falling back to CHEAP")
                    reflection = await self.cheap_service.generate_reflection(episode)
                    reflection.tier = ReflectionTier.CHEAP.value
                    tier = ReflectionTier.CHEAP
            
            else:
                # Unknown tier - fallback to cheap
                logger.warning(f"Unknown tier {tier}, falling back to CHEAP")
                reflection = await self.cheap_service.generate_reflection(episode)
                reflection.tier = ReflectionTier.CHEAP.value
                tier = ReflectionTier.CHEAP
            
            # Step 3: Track metrics (thread-safe)
            with self._stats_lock:
                self.total_cost += reflection.cost_usd
                self.cost_by_tier[tier] += reflection.cost_usd
                self.count_by_tier[tier] += 1
            
            return reflection
        
        except Exception as e:
            logger.error(f"Reflection generation failed: {e}", exc_info=True)
            # Last resort fallback
            return self.cheap_service._create_fallback_reflection(episode)
    
    async def _select_tier(self, episode: EpisodeCreate) -> ReflectionTier:
        """
        Auto-select reflection tier based on episode characteristics.
        
        Priority:
        1. CACHED: Check if episode matches existing cluster (Phase 6)
        2. PREMIUM: Critical errors, novel signatures
        3. CHEAP: Everything else (default)
        
        Args:
            episode: Episode data
        
        Returns:
            Selected tier
        """
        # Phase 6: Check 0 - Cluster match for CACHED tier
        if self.clusterer and self.embedding_service:
            try:
                # Get episode embedding
                text_content = f"{episode.goal}\n\n{episode.error_trace}"
                episode_embedding = self.embedding_service.embed_text(text_content)
                
                #Check for cluster match
                cluster_template = await self.clusterer.find_matching_cluster(
                    episode_embedding=episode_embedding,
                    customer_id=episode.customer_id,
                    similarity_threshold=0.85
                )
                
                if cluster_template:
                    logger.info(
                        f"Selected CACHED tier (cluster {cluster_template.cluster_id}, "
                        f"similarity: {cluster_template.avg_similarity:.2f})"
                    )
                    # Store template in thread-local storage (prevents race conditions)
                    self._cached_template_local.cluster_template = cluster_template
                    return ReflectionTier.CACHED
            except Exception as e:
                logger.warning(f"Cluster matching failed: {e}, falling back to normal tier selection")
        
        # Check 1: Critical error class
        if episode.error_class.value in self.PREMIUM_ERROR_CLASSES:
            logger.info(f"Selected PREMIUM tier (critical error: {episode.error_class.value})")
            return ReflectionTier.PREMIUM
        
        # Check 2: Novel error signature
        # TODO: Implement novelty detection (requires KyroDB search for similar errors)
        # For now, simplified heuristic based on metadata
        if hasattr(episode, 'metadata') and episode.metadata:
            retry_count = episode.metadata.get("retry_count", 0)
            if retry_count > 0:
                logger.info(f"Selected PREMIUM tier (retry after low confidence, count={retry_count})")
                return ReflectionTier.PREMIUM
        
        # Check 3: Explicit premium request
        if hasattr(episode, 'tags') and episode.tags:
            if "premium_reflection" in [t.lower() for t in episode.tags]:
                logger.info("Selected PREMIUM tier (explicit tag)")
                return ReflectionTier.PREMIUM
        
        # Default: CHEAP tier (90% of episodes)
        logger.info("Selected CHEAP tier (default)")
        return ReflectionTier.CHEAP
    
    def _validate_cheap_quality(self, reflection: Reflection) -> bool:
        """
        Validate quality of cheap reflection.
        
        Quality gates:
        - Confidence >= 0.6
        - Preconditions not empty
        - Resolution not generic/placeholder
        - Root cause not "Unknown"
        
        Returns:
            True if quality acceptable, False if should upgrade to premium
        """
        # Gate 1: Minimum confidence
        if reflection.confidence_score < 0.6:
            logger.warning(f"Quality check failed: confidence too low ({reflection.confidence_score:.2f} < 0.6)")
            return False
        
        # Gate 2: Must have preconditions
        if not reflection.preconditions or len(reflection.preconditions) == 0:
            logger.warning("Quality check failed: no preconditions extracted")
            return False
        
        # Gate 3: Root cause must be meaningful
        if reflection.root_cause.lower() in {"unknown", "error", "failure", "issue"}:
            logger.warning(f"Quality check failed: generic root cause '{reflection.root_cause}'")
            return False
        
        # Gate 4: Resolution must be actionable (not generic placeholder)
        generic_resolutions = {
            "manual investigation required",
            "check error trace",
            "review logs",
            "unknown",
        }
        if reflection.resolution_strategy.lower() in generic_resolutions:
            logger.warning(f"Quality check failed: generic resolution '{reflection.resolution_strategy}'")
            return False
        
        # All gates passed
        return True
    
    def get_stats(self) -> dict:
        """
        Get tiered reflection statistics.
        
        Returns:
            Dict with cost and usage metrics by tier
        """
        total_reflections = sum(self.count_by_tier.values())
        
        # Calculate percentages
        tier_percentages = {
            tier: (count / total_reflections * 100 if total_reflections > 0 else 0.0)
            for tier, count in self.count_by_tier.items()
        }
        
        # Calculate average cost
        avg_cost = (
            self.total_cost / total_reflections
            if total_reflections > 0
            else 0.0
        )
        
        # Calculate savings vs all-premium
        premium_cost_per_reflection = 0.096  # Multi-perspective cost
        if_all_premium_cost = total_reflections * premium_cost_per_reflection
        savings = if_all_premium_cost - self.total_cost
        savings_percentage = (
            (savings / if_all_premium_cost * 100)
            if if_all_premium_cost > 0
            else 0.0
        )
        
        return {
            "total_cost_usd": self.total_cost,
            "total_reflections": total_reflections,
            "average_cost_per_reflection": avg_cost,
            "cost_by_tier": dict(self.cost_by_tier),
            "count_by_tier": dict(self.count_by_tier),
            "percentage_by_tier": tier_percentages,
            "cost_savings_usd": savings,
            "cost_savings_percentage": savings_percentage,
            "cheap_service_stats": self.cheap_service.get_stats(),
            "premium_service_stats": self.premium_service.get_usage_stats(),
        }


# Singleton instance (thread-safe)
_tiered_service: Optional[TieredReflectionService] = None
_tiered_service_lock = threading.Lock()


def get_tiered_reflection_service(
    config: Optional[LLMConfig] = None,
    kyrodb_router: Optional[object] = None,
    embedding_service: Optional[object] = None
) -> TieredReflectionService:
    """
    Get global tiered reflection service (singleton).
    
    IMPORTANT: Clustering must be configured on first call. Parameters are ignored
    on subsequent calls to maintain singleton integrity.
    
    Args:
        config: LLM configuration (defaults to settings.llm)
        kyrodb_router: Optional KyroDB router for Phase 6 clustering (first call only)
        embedding_service: Optional embedding service for cluster matching (first call only)
    
    Returns:
        TieredReflectionService: Singleton instance
    
    Thread Safety:
        Uses double-check locking for thread-safe initialization.
    """
    global _tiered_service
    
    # Fast path: service already initialized
    if _tiered_service is not None:
        # Validate parameters match if provided (warn about ignored parameters)
        if kyrodb_router is not None and _tiered_service.clusterer is None:
            logger.warning(
                "Singleton already initialized without clustering support. "
                "kyrodb_router parameter ignored. Clustering features unavailable."
            )
        if embedding_service is not None and _tiered_service.embedding_service is None:
            logger.warning(
                "Singleton already initialized without embedding service. "
                "embedding_service parameter ignored."
            )
        return _tiered_service
    
    # Slow path: initialize service (thread-safe)
    with _tiered_service_lock:
        # Double-check after acquiring lock
        if _tiered_service is not None:
            return _tiered_service
        
        # Load config if not provided
        if config is None:
            from src.config import settings
            config = settings.llm
        
        # Create service with Phase 6 clustering support
        _tiered_service = TieredReflectionService(
            config=config,
            kyrodb_router=kyrodb_router,
            embedding_service=embedding_service
        )
        
        logger.info("Tiered reflection service singleton initialized")
        return _tiered_service
