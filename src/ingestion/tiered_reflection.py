"""
Tiered reflection generation for cost optimization.

Implements three-tier system:
- CHEAP: OpenRouter free tier model (~$0.00/reflection)
- CACHED: Cluster templates 
- PREMIUM: Multi-perspective consensus via OpenRouter

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

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx

from src.config import LLMConfig
from src.hygiene.clustering import EpisodeClusterer
from src.hygiene.templates import TemplateGenerator
from src.ingestion.multi_perspective_reflection import (
    MultiPerspectiveReflectionService,
    PromptInjectionDefense,
)
from src.models.clustering import ClusterTemplate
from src.models.episode import (
    PRECONDITION_MAX_ITEMS,
    EpisodeCreate,
    Reflection,
    ReflectionTier,
)

if TYPE_CHECKING:
    from src.ingestion.embedding import EmbeddingService
    from src.kyrodb.router import KyroDBRouter

logger = logging.getLogger(__name__)
MAX_PRECONDITIONS = PRECONDITION_MAX_ITEMS

_cached_cluster_template_var: ContextVar[ClusterTemplate | None] = ContextVar(
    "cached_cluster_template", default=None
)


class CheapReflectionService:
    """
    Single-model reflection using OpenRouter free tier for cost optimization.

    Cost: ~$0.00 per reflection (free tier models)
    Quality: Confidence ~0.6-0.7 (acceptable for non-critical errors)

    Security:
    - Input sanitization via PromptInjectionDefense
    - Output validation with strict schema
    - Timeout enforcement
    """

    # Same system prompt as multi-perspective for consistency
    SYSTEM_PROMPT_TEMPLATE = """You are an expert AI assistant analyzing software development failures.

Extract the following in valid JSON format:
{
  "root_cause": "Fundamental reason for failure (not symptoms) - be concise",
  "preconditions": {"key": "value", "...": "..."},
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
- Preconditions MUST be structured key/value context (snake_case keys), max __MAX_PRECONDITIONS__ entries
- Generalization: 0.0 = very specific, 1.0 = universal pattern
- Confidence: how certain you are about this analysis
- Keep reasoning under 200 words

Return ONLY valid JSON, no markdown."""

    SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.replace("__MAX_PRECONDITIONS__", str(MAX_PRECONDITIONS))

    # OpenRouter configuration
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    COST_PER_REFLECTION_USD = 0.0  # Free tier

    def __init__(self, config: LLMConfig):
        """
        Initialize cheap reflection service with OpenRouter.

        Args:
            config: LLM configuration with OpenRouter API key
        """
        self.config = config

        # Get OpenRouter API key from config (LLM_OPENROUTER_API_KEY)
        self.openrouter_api_key = config.openrouter_api_key
        self.model = config.cheap_model
        self.enabled = bool(self.openrouter_api_key)

        if self.enabled:
            logger.info(f"Cheap reflection service initialized with OpenRouter ({self.model})")
        else:
            logger.warning("LLM_OPENROUTER_API_KEY not set - cheap reflections disabled")

        # Cost tracking (thread-safe)
        self._stats_lock = threading.Lock()
        self.total_cost_usd = 0.0
        self.total_requests = 0

    async def generate_reflection(self, episode: EpisodeCreate) -> Reflection:
        """
        Generate reflection using OpenRouter free tier model.

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

        # Check if OpenRouter available
        if not self.enabled:
            logger.warning("OpenRouter not available, using fallback")
            return self._create_fallback_reflection(sanitized_episode)

        # Build prompt
        user_prompt = self._build_user_prompt(sanitized_episode)

        # Call OpenRouter with retry
        try:
            max_retries = self.config.max_retries

            for attempt in range(max_retries):
                try:
                    content = await self._call_openrouter(f"{self.SYSTEM_PROMPT}\n\n{user_prompt}")

                    # Extract JSON from markdown if present
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]

                    # Security: Validate JSON
                    data = json.loads(content.strip())

                    preconditions = data.get("preconditions", {})
                    if isinstance(preconditions, dict) and len(preconditions) > MAX_PRECONDITIONS:
                        logger.warning(
                            "Preconditions exceeded limit; truncating to %d entries",
                            MAX_PRECONDITIONS,
                        )
                        preconditions = dict(list(preconditions.items())[:MAX_PRECONDITIONS])
                    elif not isinstance(preconditions, dict):
                        logger.warning("Preconditions is not a mapping; ignoring value")
                        preconditions = {}

                    # Create reflection
                    latency_ms = (time.perf_counter() - start_time) * 1000

                    reflection = Reflection(
                        root_cause=data.get("root_cause", "Unknown"),
                        preconditions=preconditions,
                        resolution_strategy=data.get("resolution_strategy", ""),
                        environment_factors=data.get("environment_factors", []),
                        affected_components=data.get("affected_components", []),
                        generalization_score=float(data.get("generalization_score", 0.5)),
                        confidence_score=float(data.get("confidence_score", 0.6)),
                        llm_model=self.model,
                        generated_at=datetime.now(UTC),
                        cost_usd=self.COST_PER_REFLECTION_USD,
                        generation_latency_ms=latency_ms,
                        tier=ReflectionTier.CHEAP,  # Mark tier
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
                    logger.error(
                        f"Cheap reflection parsing failed (attempt {attempt+1}/{max_retries}): {e}"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    # Fall through to fallback

                except Exception as e:
                    logger.error(
                        f"Cheap reflection call failed (attempt {attempt+1}/{max_retries}): {e}"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    # Fall through to fallback

            # All retries failed
            logger.warning("All OpenRouter attempts failed, using fallback")
            return self._create_fallback_reflection(sanitized_episode)

        except Exception as e:
            logger.error(f"Cheap reflection generation failed: {e}", exc_info=True)
            return self._create_fallback_reflection(sanitized_episode)

    async def _call_openrouter(self, prompt: str) -> str:
        """
        Call OpenRouter API.

        Args:
            prompt: Full prompt text

        Returns:
            Response content string
        """
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/episodic-memory",
            "X-Title": "EpisodicMemory Cheap Reflection",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": 1000,  # Shorter than premium
        }

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                f"{self.OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("OpenRouter response missing choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise ValueError("OpenRouter response choice has invalid type")
        message = first.get("message")
        if not isinstance(message, dict):
            raise ValueError("OpenRouter response missing message")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("OpenRouter response missing string content")
        return content

    def _sanitize_episode(self, episode: EpisodeCreate) -> EpisodeCreate:
        """Security: Sanitize all episode fields."""
        return EpisodeCreate(
            customer_id=episode.customer_id,
            episode_type=episode.episode_type,
            goal=PromptInjectionDefense.sanitize_text(episode.goal, "goal"),
            tool_chain=PromptInjectionDefense.sanitize_list(episode.tool_chain, "tool_chain"),
            actions_taken=PromptInjectionDefense.sanitize_list(
                episode.actions_taken, "actions_taken"
            ),
            error_trace=PromptInjectionDefense.sanitize_text(episode.error_trace, "error_trace"),
            error_class=episode.error_class,
            code_state_diff=(
                PromptInjectionDefense.sanitize_text(
                    episode.code_state_diff or "", "code_state_diff"
                )
                if episode.code_state_diff
                else None
            ),
            environment_info=episode.environment_info,
            screenshot_base64=None,  # Never send raw image data to LLM
            resolution=(
                PromptInjectionDefense.sanitize_text(episode.resolution or "", "resolution")
                if episode.resolution
                else None
            ),
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
                str(k)[:100]: str(v)[:500] for k, v in list(episode.environment_info.items())[:20]
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

        preconditions = {
            "tool": episode.tool_chain[0] if episode.tool_chain else "unknown",
            "error_class": episode.error_class.value,
        }

        resolution_strategy = (
            episode.resolution
            if episode.resolution
            else "Manual investigation required - check error trace and environment"
        )

        return Reflection(
            root_cause=root_cause,
            preconditions=preconditions,
            resolution_strategy=resolution_strategy,
            environment_factors=(
                list(episode.environment_info.keys())[:10] if episode.environment_info else []
            ),
            affected_components=episode.tool_chain[:5] if episode.tool_chain else [],
            generalization_score=0.3,
            confidence_score=0.4,  # Low confidence for fallback
            llm_model="fallback_heuristic",
            generated_at=datetime.now(UTC),
            cost_usd=0.0,
            generation_latency_ms=0.0,
            tier=ReflectionTier.CHEAP,
        )

    def get_stats(self) -> dict:
        """Get service statistics."""
        return {
            "total_cost_usd": self.total_cost_usd,
            "total_requests": self.total_requests,
            "average_cost_per_request": (
                self.total_cost_usd / self.total_requests if self.total_requests > 0 else 0.0
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

    Budget Controls :
    - Daily cost tracking with automatic reset at midnight UTC
    - Warning alert at $10/day threshold
    - Hard limit at $50/day - premium tier blocked
    - Cheap tier always available (even when over budget)
    """

    # Critical error classes that require premium tier
    PREMIUM_ERROR_CLASSES = {
        "data_loss",
        "security_breach",
        "production_outage",
        "corruption",
    }

    # Daily budget thresholds (USD)
    DAILY_COST_WARNING_USD = 10.0  # Log warning
    DAILY_COST_LIMIT_USD = 50.0  # Block premium tier

    def __init__(
        self,
        config: LLMConfig,
        kyrodb_router: KyroDBRouter | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        """
        Initialize tiered reflection service.

        Args:
            config: LLM configuration
            kyrodb_router: Optional KyroDB router (for Phase 6 clustering)
            embedding_service: Optional embedding service (for cluster matching)
        """
        self.config = config
        self.llm_config = config  # Store for health check access

        # Initialize services
        self.cheap_service = CheapReflectionService(config)
        self.premium_service = MultiPerspectiveReflectionService(config)

        # Phase 6: Clustering services (optional)
        self.clusterer: EpisodeClusterer | None = None
        self.template_generator: TemplateGenerator | None = None
        self.embedding_service = embedding_service
        self._clustering_config = None

        if kyrodb_router is not None:
            try:
                from src.config import get_settings

                clustering_config = get_settings().clustering
                self._clustering_config = clustering_config

                if not clustering_config.enabled:
                    logger.info("Clustering disabled (CLUSTERING_ENABLED=false)")
                elif self.embedding_service is None:
                    logger.warning(
                        "Clustering enabled but embedding_service not provided; cached tier disabled"
                    )
                else:
                    self.clusterer = EpisodeClusterer(
                        kyrodb_router=kyrodb_router,
                        min_cluster_size=clustering_config.min_cluster_size,
                        min_samples=clustering_config.min_samples,
                        metric=clustering_config.metric,
                        cluster_cache_ttl_seconds=clustering_config.cluster_cache_ttl_seconds,
                    )
                    self.template_generator = TemplateGenerator(
                        kyrodb_router=kyrodb_router,
                        reflection_service=self,
                    )
                    logger.info("Clustering services initialized for cached tier")
            except Exception as e:
                logger.warning(f"Failed to initialize clustering services: {e}", exc_info=True)

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

        # Daily cost tracking
        self._daily_cost_usd = 0.0
        self._daily_cost_date = datetime.now(UTC).date()
        self._daily_warning_logged = False
        self._daily_limit_logged = False

        logger.info("Tiered reflection service initialized")

    async def generate_reflection(
        self, episode: EpisodeCreate, *, episode_id: int, tier: ReflectionTier | None = None
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
                reflection = await self.premium_service.generate_multi_perspective_reflection(
                    episode
                )
                reflection.tier = ReflectionTier.PREMIUM
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
                    reflection = await self.premium_service.generate_multi_perspective_reflection(
                        episode
                    )
                    reflection.tier = ReflectionTier.PREMIUM
                    tier = ReflectionTier.PREMIUM  # Track as premium for metrics
                else:
                    logger.info(
                        f"Generated CHEAP reflection (cost: ${reflection.cost_usd:.4f}, "
                        f"confidence: {reflection.confidence_score:.2f})"
                    )

            elif tier == ReflectionTier.CACHED:
                # Phase 6: Cluster-based cached reflections
                # Use contextvars for per-request isolation across concurrent async tasks.
                cluster_template = _cached_cluster_template_var.get()
                _cached_cluster_template_var.set(None)  # one-shot consumption

                if self.template_generator and cluster_template is not None:
                    reflection = await self.template_generator.get_cached_reflection(
                        cluster_template=cluster_template, episode_id=episode_id
                    )

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
                    logger.warning(
                        "CACHED tier selected but no template available, falling back to CHEAP"
                    )
                    reflection = await self.cheap_service.generate_reflection(episode)
                    reflection.tier = ReflectionTier.CHEAP
                    tier = ReflectionTier.CHEAP

            else:
                # Unknown tier - fallback to cheap
                logger.warning(f"Unknown tier {tier}, falling back to CHEAP")
                reflection = await self.cheap_service.generate_reflection(episode)
                reflection.tier = ReflectionTier.CHEAP
                tier = ReflectionTier.CHEAP

            # Step 3: Track metrics (thread-safe)
            with self._stats_lock:
                self.total_cost += reflection.cost_usd
                self.cost_by_tier[tier] += reflection.cost_usd
                self.count_by_tier[tier] += 1

                # Daily cost tracking
                self._track_daily_cost(reflection.cost_usd)

            return reflection

        except Exception as e:
            logger.error(f"Reflection generation failed: {e}", exc_info=True)
            # Last resort fallback
            return self.cheap_service._create_fallback_reflection(episode)

    async def _select_tier(self, episode: EpisodeCreate) -> ReflectionTier:
        """
        Auto-select reflection tier based on episode characteristics.

        Priority:
        1. Budget check: If daily budget exceeded, force CHEAP tier
        2. CACHED: Check if episode matches existing cluster (Phase 6)
        3. PREMIUM: Critical errors, novel signatures
        4. CHEAP: Everything else (default)

        Args:
            episode: Episode data

        Returns:
            Selected tier
        """
        #  Budget check - if daily limit exceeded, force CHEAP tier
        if self._is_daily_budget_exceeded():
            logger.warning(
                f"Daily budget exceeded (${self._daily_cost_usd:.2f} > ${self.DAILY_COST_LIMIT_USD}), "
                f"forcing CHEAP tier for episode"
            )
            return ReflectionTier.CHEAP

        # Phase 6: Check 0 - Cluster match for CACHED tier
        if self.clusterer and self.embedding_service:
            try:
                # Get episode embedding
                text_content = f"{episode.goal}\n\n{episode.error_trace}"
                episode_embedding = await self.embedding_service.embed_text_async(text_content)

                # Check for cluster match
                if not episode.customer_id:
                    raise ValueError("customer_id is required for cached tier matching")

                similarity_threshold = (
                    self._clustering_config.template_match_min_similarity
                    if self._clustering_config is not None
                    else 0.85
                )
                match_k = (
                    self._clustering_config.template_match_k
                    if self._clustering_config is not None
                    else 5
                )
                cluster_template = await self.clusterer.find_matching_cluster(
                    episode_embedding=episode_embedding,
                    customer_id=episode.customer_id,
                    similarity_threshold=similarity_threshold,
                    k=match_k,
                )

                if cluster_template:
                    match_similarity = cluster_template.match_similarity
                    logger.info(
                        f"Selected CACHED tier (cluster {cluster_template.cluster_id}, "
                        f"match_similarity: {(match_similarity if match_similarity is not None else 0.0):.2f})"
                    )
                    # Store template in a task-local ContextVar (safe under asyncio concurrency).
                    _cached_cluster_template_var.set(cluster_template)
                    return ReflectionTier.CACHED
            except Exception as e:
                logger.warning(
                    f"Cluster matching failed: {e}, falling back to normal tier selection"
                )

        # Check 1: Critical error class
        if episode.error_class.value in self.PREMIUM_ERROR_CLASSES:
            logger.info(f"Selected PREMIUM tier (critical error: {episode.error_class.value})")
            return ReflectionTier.PREMIUM

        # Check 2: Retry hint from environment metadata
        # Agents can pass retry_count in environment_info to force deeper analysis.
        retry_count_raw = episode.environment_info.get("retry_count", 0)
        try:
            retry_count = int(retry_count_raw)
        except (TypeError, ValueError):
            retry_count = 0
        if retry_count > 0:
            logger.info(f"Selected PREMIUM tier (retry after low confidence, count={retry_count})")
            return ReflectionTier.PREMIUM

        # Check 3: Explicit premium request
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
            logger.warning(
                f"Quality check failed: confidence too low ({reflection.confidence_score:.2f} < 0.6)"
            )
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
            logger.warning(
                f"Quality check failed: generic resolution '{reflection.resolution_strategy}'"
            )
            return False

        # All gates passed
        return True

    def _track_daily_cost(self, cost_usd: float) -> None:
        """
        Track daily cost and trigger alerts when thresholds are exceeded.

        Called within _stats_lock, so no additional locking needed.

        Args:
            cost_usd: Cost of current reflection in USD
        """
        current_date = datetime.now(UTC).date()

        # Reset daily tracking if date changed (midnight UTC)
        if current_date != self._daily_cost_date:
            logger.info(
                f"Daily cost reset: previous day spent ${self._daily_cost_usd:.4f}, "
                f"starting new day {current_date}"
            )
            self._daily_cost_usd = 0.0
            self._daily_cost_date = current_date
            self._daily_warning_logged = False
            self._daily_limit_logged = False

        # Add current cost
        self._daily_cost_usd += cost_usd

        # Warning threshold ($10/day)
        if self._daily_cost_usd >= self.DAILY_COST_WARNING_USD and not self._daily_warning_logged:
            logger.warning(
                f"COST WARNING: Daily reflection cost reached ${self._daily_cost_usd:.2f} "
                f"(threshold: ${self.DAILY_COST_WARNING_USD}). "
                f"Consider reviewing tier selection."
            )
            self._daily_warning_logged = True

        # Hard limit threshold ($50/day)
        if self._daily_cost_usd >= self.DAILY_COST_LIMIT_USD and not self._daily_limit_logged:
            logger.error(
                f"COST LIMIT EXCEEDED: Daily reflection cost reached ${self._daily_cost_usd:.2f} "
                f"(limit: ${self.DAILY_COST_LIMIT_USD}). "
                f"Premium tier will be blocked until midnight UTC."
            )
            self._daily_limit_logged = True

            # Intentionally no external telemetry here; logs are the source of truth.

    def _is_daily_budget_exceeded(self) -> bool:
        """
        Check if daily budget limit has been exceeded.

        Thread-safe: Acquires stats lock for reading.

        Returns:
            True if daily cost >= DAILY_COST_LIMIT_USD
        """
        with self._stats_lock:
            # Check if date changed (reset if needed)
            current_date = datetime.now(UTC).date()
            if current_date != self._daily_cost_date:
                return False  # New day, budget reset

            return self._daily_cost_usd >= self.DAILY_COST_LIMIT_USD

    def get_daily_cost_stats(self) -> dict:
        """
        Get daily cost statistics.

        Thread-safe: Acquires stats lock.

        Returns:
            Dict with daily cost metrics
        """
        with self._stats_lock:
            current_date = datetime.now(UTC).date()

            # Check if date changed
            if current_date != self._daily_cost_date:
                return {
                    "date": str(current_date),
                    "daily_cost_usd": 0.0,
                    "warning_threshold_usd": self.DAILY_COST_WARNING_USD,
                    "limit_threshold_usd": self.DAILY_COST_LIMIT_USD,
                    "warning_triggered": False,
                    "limit_exceeded": False,
                    "budget_remaining_usd": self.DAILY_COST_LIMIT_USD,
                }

            return {
                "date": str(self._daily_cost_date),
                "daily_cost_usd": self._daily_cost_usd,
                "warning_threshold_usd": self.DAILY_COST_WARNING_USD,
                "limit_threshold_usd": self.DAILY_COST_LIMIT_USD,
                "warning_triggered": self._daily_warning_logged,
                "limit_exceeded": self._daily_limit_logged,
                "budget_remaining_usd": max(0, self.DAILY_COST_LIMIT_USD - self._daily_cost_usd),
            }

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
        avg_cost = self.total_cost / total_reflections if total_reflections > 0 else 0.0

        # Calculate savings vs all-premium
        premium_cost_per_reflection = 0.096  # Multi-perspective cost
        if_all_premium_cost = total_reflections * premium_cost_per_reflection
        savings = if_all_premium_cost - self.total_cost
        savings_percentage = (
            (savings / if_all_premium_cost * 100) if if_all_premium_cost > 0 else 0.0
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
            "daily_cost": self.get_daily_cost_stats(),  # Day 5: Add daily cost stats
            "cheap_service_stats": self.cheap_service.get_stats(),
            "premium_service_stats": self.premium_service.get_usage_stats(),
        }


# Singleton instance (thread-safe)
_tiered_service: TieredReflectionService | None = None
_tiered_service_lock = threading.Lock()


def get_tiered_reflection_service(
    config: LLMConfig | None = None,
    kyrodb_router: KyroDBRouter | None = None,
    embedding_service: EmbeddingService | None = None,
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
            from src.config import get_settings

            config = get_settings().llm

        # Create service with Phase 6 clustering support
        _tiered_service = TieredReflectionService(
            config=config, kyrodb_router=kyrodb_router, embedding_service=embedding_service
        )

        logger.info("Tiered reflection service singleton initialized")
        return _tiered_service
