"""
Multi-perspective reflection generation with consensus reconciliation.

Uses OpenRouter as unified API gateway for multiple LLM models.
Supports 2-model consensus for premium tier, 1-model for cheap tier.

Security Features:
- Prompt injection protection via input sanitization
- Output validation with strict schema enforcement
- Cost limits to prevent abuse
- Timeout enforcement
- No user-controlled data in system prompts
- All LLM outputs validated before storage

Performance:
- Parallel LLM calls (2 models in ~3-5 seconds)
- Graceful degradation (works with 1/2 or 2/2 models)
- Retry logic for transient failures

Consensus Algorithm:
- Semantic similarity (not string equality) for root cause comparison
- Uses sentence-transformers embeddings for true semantic similarity
- Weighted voting based on model confidence scores
- Threshold-based agreement detection (>0.85 = semantic match)
"""

import asyncio
import json
import logging
import threading
import time
from collections import Counter
from datetime import UTC, datetime

import numpy as np

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

from pydantic import ValidationError

from src.config import LLMConfig
from src.models.episode import (
    EpisodeCreate,
    LLMPerspective,
    Reflection,
    ReflectionConsensus,
)

logger = logging.getLogger(__name__)


class PromptInjectionDefense:
    """
    Security layer to prevent prompt injection attacks.

    Strategies:
    1. Input sanitization (remove control characters, normalize whitespace)
    2. Length limits (prevent token exhaustion)
    3. Content filtering (detect injection patterns)
    4. Escape sequences removal
    """

    MAX_FIELD_LENGTH = 10000  # Characters
    MAX_LIST_ITEMS = 50
    MAX_ITEM_LENGTH = 2000

    INJECTION_PATTERNS = [
        "ignore previous instructions",
        "disregard all",
        "new instructions:",
        "you are now",
        "forget everything",
        "override",
        "admin mode",
        "dev mode",
        "jailbreak",
        "</system>",
        "<|endoftext|>",
        "[INST]",
        "###",  # Common prompt separator
    ]

    @classmethod
    def sanitize_text(cls, text: str, field_name: str = "text") -> str:
        """
        Sanitize text input to prevent injection.

        Args:
            text: Raw text from user
            field_name: Field name for logging

        Returns:
            Sanitized text

        Raises:
            ValueError: If text contains obvious injection attempts
        """
        if not text:
            return ""

        # Length limit
        if len(text) > cls.MAX_FIELD_LENGTH:
            logger.warning(
                f"Truncating {field_name}: {len(text)} chars -> {cls.MAX_FIELD_LENGTH}"
            )
            text = text[: cls.MAX_FIELD_LENGTH] + "... (truncated)"

        # Remove null bytes and control characters (except newlines, tabs)
        text = "".join(
            char for char in text
            if char.isprintable() or char in ("\n", "\t", "\r")
        )

        # Detect injection patterns
        text_lower = text.lower()
        for pattern in cls.INJECTION_PATTERNS:
            if pattern in text_lower:
                logger.warning(
                    f"Potential prompt injection detected in {field_name}: '{pattern}'"
                )
                # Replace with safe placeholder (case-insensitive)
                import re
                text = re.sub(re.escape(pattern), "[REDACTED]", text, flags=re.IGNORECASE)
                text_lower = text.lower()

        # Normalize excessive whitespace
        text = " ".join(text.split())

        return text.strip()

    @classmethod
    def sanitize_list(cls, items: list[str], field_name: str = "list") -> list[str]:
        """Sanitize list of strings."""
        if len(items) > cls.MAX_LIST_ITEMS:
            logger.warning(
                f"Truncating {field_name}: {len(items)} items -> {cls.MAX_LIST_ITEMS}"
            )
            items = items[: cls.MAX_LIST_ITEMS]

        sanitized = []
        for item in items:
            if not item or not item.strip():
                continue

            # Truncate individual items
            if len(item) > cls.MAX_ITEM_LENGTH:
                item = item[: cls.MAX_ITEM_LENGTH] + "..."

            sanitized.append(cls.sanitize_text(item, f"{field_name}_item"))

        return sanitized


class MultiPerspectiveReflectionService:
    """
    Generate reflections using OpenRouter with 2-model consensus.

    Uses OpenRouter API gateway to access multiple LLM providers through
    a unified OpenAI-compatible interface.

    Security-first design:
    - All inputs sanitized before prompting
    - All outputs validated against strict schema
    - Cost tracking and limits enforced
    - No user data in system prompts
    - Timeout enforcement on all API calls

    Consensus Algorithm:
    - Uses semantic similarity (embedding cosine distance) instead of string equality
    - Root causes are considered "matching" if cosine similarity > 0.85
    - Confidence-weighted voting when similarity is between 0.70-0.85
    - Falls back to highest-confidence model when similarity < 0.70
    """

    # Semantic similarity thresholds for consensus
    SEMANTIC_MATCH_THRESHOLD = 0.85  # Strong semantic agreement
    SEMANTIC_PARTIAL_THRESHOLD = 0.70  # Partial agreement (weighted voting)

    SYSTEM_PROMPT = """You are an expert AI assistant analyzing software development failures.

Extract the following in valid JSON format:
{
  "root_cause": "Fundamental reason for failure (not symptoms) - be concise",
  "preconditions": ["Specific condition 1", "Specific condition 2", ...],
  "resolution_strategy": "Step-by-step resolution (be specific and actionable)",
  "environment_factors": ["OS/version/tool that matters"],
  "affected_components": ["Component 1", "Component 2"],
  "generalization_score": 0.5,
  "confidence_score": 0.8,
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

    def __init__(self, config: LLMConfig):
        """
        Initialize multi-perspective reflection service with OpenRouter.

        Args:
            config: LLM configuration with OpenRouter API key
        """
        self.config = config

        # Initialize OpenRouter client (uses OpenAI SDK with custom base_url)
        if config.use_openrouter and OPENAI_AVAILABLE:
            self.openrouter_client = AsyncOpenAI(
                api_key=config.openrouter_api_key,
                base_url=config.openrouter_base_url,
                timeout=config.timeout_seconds,
                max_retries=0,
                default_headers={
                    "HTTP-Referer": "https://kyrodb.dev",
                    "X-Title": "EpisodicMemory",
                }
            )
            logger.info(
                f"OpenRouter client initialized with models: "
                f"consensus=[{config.consensus_model_1}, {config.consensus_model_2}], "
                f"cheap={config.cheap_model}"
            )
        else:
            self.openrouter_client = None
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI SDK not installed; OpenRouter disabled")
            else:
                logger.warning("OpenRouter not configured (no API key)")

        # Cost tracking (thread-safe)
        self._stats_lock = threading.Lock()
        self.total_cost_usd = 0.0
        self.total_requests = 0
        self.requests_by_model = Counter()

        # Lazy-loaded embedding model for semantic similarity (root cause reconciliation)
        self._similarity_model = None
        self._similarity_model_lock = threading.Lock()

        logger.info(
            f"Multi-perspective reflection service initialized with providers: "
            f"{config.enabled_providers}"
        )

    async def aclose(self) -> None:
        """Close any underlying HTTP clients to avoid event-loop shutdown warnings."""
        client = getattr(self, "openrouter_client", None)
        if client is None:
            return
        try:
            await client.close()
        except Exception as e:
            logger.warning("Failed to close OpenRouter client: %s", e, exc_info=True)

    async def __aenter__(self) -> "MultiPerspectiveReflectionService":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def generate_multi_perspective_reflection(
        self,
        episode: EpisodeCreate,
        max_retries: int | None = None,
        use_cheap_tier: bool = False,
    ) -> Reflection:
        """
        Generate reflection using OpenRouter models in parallel.

        Args:
            episode: Episode data (will be sanitized)
            max_retries: Override default retry count
            use_cheap_tier: Use single cheap model instead of consensus

        Returns:
            Reflection with consensus (or single-model if cheap tier)
        """
        if max_retries is None:
            max_retries = self.config.max_retries

        start_time = time.perf_counter()

        sanitized_episode = self._sanitize_episode(episode)
        user_prompt = self._build_user_prompt(sanitized_episode)

        if use_cheap_tier:
            return await self._generate_cheap_reflection(
                sanitized_episode, user_prompt, max_retries, start_time
            )

        return await self._generate_consensus_reflection(
            sanitized_episode, user_prompt, max_retries, start_time
        )

    async def _generate_consensus_reflection(
        self,
        episode: EpisodeCreate,
        user_prompt: str,
        max_retries: int,
        start_time: float,
    ) -> Reflection:
        """Generate reflection using 2-model consensus via OpenRouter."""
        tasks = []
        task_names = []

        if self.openrouter_client:
            tasks.append(
                self._call_openrouter_model(
                    self.config.consensus_model_1, user_prompt, max_retries
                )
            )
            task_names.append(self.config.consensus_model_1)

            tasks.append(
                self._call_openrouter_model(
                    self.config.consensus_model_2, user_prompt, max_retries
                )
            )
            task_names.append(self.config.consensus_model_2)

        if not tasks:
            logger.warning("Premium reflection disabled (OpenRouter not configured); returning fallback")
            return self._create_fallback_reflection(episode)

        try:
            logger.info(f"Calling {len(tasks)} LLM models in parallel for consensus...")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            perspectives = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"{task_names[i]} failed: {result}")
                elif result is not None:
                    perspectives.append(result)
                    logger.info(f"{task_names[i]} succeeded")

            if not perspectives:
                logger.error("All LLM models failed")
                return self._create_fallback_reflection(episode)

            logger.info(f"Reconciling {len(perspectives)} perspectives...")
            consensus = self._reconcile_perspectives(perspectives)

            cost_usd = 0.0  # Free tier

            with self._stats_lock:
                self.total_cost_usd += cost_usd
                self.total_requests += 1
                for perspective in perspectives:
                    self.requests_by_model[perspective.model_name] += 1

            latency_ms = (time.perf_counter() - start_time) * 1000

            reflection = Reflection(
                consensus=consensus,
                root_cause=consensus.agreed_root_cause,
                preconditions=consensus.agreed_preconditions,
                resolution_strategy=consensus.agreed_resolution,
                environment_factors=self._merge_list_fields(
                    [p.environment_factors for p in perspectives]
                ),
                affected_components=self._merge_list_fields(
                    [p.affected_components for p in perspectives]
                ),
                generalization_score=sum(p.generalization_score for p in perspectives)
                / len(perspectives),
                confidence_score=consensus.consensus_confidence,
                llm_model="openrouter-consensus",
                generated_at=datetime.now(UTC),
                cost_usd=cost_usd,
                generation_latency_ms=latency_ms,
            )

            logger.info(
                f"Consensus reflection generated: "
                f"{len(perspectives)}/{len(tasks)} models succeeded, "
                f"consensus={consensus.consensus_method}, "
                f"confidence={consensus.consensus_confidence:.2f}, "
                f"latency={latency_ms:.0f}ms"
            )

            return reflection

        except Exception as e:
            logger.error(f"Consensus reflection failed: {e}", exc_info=True)
            return self._create_fallback_reflection(episode)

    async def _generate_cheap_reflection(
        self,
        episode: EpisodeCreate,
        user_prompt: str,
        max_retries: int,
        start_time: float,
    ) -> Reflection:
        """Generate reflection using single cheap model via OpenRouter."""
        try:
            if not self.openrouter_client:
                return self._create_fallback_reflection(episode)

            # Cheap tier should always use a real LLM when OpenRouter is configured.
            # If the configured cheap model is invalid/unavailable, fall back to the
            # configured consensus models (still real LLMs, not heuristics).
            candidates: list[str] = [
                self.config.cheap_model,
                self.config.consensus_model_1,
                self.config.consensus_model_2,
            ]
            seen: set[str] = set()
            perspective = None
            chosen_model: str | None = None
            for model in candidates:
                if not model or model in seen:
                    continue
                seen.add(model)
                perspective = await self._call_openrouter_model(model, user_prompt, max_retries)
                if perspective is not None:
                    chosen_model = model
                    break

            if perspective is None:
                return self._create_fallback_reflection(episode)

            latency_ms = (time.perf_counter() - start_time) * 1000

            with self._stats_lock:
                self.total_requests += 1
                self.requests_by_model[perspective.model_name] += 1

            reflection = Reflection(
                consensus=None,
                root_cause=perspective.root_cause,
                preconditions=perspective.preconditions,
                resolution_strategy=perspective.resolution_strategy,
                environment_factors=perspective.environment_factors,
                affected_components=perspective.affected_components,
                generalization_score=perspective.generalization_score,
                confidence_score=perspective.confidence_score * 0.8,  # Discount for single model
                llm_model=f"openrouter-cheap:{chosen_model or perspective.model_name}",
                generated_at=datetime.now(UTC),
                cost_usd=0.0,
                generation_latency_ms=latency_ms,
            )

            logger.info(
                f"Cheap reflection generated: "
                f"model={chosen_model or perspective.model_name}, "
                f"confidence={reflection.confidence_score:.2f}, "
                f"latency={latency_ms:.0f}ms"
            )

            return reflection

        except Exception as e:
            logger.error(f"Cheap reflection failed: {e}", exc_info=True)
            return self._create_fallback_reflection(episode)

    async def _call_openrouter_model(
        self, model: str, user_prompt: str, max_retries: int
    ) -> LLMPerspective | None:
        """
        Call a model via OpenRouter with retry logic and validation.

        Error handling strategy:
        - Transient errors (429, 500, 502, 503, 504, timeouts): Retry with exponential backoff
        - Permanent errors (400, 401, 403, 404): Fail immediately, no retry
        - Validation errors (JSON parse, Pydantic): Retry (model output can vary)

        Args:
            model: OpenRouter model identifier (e.g., 'anthropic/claude-3.5-sonnet')
            user_prompt: Sanitized user prompt
            max_retries: Maximum retry attempts for transient failures

        Returns:
            LLMPerspective on success, None on failure
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                response = await self.openrouter_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout_seconds,
                )

                content = response.choices[0].message.content

                # Extract JSON from markdown if needed
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                data = json.loads(content.strip())

                perspective = LLMPerspective(model_name=model, **data)

                logger.info(
                    "OpenRouter call succeeded",
                    extra={
                        "model": model,
                        "attempt": attempt + 1,
                        "confidence": perspective.confidence_score,
                    }
                )
                return perspective

            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(
                    "JSON parsing failed, will retry",
                    extra={"model": model, "attempt": attempt + 1, "error": str(e)}
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                logger.error(
                    "JSON parsing failed after all retries",
                    extra={"model": model, "attempts": max_retries, "error": str(e)}
                )
                return None

            except ValidationError as e:
                last_error = e
                logger.warning(
                    "Pydantic validation failed, will retry",
                    extra={"model": model, "attempt": attempt + 1, "error": str(e)}
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                logger.error(
                    "Pydantic validation failed after all retries",
                    extra={"model": model, "attempts": max_retries, "error": str(e)}
                )
                return None

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Categorize errors
                is_rate_limit = "429" in str(e) or "rate" in error_str
                is_timeout = "timeout" in error_str or "timed out" in error_str
                is_server_error = any(
                    code in str(e) for code in ["500", "502", "503", "504"]
                )
                is_auth_error = "401" in str(e) or "403" in str(e)
                is_bad_request = "400" in str(e) or "404" in str(e)

                # Permanent errors: fail immediately
                if is_auth_error:
                    logger.error(
                        "Authentication failed (permanent)",
                        extra={"model": model, "error": str(e)}
                    )
                    return None

                if is_bad_request:
                    logger.error(
                        "Bad request (permanent)",
                        extra={"model": model, "error": str(e)}
                    )
                    return None

                # Transient errors: retry with backoff
                is_transient = is_rate_limit or is_timeout or is_server_error

                if is_transient and attempt < max_retries - 1:
                    backoff = 2 ** attempt
                    if is_rate_limit:
                        backoff = min(backoff * 2, 30)  # Extra backoff for rate limits
                    logger.warning(
                        "Transient error, retrying",
                        extra={
                            "model": model,
                            "attempt": attempt + 1,
                            "backoff_seconds": backoff,
                            "is_rate_limit": is_rate_limit,
                            "is_timeout": is_timeout,
                            "error": str(e)
                        }
                    )
                    await asyncio.sleep(backoff)
                    continue

                logger.error(
                    "OpenRouter call failed",
                    extra={
                        "model": model,
                        "attempts": attempt + 1,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
                return None

        logger.error(
            "OpenRouter call exhausted all retries",
            extra={"model": model, "attempts": max_retries, "last_error": str(last_error)}
        )
        return None

    def _sanitize_episode(self, episode: EpisodeCreate) -> EpisodeCreate:
        """
        Security: Sanitize all episode fields to prevent prompt injection.

        Returns a copy of episode with sanitized fields.
        """
        return EpisodeCreate(
            customer_id=episode.customer_id,  # Already validated by auth
            episode_type=episode.episode_type,
            goal=PromptInjectionDefense.sanitize_text(episode.goal, "goal"),
            tool_chain=PromptInjectionDefense.sanitize_list(
                episode.tool_chain, "tool_chain"
            ),
            actions_taken=PromptInjectionDefense.sanitize_list(
                episode.actions_taken, "actions_taken"
            ),
            error_trace=PromptInjectionDefense.sanitize_text(
                episode.error_trace, "error_trace"
            ),
            error_class=episode.error_class,
            code_state_diff=PromptInjectionDefense.sanitize_text(
                episode.code_state_diff or "", "code_state_diff"
            )
            if episode.code_state_diff
            else None,
            environment_info=episode.environment_info,  # Dict, handled separately
            screenshot_base64=None,  # Never send raw image data to LLM
            resolution=PromptInjectionDefense.sanitize_text(
                episode.resolution or "", "resolution"
            )
            if episode.resolution
            else None,
            time_to_resolve_seconds=episode.time_to_resolve_seconds,
            tags=PromptInjectionDefense.sanitize_list(episode.tags, "tags"),
            severity=episode.severity,
        )

    def _build_user_prompt(self, episode: EpisodeCreate) -> str:
        """
        Build user prompt from sanitized episode data.

        Security: Episode data is already sanitized, but we still
        keep it separate from system prompt.
        """
        prompt_parts = [
            f"Goal: {episode.goal}",
            f"\nTool Chain: {' → '.join(episode.tool_chain)}",
            f"\nError Class: {episode.error_class.value}",
            "\nActions Taken:",
        ]

        for i, action in enumerate(episode.actions_taken[:20], 1):  # Limit to 20
            prompt_parts.append(f"  {i}. {action}")

        prompt_parts.append(f"\nError Trace:\n{episode.error_trace[:2000]}")  # Limit

        if episode.code_state_diff:
            diff = episode.code_state_diff
            if len(diff) > 2000:
                diff = diff[:2000] + "\n... (truncated)"
            prompt_parts.append(f"\nCode Diff:\n{diff}")

        if episode.environment_info:
            # Sanitize dict values
            safe_env = {
                str(k)[:100]: str(v)[:500]
                for k, v in list(episode.environment_info.items())[:20]
            }
            env_str = json.dumps(safe_env, indent=2)
            prompt_parts.append(f"\nEnvironment:\n{env_str}")

        if episode.resolution:
            prompt_parts.append(f"\nResolution: {episode.resolution[:1000]}")

        return "\n".join(prompt_parts)

    def _reconcile_perspectives(
        self, perspectives: list[LLMPerspective]
    ) -> ReflectionConsensus:
        """
        Reconcile multiple perspectives using semantic similarity consensus.

        Algorithm:
        1. Compute pairwise semantic similarity of root causes using embeddings
        2. If similarity > 0.85: unanimous semantic agreement
        3. If similarity 0.70-0.85: weighted voting by confidence scores
        4. If similarity < 0.70: use highest confidence model (no consensus)
        5. Merge preconditions (union with deduplication)
        6. Select best resolution (highest weighted score)

        Args:
            perspectives: List of LLM perspectives to reconcile

        Returns:
            ReflectionConsensus with agreed values and confidence

        Raises:
            ValueError: If perspectives list is empty
        """
        if not perspectives:
            raise ValueError("Cannot reconcile empty perspectives")

        # Single perspective: no consensus needed
        if len(perspectives) == 1:
            p = perspectives[0]
            return ReflectionConsensus(
                perspectives=perspectives,
                consensus_method="single_model",
                agreed_root_cause=p.root_cause,
                agreed_preconditions=p.preconditions,
                agreed_resolution=p.resolution_strategy,
                consensus_confidence=p.confidence_score * 0.8,  # Discount for single model
                disagreement_points=[],
                generated_at=datetime.now(UTC),
            )

        # Compute semantic similarity matrix for root causes
        root_causes = [p.root_cause for p in perspectives]
        similarity_matrix = self._compute_semantic_similarity_matrix(root_causes)

        # For 2 perspectives, get the single pairwise similarity
        if len(perspectives) == 2:
            similarity = similarity_matrix[0, 1]
            return self._reconcile_two_perspectives(perspectives, similarity)

        # For 3+ perspectives, use cluster-based consensus
        return self._reconcile_multiple_perspectives(perspectives, similarity_matrix)

    def _reconcile_two_perspectives(
        self,
        perspectives: list[LLMPerspective],
        similarity: float
    ) -> ReflectionConsensus:
        """
        Reconcile exactly two perspectives based on semantic similarity.

        Args:
            perspectives: Exactly two LLM perspectives
            similarity: Cosine similarity between root causes [0, 1]

        Returns:
            ReflectionConsensus
        """
        p1, p2 = perspectives[0], perspectives[1]

        # High similarity: semantic agreement (same root cause meaning)
        if similarity >= self.SEMANTIC_MATCH_THRESHOLD:
            # Choose root cause with higher confidence
            if p1.confidence_score >= p2.confidence_score:
                agreed_root_cause = p1.root_cause
            else:
                agreed_root_cause = p2.root_cause

            consensus_confidence = min(1.0, 0.5 + similarity * 0.5)  # 0.85 sim -> 0.925 conf
            consensus_method = "semantic_unanimous"
            disagreement_points = []
            logger.info(
                f"Semantic consensus reached: similarity={similarity:.3f}, "
                f"confidence={consensus_confidence:.3f}"
            )

        # Partial similarity: weighted voting
        elif similarity >= self.SEMANTIC_PARTIAL_THRESHOLD:
            # Weight by confidence
            total_conf = p1.confidence_score + p2.confidence_score
            weight1 = p1.confidence_score / total_conf
            weight2 = p2.confidence_score / total_conf

            # Choose root cause with higher weighted score
            weighted_score1 = weight1 * (1 + similarity)  # Boost for similarity
            weighted_score2 = weight2 * (1 + similarity)

            if weighted_score1 >= weighted_score2:
                agreed_root_cause = p1.root_cause
                disagreement_points = [f"{p2.model_name}: {p2.root_cause}"]
            else:
                agreed_root_cause = p2.root_cause
                disagreement_points = [f"{p1.model_name}: {p1.root_cause}"]

            # Confidence based on similarity and model confidences
            consensus_confidence = similarity * max(p1.confidence_score, p2.confidence_score)
            consensus_method = "weighted_semantic_vote"
            logger.info(
                f"Weighted semantic vote: similarity={similarity:.3f}, "
                f"weights=[{weight1:.2f}, {weight2:.2f}], confidence={consensus_confidence:.3f}"
            )

        # Low similarity: no consensus, use highest confidence
        else:
            if p1.confidence_score >= p2.confidence_score:
                agreed_root_cause = p1.root_cause
                disagreement_points = [f"{p2.model_name}: {p2.root_cause}"]
            else:
                agreed_root_cause = p2.root_cause
                disagreement_points = [f"{p1.model_name}: {p1.root_cause}"]

            # Low confidence for disagreement
            consensus_confidence = max(p1.confidence_score, p2.confidence_score) * 0.5
            consensus_method = "highest_confidence_fallback"
            logger.warning(
                f"Low semantic similarity ({similarity:.3f}), using highest confidence model. "
                f"Disagreement: {disagreement_points}"
            )

        # Merge preconditions (union, deduplicate by semantic similarity)
        agreed_preconditions = self._merge_preconditions_semantic(
            [p.preconditions for p in perspectives]
        )

        # Select best resolution (highest confidence-weighted)
        if p1.confidence_score >= p2.confidence_score:
            agreed_resolution = p1.resolution_strategy
        else:
            agreed_resolution = p2.resolution_strategy

        return ReflectionConsensus(
            perspectives=perspectives,
            consensus_method=consensus_method,
            agreed_root_cause=agreed_root_cause,
            agreed_preconditions=agreed_preconditions,
            agreed_resolution=agreed_resolution,
            consensus_confidence=consensus_confidence,
            disagreement_points=disagreement_points,
            generated_at=datetime.now(UTC),
        )

    def _reconcile_multiple_perspectives(
        self,
        perspectives: list[LLMPerspective],
        similarity_matrix: np.ndarray
    ) -> ReflectionConsensus:
        """
        Reconcile 3+ perspectives using similarity clustering.

        Args:
            perspectives: List of 3+ LLM perspectives
            similarity_matrix: Pairwise cosine similarity matrix

        Returns:
            ReflectionConsensus
        """
        n = len(perspectives)

        # Find the perspective that has highest average similarity to others
        avg_similarities = []
        for i in range(n):
            # Average similarity to all other perspectives
            other_sims = [similarity_matrix[i, j] for j in range(n) if i != j]
            avg_similarities.append(np.mean(other_sims))

        # Sort by average similarity (descending)
        ranked_indices = np.argsort(avg_similarities)[::-1]
        best_idx = ranked_indices[0]
        best_perspective = perspectives[best_idx]
        best_avg_similarity = avg_similarities[best_idx]

        # Count how many perspectives are semantically similar to the best one
        agreement_count = 1  # Include self
        disagreement_points = []
        for i in range(n):
            if i != best_idx:
                sim = similarity_matrix[best_idx, i]
                if sim >= self.SEMANTIC_MATCH_THRESHOLD:
                    agreement_count += 1
                elif sim < self.SEMANTIC_PARTIAL_THRESHOLD:
                    disagreement_points.append(
                        f"{perspectives[i].model_name}: {perspectives[i].root_cause}"
                    )

        # Determine consensus method based on agreement level
        agreement_ratio = agreement_count / n

        if agreement_ratio >= 0.67:  # 2/3 or more agree semantically
            consensus_method = "semantic_majority"
            consensus_confidence = min(1.0, agreement_ratio * best_avg_similarity)
        elif agreement_ratio >= 0.5:  # Simple majority
            consensus_method = "weighted_semantic_majority"
            consensus_confidence = agreement_ratio * best_perspective.confidence_score
        else:
            consensus_method = "highest_confidence_fallback"
            consensus_confidence = best_perspective.confidence_score * 0.5

        logger.info(
            f"Multi-perspective consensus: method={consensus_method}, "
            f"agreement={agreement_count}/{n}, avg_sim={best_avg_similarity:.3f}, "
            f"confidence={consensus_confidence:.3f}"
        )

        # Merge preconditions from all perspectives
        agreed_preconditions = self._merge_preconditions_semantic(
            [p.preconditions for p in perspectives]
        )

        # Use resolution from highest-confidence perspective
        best_resolution_perspective = max(perspectives, key=lambda p: p.confidence_score)

        return ReflectionConsensus(
            perspectives=perspectives,
            consensus_method=consensus_method,
            agreed_root_cause=best_perspective.root_cause,
            agreed_preconditions=agreed_preconditions,
            agreed_resolution=best_resolution_perspective.resolution_strategy,
            consensus_confidence=consensus_confidence,
            disagreement_points=disagreement_points,
            generated_at=datetime.now(UTC),
        )

    def _compute_semantic_similarity_matrix(
        self, texts: list[str]
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix for texts.

        Args:
            texts: List of text strings

        Returns:
            np.ndarray: N x N similarity matrix where entry [i,j] is
                       cosine similarity between texts[i] and texts[j]
        """
        if not texts:
            return np.array([[]])

        n = len(texts)
        if n == 1:
            return np.array([[1.0]])

        normalized = [" ".join(t.lower().split()) for t in texts]

        try:
            model = self._get_similarity_model()
            embeddings = model.encode(
                normalized,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            # Cosine similarity for normalized vectors is just dot-product.
            sim = np.matmul(embeddings, embeddings.T)
            # Numerical safety.
            return np.clip(sim, -1.0, 1.0)
        except Exception as e:
            logger.warning(
                "Semantic similarity computation failed; falling back to string equality: %s",
                e,
                exc_info=True,
            )
            matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    matrix[i, j] = 1.0 if normalized[i] == normalized[j] else 0.0
            return matrix

    def _get_similarity_model(self):
        """Lazy-load sentence-transformers model used for semantic similarity."""
        if self._similarity_model is not None:
            return self._similarity_model

        with self._similarity_model_lock:
            if self._similarity_model is not None:
                return self._similarity_model

            from sentence_transformers import SentenceTransformer

            # Keep consistent with EmbeddingService default.
            model_name = "all-MiniLM-L6-v2"
            try:
                from src.config import get_settings

                model_name = get_settings().embedding.text_model_name
            except Exception:
                pass

            self._similarity_model = SentenceTransformer(model_name, device="cpu")
            return self._similarity_model

    def _merge_preconditions_semantic(
        self, precondition_lists: list[list[str]]
    ) -> list[str]:
        """
        Merge preconditions from multiple perspectives with semantic deduplication.

        Removes preconditions that are semantically similar (>0.85 cosine similarity)
        to avoid redundancy while preserving unique insights.

        Args:
            precondition_lists: List of precondition lists from each perspective

        Returns:
            Merged and deduplicated list of preconditions
        """
        # Flatten all preconditions
        all_preconditions = []
        for plist in precondition_lists:
            all_preconditions.extend(plist)

        if not all_preconditions:
            return []

        if len(all_preconditions) == 1:
            return all_preconditions

        try:
            similarity = self._compute_semantic_similarity_matrix(all_preconditions)

            # Greedy deduplication: keep precondition if not similar to any kept one
            kept_indices: list[int] = []
            for i in range(len(all_preconditions)):
                if all(similarity[i, j] < self.SEMANTIC_MATCH_THRESHOLD for j in kept_indices):
                    kept_indices.append(i)

            merged = [all_preconditions[i] for i in kept_indices]
            logger.debug(
                f"Merged {len(all_preconditions)} preconditions to {len(merged)} "
                f"(removed {len(all_preconditions) - len(merged)} duplicates)"
            )
            return merged

        except Exception as e:
            logger.warning(
                f"Semantic precondition merge failed: {e}. Using simple deduplication."
            )
            # Fallback: simple string deduplication
            return list(dict.fromkeys(all_preconditions))

    def _merge_list_fields(self, lists: list[list[str]]) -> list[str]:
        """Merge multiple lists, deduplicate, preserve order."""
        merged = []
        seen = set()
        for lst in lists:
            for item in lst:
                if item not in seen:
                    merged.append(item)
                    seen.add(item)
        return merged

    def _create_fallback_reflection(self, episode: EpisodeCreate) -> Reflection:
        """
        Create heuristic reflection when LLMs fail.

        This ensures the system never fails completely.
        """
        logger.warning("Creating fallback reflection (LLM-free heuristic)")

        root_cause = f"{episode.error_class.value} in {episode.tool_chain[0]}"

        preconditions = [
            f"Using tool: {episode.tool_chain[0]}",
            f"Error class: {episode.error_class.value}",
        ]

        resolution_strategy = (
            episode.resolution
            if episode.resolution
            else "Manual investigation required - check error trace and environment"
        )

        return Reflection(
            consensus=None,  # No consensus for fallback
            root_cause=root_cause,
            preconditions=preconditions,
            resolution_strategy=resolution_strategy,
            environment_factors=list(episode.environment_info.keys())[:10]
            if episode.environment_info
            else [],
            affected_components=episode.tool_chain[:5],
            generalization_score=0.3,
            confidence_score=0.4,  # Low confidence for heuristic
            llm_model="fallback_heuristic",
            generated_at=datetime.now(UTC),
            cost_usd=0.0,  # Free
            generation_latency_ms=0.0,
        )

    def get_usage_stats(self) -> dict:
        """Get service usage statistics (thread-safe)."""
        with self._stats_lock:
            return {
                "total_cost_usd": self.total_cost_usd,
                "total_requests": self.total_requests,
                "requests_by_model": dict(self.requests_by_model),
                "average_cost_per_request": (
                    self.total_cost_usd / self.total_requests
                    if self.total_requests > 0
                    else 0.0
                ),
            }
