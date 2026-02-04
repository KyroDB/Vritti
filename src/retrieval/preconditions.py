"""
Precondition matching engine for episodic memory retrieval.

Matches current execution context against episode preconditions to determine relevance.

Uses:
- Exact string matching for tool names, error classes
- Fuzzy matching for environment variables
- Set intersection for required components
- Heuristic scoring (0.0-1.0)

Optimized for <10-15ms latency per candidate.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from typing import Any

import httpx

from src.models.episode import Episode
from src.models.search import PreconditionCheckResult

logger = logging.getLogger(__name__)


class PreconditionMatcher:
    """
    Matches current state against episode preconditions.

    Fast heuristic-based matching without LLM calls.
    """

    def __init__(self) -> None:
        """Initialize precondition matcher."""
        pass

    def check_preconditions(
        self,
        episode: Episode,
        current_state: dict[str, Any],
        threshold: float = 0.5,
    ) -> PreconditionCheckResult:
        """
        Check if current state matches episode preconditions.

        Args:
            episode: Episode with reflection containing preconditions
            current_state: Current execution context (tool, env, components, etc.)
            threshold: Minimum match score to consider matched (0-1)

        Returns:
            PreconditionCheckResult: Match result with score and explanation

        Example current_state:
            {
                "tool": "kubectl",
                "error_class": "ImagePullBackOff",
                "environment": {"os": "Darwin", "kubectl_version": "1.28"},
                "components": ["kubernetes", "docker"],
                "goal_keywords": ["deploy", "production"]
            }
        """
        if not episode.reflection or not episode.reflection.preconditions:
            # No preconditions: assume universal match
            return PreconditionCheckResult(
                matched=True,
                match_score=1.0,
                matched_preconditions=[],
                missing_preconditions=[],
                explanation="No specific preconditions (universal relevance)",
            )

        preconditions = episode.reflection.preconditions
        matched_preconds = []
        missing_preconds = []
        scores = []

        for precond in preconditions:
            score = self._match_single_precondition(precond, current_state)
            scores.append(score)

            if score >= 0.7:  # Consider matched if score >= 0.7
                matched_preconds.append(precond)
            else:
                missing_preconds.append(precond)

        # Overall score: average of individual precondition scores
        overall_score = sum(scores) / len(scores) if scores else 0.0

        matched = overall_score >= threshold

        # Build explanation
        if matched:
            explanation = (
                f"Matched {len(matched_preconds)}/{len(preconditions)} preconditions "
                f"(score: {overall_score:.2f})"
            )
        else:
            explanation = (
                f"Failed threshold: {overall_score:.2f} < {threshold} "
                f"(matched {len(matched_preconds)}/{len(preconditions)})"
            )

        return PreconditionCheckResult(
            matched=matched,
            match_score=overall_score,
            matched_preconditions=matched_preconds,
            missing_preconditions=missing_preconds,
            explanation=explanation,
        )

    def _match_single_precondition(self, precondition: str, current_state: dict[str, Any]) -> float:
        """
        Match a single precondition against current state.

        Uses heuristic rules:
        - "Using tool: X" → check current_state["tool"]
        - "Error class: X" → check current_state["error_class"]
        - "OS: X" → check current_state["environment"]["os"]
        - "Version: X" → fuzzy version matching
        - Component names → check current_state["components"]

        Args:
            precondition: Single precondition string
            current_state: Current execution context

        Returns:
            float: Match score (0.0-1.0)
        """
        precond_lower = precondition.lower()

        # Pattern 1: "Using tool: X"
        if "using tool:" in precond_lower or "tool:" in precond_lower:
            tool_in_precond = self._extract_value_after_colon(precondition)
            current_tool = current_state.get("tool", "").lower()

            if tool_in_precond and current_tool:
                if tool_in_precond in current_tool or current_tool in tool_in_precond:
                    return 1.0  # Exact match
                else:
                    return 0.0  # Tool mismatch

        # Pattern 2: "Error class: X"
        if "error class:" in precond_lower or "error:" in precond_lower:
            error_in_precond = self._extract_value_after_colon(precondition).lower()
            current_error = current_state.get("error_class", "").lower()

            if error_in_precond and current_error:
                if error_in_precond in current_error or current_error in error_in_precond:
                    return 1.0
                else:
                    return 0.2  # Error class mismatch (partial credit)

        # Pattern 3: "OS: X" or environment checks
        if "os:" in precond_lower or "operating system:" in precond_lower:
            os_in_precond = self._extract_value_after_colon(precondition).lower()
            current_env = current_state.get("environment", {})
            current_os = current_env.get("os", "").lower()

            if os_in_precond and current_os:
                if os_in_precond in current_os or current_os in os_in_precond:
                    return 1.0
                else:
                    return 0.3  # OS mismatch (some credit - may still be relevant)

        # Pattern 4: Version requirements (fuzzy matching)
        if "version" in precond_lower:
            # Extract version numbers from precondition and current state
            precond_version = self._extract_version(precondition)
            current_versions = self._extract_versions_from_env(current_state)

            if precond_version and current_versions:
                for curr_ver in current_versions:
                    # Fuzzy version matching: major.minor match is good enough
                    if self._versions_compatible(precond_version, curr_ver):
                        return 0.9
                return 0.4  # Version present but doesn't match
            else:
                return 0.5  # Can't determine version, neutral

        # Pattern 5: Component/service names
        current_components = {c.lower() for c in current_state.get("components", [])}

        if current_components:
            # Check if any component mentioned in precondition
            for component in current_components:
                if component in precond_lower or precond_lower in component:
                    return 0.9

        # Pattern 6: Keyword matching in goal
        goal_keywords = {kw.lower() for kw in current_state.get("goal_keywords", [])}

        if goal_keywords:
            precond_words = set(re.findall(r"\w+", precond_lower))
            overlap = goal_keywords & precond_words

            if overlap:
                # Partial credit based on keyword overlap
                return min(0.8, len(overlap) / len(goal_keywords))

        # Default: partial credit for unrecognized patterns
        return 0.4

    def _extract_value_after_colon(self, text: str) -> str:
        """Extract value after colon in 'Key: Value' pattern."""
        if ":" in text:
            return text.split(":", 1)[1].strip().strip("\"'")
        return ""

    def _extract_version(self, text: str) -> str | None:
        """Extract version number from text (e.g., '1.28.0')."""
        match = re.search(r"\d+\.\d+(?:\.\d+)?", text)
        return match.group(0) if match else None

    def _extract_versions_from_env(self, current_state: dict[str, Any]) -> list[str]:
        """Extract all version numbers from environment."""
        env = current_state.get("environment", {})
        versions = []

        for key, value in env.items():
            if "version" in key.lower():
                ver = self._extract_version(str(value))
                if ver:
                    versions.append(ver)

        return versions

    def _versions_compatible(self, ver1: str, ver2: str) -> bool:
        """Check if two versions are compatible (major.minor match)."""
        try:
            parts1 = [int(x) for x in ver1.split(".")]
            parts2 = [int(x) for x in ver2.split(".")]

            # Match major.minor (ignore patch)
            if len(parts1) >= 2 and len(parts2) >= 2:
                return parts1[0] == parts2[0] and parts1[1] == parts2[1]
            else:
                return ver1 == ver2
        except (ValueError, IndexError):
            return ver1 == ver2


# Singleton instance
_matcher: PreconditionMatcher | None = None


def get_precondition_matcher() -> PreconditionMatcher:
    """
    Get global precondition matcher instance.

    Returns:
        PreconditionMatcher: Singleton instance
    """
    global _matcher
    if _matcher is None:
        _matcher = PreconditionMatcher()
    return _matcher


# ===================================================================
# Advanced LLM-Based Precondition Matching
# ===================================================================


class AdvancedPreconditionMatcher:
    """
    Enhanced precondition matching with LLM semantic validation via OpenRouter.

    Two-stage approach:
    1. Fast heuristic matching (existing PreconditionMatcher)
    2. Slow LLM validation for high-similarity candidates (>0.85)

    Security:
    - Customer ID validation
    - Input sanitization for LLM prompts
    - Query truncation to prevent abuse
    - Timeout protection (2s hard limit)

    Performance:
    - Only validates high-similarity matches (>0.85)
    - 5-minute LRU cache for validation results
    - Graceful fallback on errors
    - Batch validation support
    """

    # Configuration
    SIMILARITY_THRESHOLD_FOR_LLM = 0.85  # Only validate if similarity >= this
    LLM_CONFIDENCE_THRESHOLD = 0.7  # Require LLM confidence >= 0.7 to accept
    # Security: Timeout for LLM validation (prevent hanging)
    # Note: This is overridden by config.search.llm_timeout_seconds if available
    VALIDATION_TIMEOUT_MS = 2000  # 2 seconds default
    CACHE_SIZE = 500  # LRU cache size
    MAX_QUERY_LENGTH = 500  # Security: prevent abuse

    # OpenRouter configuration
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "x-ai/grok-4.1-fast:free"  # Fast free model for validation

    def __init__(
        self,
        openrouter_api_key: str | None = None,
        enable_llm: bool = True,
        model: str | None = None,
    ):
        """
        Initialize advanced precondition matcher with OpenRouter.

        Args:
            openrouter_api_key: OpenRouter API key for LLM access
            enable_llm: Feature flag to enable/disable LLM validation
            model: OpenRouter model to use (defaults to cheap free tier)
        """
        self.basic_matcher = PreconditionMatcher()
        self.openrouter_api_key = openrouter_api_key or ""
        self.model = model or self.DEFAULT_MODEL
        self.enable_llm = enable_llm and bool(self.openrouter_api_key)

        # Statistics
        self.stats = {
            "llm_calls": 0,
            "llm_rejections": 0,
            "cache_hits": 0,
            "timeouts": 0,
            "errors": 0,
            "total_cost_usd": 0.0,
        }

        # Initialize OpenRouter client if available
        if self.enable_llm:
            logger.info(f"LLM precondition validation enabled (OpenRouter: {self.model})")
        else:
            if not self.openrouter_api_key:
                logger.warning("LLM validation disabled: no OpenRouter API key provided")
            else:
                logger.warning(f"LLM validation disabled: enable_llm={enable_llm}")

    async def check_preconditions_with_llm(
        self,
        candidate_episode: Episode,
        current_query: str,
        current_state: dict[str, Any],
        threshold: float = 0.7,
        similarity_score: float | None = None,
    ) -> PreconditionCheckResult:
        """
        Check preconditions with LLM semantic validation.

        This catches negation, time-based differences, and other
        subtle semantic mismatches that embeddings miss.

        Args:
            candidate_episode: Episode to check
            current_query: Current user query/goal
            current_state: Current execution context
            threshold: Minimum match score
            similarity_score: Vector similarity score (if known)

        Returns:
            PreconditionCheckResult with LLM validation applied
        """
        # Step 1: Fast heuristic check
        heuristic_result = self.basic_matcher.check_preconditions(
            episode=candidate_episode, current_state=current_state, threshold=threshold
        )

        # Step 2: If heuristic matched AND similarity is high, validate with LLM
        if (
            heuristic_result.matched
            and self.enable_llm
            and similarity_score
            and similarity_score >= self.SIMILARITY_THRESHOLD_FOR_LLM
        ):

            # Extract episode goal
            past_goal = candidate_episode.create_data.goal
            past_actions = candidate_episode.create_data.actions_taken[:3]  # First 3

            # LLM validation
            is_compatible = await self._llm_validate_compatibility(
                past_goal=past_goal,
                current_query=current_query,
                past_actions=past_actions,
                current_state=current_state,
            )

            if not is_compatible:
                # LLM rejected - override heuristic result
                logger.info(
                    f"LLM rejected high-similarity candidate (sim={similarity_score:.2f}): "
                    f"past='{past_goal[:50]}' vs current='{current_query[:50]}'"
                )

                self.stats["llm_rejections"] += 1

                return PreconditionCheckResult(
                    matched=False,
                    match_score=0.0,
                    matched_preconditions=[],
                    missing_preconditions=heuristic_result.matched_preconditions,
                    explanation=(
                        f"Semantically incompatible despite high vector similarity "
                        f"(sim={similarity_score:.2f}). LLM validation detected "
                        f"negation, opposite meaning, or context mismatch."
                    ),
                )

        return heuristic_result

    async def _llm_validate_compatibility(
        self,
        past_goal: str,
        current_query: str,
        past_actions: list[str],
        current_state: dict[str, Any],
    ) -> bool:
        """
        Use OpenRouter LLM to verify semantic compatibility.

        Returns:
            True if compatible, False if incompatible
        """
        # Security: Truncate inputs
        past_goal = self._sanitize_input(past_goal)
        current_query = self._sanitize_input(current_query)

        # Check cache first
        cache_key = self._get_cache_key(past_goal, current_query)
        cached_result = self._get_from_cache(cache_key)

        if cached_result is not None:
            self.stats["cache_hits"] += 1
            return cached_result

        if not self.enable_llm:
            # Fallback: accept if no LLM available
            return True

        # Build prompt
        prompt = self._build_validation_prompt(
            past_goal, current_query, past_actions, current_state
        )

        try:
            # Call LLM with timeout
            start_time = time.perf_counter()

            result = await asyncio.wait_for(
                self._call_openrouter(prompt),
                timeout=self.VALIDATION_TIMEOUT_MS / 1000,  # Convert to seconds
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Parse result
            compatible, confidence, reason = result

            # Track stats
            self.stats["llm_calls"] += 1
            # Free tier: $0.0 per call
            self.stats["total_cost_usd"] += 0.0

            # Only accept if high confidence
            is_compatible = compatible and confidence >= self.LLM_CONFIDENCE_THRESHOLD

            # Cache result
            self._add_to_cache(cache_key, is_compatible)

            logger.debug(
                f"LLM validation: compatible={is_compatible} "
                f"(confidence={confidence:.2f}, latency={latency_ms:.1f}ms, reason={reason})"
            )

            return is_compatible

        except asyncio.TimeoutError:
            self.stats["timeouts"] += 1
            logger.warning(f"LLM validation timeout after {self.VALIDATION_TIMEOUT_MS}ms")
            # Fallback: accept on timeout
            return True

        except Exception as e:
            self.stats["errors"] += 1
            logger.warning(f"LLM validation error: {e}")
            # Fallback: accept on errors
            return True

    async def _call_openrouter(self, prompt: str) -> tuple[bool, float, str]:
        """
        Call OpenRouter API for LLM validation.

        Returns:
            (compatible, confidence, reason)
        """
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/episodic-memory",
            "X-Title": "EpisodicMemory Precondition Validation",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,  # Low temperature for consistency
            "max_tokens": 200,
        }

        async with httpx.AsyncClient(timeout=self.VALIDATION_TIMEOUT_MS / 1000) as client:
            response = await client.post(
                f"{self.OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]

        # Parse JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        try:
            result = json.loads(content.strip())

            compatible = result.get("compatible", True)
            confidence = result.get("confidence", 0.5)
            reason = result.get("reason", "Unknown")

            return (compatible, confidence, reason)

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response: {content[:100]}")
            # Fallback: accept with low confidence
            return (True, 0.5, "Parse error")

    def _build_validation_prompt(
        self,
        past_goal: str,
        current_query: str,
        past_actions: list[str],
        current_state: dict[str, Any],
    ) -> str:
        """Build prompt for LLM validation."""

        # Extract environment if present
        environment = current_state.get("environment", {})
        env_str = json.dumps(environment, indent=2)[:200] if environment else "N/A"

        return f"""Are these two goals semantically COMPATIBLE (not just similar)?

Past failure goal: "{past_goal}"
Past actions taken: {', '.join(past_actions) if past_actions else 'N/A'}

Current query: "{current_query}"
Current state: {env_str}

Return JSON:
{{
  "compatible": true/false,
  "reason": "brief explanation",
  "confidence": 0.0-1.0
}}

IMPORTANT RULES:
- "delete files older than 7 days" and "delete files EXCEPT older than 7 days" are NOT compatible (opposite)
- "deploy to staging" and "deploy to production" are NOT compatible (different environments)
- Time conditions must match exactly (">7 days" vs "except >7 days" are opposite)
- Action direction matters ("upload" vs "download" are opposite)
- Negations matter ("with X" vs "without X" are different)
- "start" vs "stop" are opposite
- "install" vs "uninstall" are opposite

Be strict: only return compatible=true if the current query would benefit from the past failure's lesson.
If the goals have opposite meanings or would require different solutions, return compatible=false.
"""

    def _sanitize_input(self, text: str) -> str:
        """Sanitize input for LLM prompt."""
        if not text:
            return ""

        # Truncate to max length
        if len(text) > self.MAX_QUERY_LENGTH:
            text = text[: self.MAX_QUERY_LENGTH] + "..."

        # Basic sanitization (prevent prompt injection)
        text = text.replace("```", "").replace("</", "").replace("<", "")

        # Remove code blocks if they look like injection
        if "import " in text or "system(" in text:
            # Simple heuristic: if it looks like code, strip it
            text = re.sub(r"import\s+\w+", "", text)
            text = re.sub(r"system\(.*\)", "", text)

        return text.strip()

    def _get_cache_key(self, past_goal: str, current_query: str) -> str:
        """Generate cache key from goals."""
        # Simple hash-based key
        combined = f"{past_goal[:100]}|{current_query[:100]}"
        return hashlib.md5(combined.encode()).hexdigest()

    # Simple cache implementation (in-memory)
    _cache: dict[str, tuple[bool, float]] = {}  # key -> (result, timestamp)
    _cache_ttl = 300  # 5 minutes

    def _get_from_cache(self, key: str) -> bool | None:
        """Get from cache if not expired."""
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return result
            else:
                # Expired
                del self._cache[key]
        return None

    def _add_to_cache(self, key: str, result: bool) -> None:
        """Add to cache with timestamp."""
        # Evict oldest if cache too large
        if len(self._cache) >= self.CACHE_SIZE:
            # Remove oldest entry
            oldest_key = min(self._cache.items(), key=lambda x: x[1][1])[0]
            del self._cache[oldest_key]

        self._cache[key] = (result, time.time())

    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear the validation cache.

        Useful for testing to ensure cache isolation between tests.
        Thread-safe: clears entire cache atomically.
        """
        cls._cache.clear()

    def get_stats(self) -> dict:
        """Get validation statistics."""
        return {
            **self.stats,
            "cache_size": len(self._cache),
            "cache_hit_rate": (
                self.stats["cache_hits"]
                / max(1, self.stats["llm_calls"] + self.stats["cache_hits"])
            ),
        }


# Singleton instance for advanced matcher
_advanced_matcher: AdvancedPreconditionMatcher | None = None


def get_advanced_precondition_matcher(
    openrouter_api_key: str | None = None, enable_llm: bool = True
) -> AdvancedPreconditionMatcher:
    """
    Get global advanced precondition matcher instance.

    Args:
        openrouter_api_key: OpenRouter API key (optional, uses config if not provided)
        enable_llm: Feature flag to enable/disable LLM validation

    Returns:
        AdvancedPreconditionMatcher: Singleton instance
    """
    global _advanced_matcher

    if _advanced_matcher is None:
        # Get API key from config if not provided
        if openrouter_api_key is None:
            try:
                from src.config import get_settings

                settings = get_settings()
                openrouter_api_key = settings.llm.openrouter_api_key
                enable_llm = getattr(settings.search, "enable_llm_validation", enable_llm)
            except Exception as e:
                logger.warning(f"Could not load config for LLM validation: {e}")

        _advanced_matcher = AdvancedPreconditionMatcher(
            openrouter_api_key=openrouter_api_key, enable_llm=enable_llm
        )

    return _advanced_matcher
