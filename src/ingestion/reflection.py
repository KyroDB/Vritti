"""
LLM-based reflection generation for episodic memory.

Generates multi-perspective analysis of failure/success episodes using GPT-4.
Extracts:
- Root cause analysis
- Preconditions for relevance
- Resolution strategy
- Environment factors
- Generalization scoring
"""

import asyncio
import json
import logging
from typing import Optional

from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError
from pydantic import ValidationError

from src.config import LLMConfig
from src.models.episode import Reflection, EpisodeCreate

logger = logging.getLogger(__name__)


class ReflectionService:
    """
    Async service for generating episode reflections using LLM.

    Handles:
    - Structured output extraction (JSON mode)
    - Retry logic for rate limits and transient failures
    - Cost tracking (token usage)
    - Graceful degradation (fallback to simple reflection)
    """

    # System prompt for reflection generation
    SYSTEM_PROMPT = """You are an expert AI assistant analyzing software development failures and successes.

Your task is to generate a structured reflection on an episode to help future debugging.

Extract the following:
1. **Root Cause**: The fundamental reason for the failure (not just symptoms)
2. **Preconditions**: Required environment/state for this episode to be relevant (be specific)
3. **Resolution Strategy**: How the issue was (or could be) resolved
4. **Environment Factors**: OS, versions, tools that matter
5. **Affected Components**: System components involved
6. **Generalization Score**: 0-1 (0=very specific context, 1=universal pattern)
7. **Confidence Score**: 0-1 (confidence in this analysis)

Return valid JSON matching this schema:
{
  "root_cause": "string (concise, actionable)",
  "preconditions": ["string", ...],  // 3-5 specific conditions
  "resolution_strategy": "string (step-by-step if possible)",
  "environment_factors": ["string", ...],
  "affected_components": ["string", ...],
  "generalization_score": float,  // 0.0 to 1.0
  "confidence_score": float  // 0.0 to 1.0
}

Focus on actionable insights that help match future similar failures."""

    def __init__(self, config: LLMConfig):
        """
        Initialize reflection service.

        Args:
            config: LLM configuration

        Raises:
            ValueError: If API key is missing
        """
        if not config.api_key:
            logger.warning(
                "OpenAI API key not configured - reflection generation disabled"
            )
            self.client = None
        else:
            self.client = AsyncOpenAI(
                api_key=config.api_key,
                timeout=config.timeout_seconds,
                max_retries=0,  # We handle retries manually
            )

        self.config = config
        self.total_tokens_used = 0
        self.total_requests = 0

    async def generate_reflection(
        self,
        episode: EpisodeCreate,
        max_retries: int = 3,
    ) -> Optional[Reflection]:
        """
        Generate reflection for an episode using LLM.

        Args:
            episode: Episode data
            max_retries: Maximum retry attempts for rate limits

        Returns:
            Reflection: Generated reflection, or None if generation fails

        Raises:
            No exceptions raised - returns None on failure
        """
        if self.client is None:
            logger.info("Reflection generation skipped (no API key)")
            return self._create_fallback_reflection(episode)

        # Build user prompt from episode
        user_prompt = self._build_user_prompt(episode)

        # Attempt generation with retries
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    response_format={"type": "json_object"},  # Force JSON output
                )

                # Track usage
                self.total_tokens_used += response.usage.total_tokens
                self.total_requests += 1

                # Parse JSON response
                content = response.choices[0].message.content
                reflection_data = json.loads(content)

                # Validate and create Reflection object
                reflection = Reflection(
                    llm_model=self.config.model_name,
                    **reflection_data
                )

                logger.info(
                    f"Reflection generated "
                    f"(tokens: {response.usage.total_tokens}, "
                    f"confidence: {reflection.confidence_score:.2f})"
                )

                return reflection

            except RateLimitError as e:
                # Retry with exponential backoff
                if attempt < max_retries - 1:
                    backoff = 2 ** attempt
                    logger.warning(
                        f"Rate limit hit, retrying in {backoff}s... "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(backoff)
                    continue
                else:
                    logger.error(f"Rate limit exceeded after {max_retries} retries")
                    return self._create_fallback_reflection(episode)

            except (APIConnectionError, APIError) as e:
                # Transient errors: retry
                if attempt < max_retries - 1:
                    backoff = 1.0 * (2 ** attempt)
                    logger.warning(
                        f"API error ({e.__class__.__name__}), "
                        f"retrying in {backoff}s..."
                    )
                    await asyncio.sleep(backoff)
                    continue
                else:
                    logger.error(
                        f"API error after {max_retries} retries: {e}"
                    )
                    return self._create_fallback_reflection(episode)

            except (json.JSONDecodeError, ValidationError, KeyError) as e:
                # Parsing errors: log and return fallback
                logger.error(
                    f"Failed to parse reflection response: {e}\n"
                    f"Response: {content if 'content' in locals() else 'N/A'}"
                )
                return self._create_fallback_reflection(episode)

            except Exception as e:
                # Unexpected errors
                logger.error(f"Unexpected error generating reflection: {e}")
                return self._create_fallback_reflection(episode)

        # Should not reach here
        return self._create_fallback_reflection(episode)

    def _build_user_prompt(self, episode: EpisodeCreate) -> str:
        """
        Build user prompt from episode data.

        Args:
            episode: Episode data

        Returns:
            str: Formatted prompt for LLM
        """
        prompt_parts = [
            f"**Goal**: {episode.goal}",
            f"\n**Tool Chain**: {' → '.join(episode.tool_chain)}",
            f"\n**Error Class**: {episode.error_class.value}",
            f"\n**Actions Taken**:",
        ]

        for i, action in enumerate(episode.actions_taken, 1):
            prompt_parts.append(f"  {i}. {action}")

        prompt_parts.append(f"\n**Error Trace**:\n```\n{episode.error_trace}\n```")

        if episode.code_state_diff:
            # Truncate large diffs
            diff = episode.code_state_diff
            if len(diff) > 2000:
                diff = diff[:2000] + "\n... (truncated)"
            prompt_parts.append(f"\n**Code Diff**:\n```diff\n{diff}\n```")

        if episode.environment_info:
            env_str = json.dumps(episode.environment_info, indent=2)
            prompt_parts.append(f"\n**Environment**:\n```json\n{env_str}\n```")

        if episode.resolution:
            prompt_parts.append(f"\n**Resolution**: {episode.resolution}")

        prompt_parts.append(
            "\n**Task**: Analyze this episode and provide structured reflection "
            "in valid JSON format."
        )

        return "\n".join(prompt_parts)

    def _create_fallback_reflection(self, episode: EpisodeCreate) -> Reflection:
        """
        Create simple rule-based reflection when LLM is unavailable.

        Args:
            episode: Episode data

        Returns:
            Reflection: Basic reflection with heuristics
        """
        logger.info("Creating fallback reflection (rule-based)")

        # Heuristic root cause
        root_cause = f"{episode.error_class.value} in {episode.tool_chain[0]}"

        # Heuristic preconditions
        preconditions = [
            f"Using tool: {episode.tool_chain[0]}",
            f"Error class: {episode.error_class.value}",
        ]

        if episode.environment_info:
            if "os" in episode.environment_info:
                preconditions.append(f"OS: {episode.environment_info['os']}")

        # Heuristic resolution
        resolution_strategy = (
            episode.resolution
            if episode.resolution
            else "Manual investigation required"
        )

        # Heuristic environment factors
        environment_factors = list(episode.environment_info.keys()) if episode.environment_info else []

        # Heuristic affected components
        affected_components = episode.tool_chain.copy()

        # Conservative scores for fallback
        generalization_score = 0.3  # Low - very specific
        confidence_score = 0.4  # Low - heuristic-based

        return Reflection(
            root_cause=root_cause,
            preconditions=preconditions,
            resolution_strategy=resolution_strategy,
            environment_factors=environment_factors,
            affected_components=affected_components,
            generalization_score=generalization_score,
            confidence_score=confidence_score,
            llm_model="fallback_heuristic",
        )

    def get_usage_stats(self) -> dict[str, int]:
        """
        Get token usage statistics.

        Returns:
            dict: Usage stats (total_tokens, total_requests, avg_tokens)
        """
        avg_tokens = (
            self.total_tokens_used // self.total_requests
            if self.total_requests > 0
            else 0
        )

        return {
            "total_tokens": self.total_tokens_used,
            "total_requests": self.total_requests,
            "avg_tokens_per_request": avg_tokens,
        }
