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

import logging
import re
from typing import Any

from src.models.episode import Episode
from src.models.search import PreconditionCheckResult

logger = logging.getLogger(__name__)


class PreconditionMatcher:
    """
    Matches current state against episode preconditions.

    Fast heuristic-based matching without LLM calls (LLM matching is Phase 2+).
    """

    def __init__(self):
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
