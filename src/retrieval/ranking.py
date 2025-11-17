"""
Weighted ranking system for episodic memory retrieval.

Combines multiple signals for relevance scoring:
1. Semantic similarity (from KyroDB cosine similarity)
2. Precondition match score (from PreconditionMatcher)
3. Recency score (exponential time decay)
4. Usage score (retrieval count with diminishing returns)

Final score = w1*similarity + w2*precondition + w3*recency + w4*usage

Optimized for <5ms latency per ranking operation.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Optional

from src.models.episode import Episode
from src.models.search import RankingWeights, SearchResult

logger = logging.getLogger(__name__)


class EpisodeRanker:
    """
    Ranks episodes using weighted multi-signal scoring.

    Scoring components:
    - Similarity: Direct from KyroDB (cosine similarity, -1 to 1, typically 0-1)
    - Precondition: Match score from PreconditionMatcher (0-1)
    - Recency: Exponential decay based on episode age
    - Usage: Logarithmic scaling of retrieval_count

    All scores normalized to 0-1 range before weighting.
    """

    # Time decay constants
    RECENCY_HALF_LIFE_DAYS = 30.0  # Score halves every 30 days
    RECENCY_DECAY_RATE = math.log(2) / RECENCY_HALF_LIFE_DAYS

    # Usage scoring constants
    USAGE_LOG_BASE = 10.0  # log10 scaling for diminishing returns
    USAGE_MAX_COUNT = 1000.0  # Soft cap for normalization

    def __init__(self):
        """Initialize episode ranker."""
        pass

    def rank_episodes(
        self,
        episodes: list[Episode],
        similarity_scores: list[float],
        precondition_scores: list[float],
        matched_preconditions_list: list[list[str]],
        weights: RankingWeights,
        current_time: Optional[datetime] = None,
    ) -> list[SearchResult]:
        """
        Rank episodes by weighted multi-signal scoring.

        Args:
            episodes: List of candidate episodes
            similarity_scores: Semantic similarity scores from KyroDB (0-1)
            precondition_scores: Precondition match scores (0-1)
            matched_preconditions_list: List of matched preconditions per episode
            weights: Configurable weights for each signal
            current_time: Current time for recency calculation (default: now)

        Returns:
            list[SearchResult]: Ranked results sorted by final score (descending)

        Raises:
            ValueError: If input lists have mismatched lengths

        Example:
            >>> ranker = EpisodeRanker()
            >>> weights = RankingWeights(
            ...     similarity_weight=0.5,
            ...     precondition_weight=0.3,
            ...     recency_weight=0.1,
            ...     usage_weight=0.1
            ... )
            >>> results = ranker.rank_episodes(
            ...     episodes, sim_scores, precond_scores, matched_list, weights
            ... )
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Validate input lengths
        expected_len = len(episodes)
        if not (
            len(similarity_scores) == expected_len
            and len(precondition_scores) == expected_len
            and len(matched_preconditions_list) == expected_len
        ):
            raise ValueError(
                f"Input length mismatch: episodes={len(episodes)}, "
                f"similarity={len(similarity_scores)}, "
                f"precondition={len(precondition_scores)}, "
                f"matched_preconditions={len(matched_preconditions_list)}"
            )

        # Compute recency and usage scores
        recency_scores = [
            self._compute_recency_score(episode.created_at, current_time)
            for episode in episodes
        ]

        usage_scores = [
            self._compute_usage_score(episode.retrieval_count) for episode in episodes
        ]

        # Compute final weighted scores and build results
        results = []
        for i, episode in enumerate(episodes):
            # Weighted combination
            combined_score = (
                weights.similarity_weight * similarity_scores[i]
                + weights.precondition_weight * precondition_scores[i]
                + weights.recency_weight * recency_scores[i]
                + weights.usage_weight * usage_scores[i]
            )

            # Build score breakdown
            scores = {
                "similarity": similarity_scores[i],
                "precondition": precondition_scores[i],
                "recency": recency_scores[i],
                "usage": usage_scores[i],
                "combined": combined_score,
            }

            # Create search result
            result = SearchResult(
                episode=episode,
                scores=scores,
                rank=1,  # Temporary rank, will be reassigned after sorting
                matched_preconditions=matched_preconditions_list[i],
                similarity_explanation=self._generate_explanation(
                    similarity_scores[i],
                    precondition_scores[i],
                    recency_scores[i],
                    usage_scores[i],
                ),
            )
            results.append(result)

        # Sort by combined score (descending)
        results.sort(key=lambda r: r.scores["combined"], reverse=True)

        # Assign ranks (1-indexed)
        for rank, result in enumerate(results, start=1):
            result.rank = rank

        return results

    def _compute_recency_score(
        self, created_at: datetime, current_time: datetime
    ) -> float:
        """
        Compute recency score using exponential time decay.

        Score halves every RECENCY_HALF_LIFE_DAYS days (30 days default).
        Recent episodes score higher than old episodes.

        Args:
            created_at: Episode creation timestamp
            current_time: Current timestamp

        Returns:
            float: Recency score (0-1)

        Example:
            Episode created today: score ≈ 1.0
            Episode created 30 days ago: score ≈ 0.5
            Episode created 90 days ago: score ≈ 0.125
        """
        # Ensure both times are timezone-aware
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        # Calculate age in days
        age_seconds = (current_time - created_at).total_seconds()
        age_days = age_seconds / 86400.0  # 86400 seconds per day

        # Handle future timestamps (clock skew)
        if age_days < 0:
            logger.warning(
                f"Episode created in future: {created_at} vs {current_time}"
            )
            return 1.0  # Treat as very recent

        # Exponential decay: score = e^(-decay_rate * age_days)
        score = math.exp(-self.RECENCY_DECAY_RATE * age_days)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))

    def _compute_usage_score(self, retrieval_count: int) -> float:
        """
        Compute usage score with logarithmic scaling.

        Uses log10 to provide diminishing returns for high retrieval counts.
        Popular episodes score higher, but with diminishing returns.

        Args:
            retrieval_count: Number of times episode was retrieved

        Returns:
            float: Usage score (0-1)

        Example:
            0 retrievals → score 0.0
            10 retrievals → score ≈ 0.5
            100 retrievals → score ≈ 0.67
            1000+ retrievals → score ≈ 0.75 (soft cap)
        """
        if retrieval_count <= 0:
            return 0.0

        # Logarithmic scaling with soft cap
        # score = log10(1 + count) / log10(1 + max_count)
        score = math.log10(1 + retrieval_count) / math.log10(
            1 + self.USAGE_MAX_COUNT
        )

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))

    def _generate_explanation(
        self,
        similarity: float,
        precondition: float,
        recency: float,
        usage: float,
    ) -> str:
        """
        Generate human-readable explanation of ranking decision.

        Args:
            similarity: Similarity score (0-1)
            precondition: Precondition match score (0-1)
            recency: Recency score (0-1)
            usage: Usage score (0-1)

        Returns:
            str: Explanation of why this result was ranked

        Example:
            "Strong semantic match (0.89), high precondition match (0.75)"
        """
        parts = []

        # Similarity assessment
        if similarity >= 0.8:
            parts.append(f"strong semantic match ({similarity:.2f})")
        elif similarity >= 0.6:
            parts.append(f"good semantic match ({similarity:.2f})")
        else:
            parts.append(f"moderate semantic match ({similarity:.2f})")

        # Precondition assessment
        if precondition >= 0.7:
            parts.append(f"high precondition match ({precondition:.2f})")
        elif precondition >= 0.5:
            parts.append(f"partial precondition match ({precondition:.2f})")

        # Recency assessment
        if recency >= 0.9:
            parts.append("very recent")
        elif recency < 0.3:
            parts.append("older episode")

        # Usage assessment
        if usage >= 0.6:
            parts.append("frequently used")

        # Combine parts
        if not parts:
            return "Low overall relevance"

        return ", ".join(parts).capitalize()


# Singleton instance
_ranker: Optional[EpisodeRanker] = None


def get_ranker() -> EpisodeRanker:
    """
    Get global ranker instance.

    Returns:
        EpisodeRanker: Singleton instance
    """
    global _ranker
    if _ranker is None:
        _ranker = EpisodeRanker()
    return _ranker
