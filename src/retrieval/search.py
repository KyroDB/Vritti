"""
Search orchestrator for episodic memory retrieval.

Implements end-to-end retrieval pipeline:
1. Query embedding generation
2. KyroDB vector search (fetch k×5 candidates for filtering headroom)
3. Metadata filtering (tool, timestamp, tags)
4. Precondition matching (relevance filtering)
5. Weighted ranking (multi-signal scoring)
6. Top-k selection

Designed for <50ms P99 latency.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from src.ingestion.embedding import EmbeddingService
from src.kyrodb.router import KyroDBRouter
from src.models.episode import Episode
from src.models.search import SearchRequest, SearchResponse, SearchResult
from src.retrieval.preconditions import PreconditionMatcher
from src.retrieval.ranking import EpisodeRanker

logger = logging.getLogger(__name__)


class SearchPipeline:
    """
    Orchestrates end-to-end episodic memory retrieval.

    Pipeline stages:
    1. Embedding generation (5-10ms)
    2. KyroDB search (1-5ms per instance)
    3. Metadata filtering (1-2ms)
    4. Precondition matching (5-15ms for 25 candidates)
    5. Ranking (1-2ms)

    Total target: <50ms P99
    """

    # Candidate expansion factor for metadata filtering
    CANDIDATE_EXPANSION_FACTOR = 5

    def __init__(
        self,
        kyrodb_router: KyroDBRouter,
        embedding_service: EmbeddingService,
        precondition_matcher: Optional[PreconditionMatcher] = None,
        ranker: Optional[EpisodeRanker] = None,
    ):
        """
        Initialize search pipeline.

        Args:
            kyrodb_router: KyroDB router for dual-instance search
            embedding_service: Multi-modal embedding service
            precondition_matcher: Optional precondition matcher (creates default if None)
            ranker: Optional ranker (creates default if None)
        """
        self.kyrodb_router = kyrodb_router
        self.embedding_service = embedding_service
        self.precondition_matcher = precondition_matcher or PreconditionMatcher()
        self.ranker = ranker or EpisodeRanker()

        self.total_searches = 0
        self.total_latency_ms = 0.0

    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Execute search pipeline.

        Args:
            request: Search request with query and parameters

        Returns:
            SearchResponse: Ranked search results with latency breakdown

        Raises:
            ValueError: If request validation fails
            KyroDBError: If search fails critically
        """
        start_time = time.perf_counter()
        latency_breakdown = {}

        try:
            # Stage 1: Generate query embedding (~5-10ms)
            stage_start = time.perf_counter()
            query_embedding = self.embedding_service.embed_text(request.goal)
            latency_breakdown["embedding_ms"] = (
                time.perf_counter() - stage_start
            ) * 1000

            # Stage 2: KyroDB search with expansion (~1-5ms)
            stage_start = time.perf_counter()
            fetch_k = request.k * self.CANDIDATE_EXPANSION_FACTOR
            candidates = await self._fetch_candidates(
                query_embedding=query_embedding,
                customer_id=request.customer_id,
                collection=request.collection,
                k=fetch_k,
                min_similarity=request.min_similarity,
            )
            latency_breakdown["search_ms"] = (
                time.perf_counter() - stage_start
            ) * 1000

            total_candidates = len(candidates)
            logger.debug(f"Fetched {total_candidates} candidates from KyroDB")

            # Stage 3: Metadata filtering (~1-2ms)
            stage_start = time.perf_counter()
            filtered_candidates = self._apply_metadata_filters(candidates, request)
            latency_breakdown["filtering_ms"] = (
                time.perf_counter() - stage_start
            ) * 1000

            total_filtered = len(filtered_candidates)
            logger.debug(
                f"Filtered to {total_filtered} candidates "
                f"(removed {total_candidates - total_filtered})"
            )

            # Stage 4: Precondition matching (~5-15ms for 25 candidates)
            stage_start = time.perf_counter()
            precondition_results = await self._check_preconditions(
                filtered_candidates, request
            )
            latency_breakdown["precondition_ms"] = (
                time.perf_counter() - stage_start
            ) * 1000

            # Stage 5: Ranking (~1-2ms)
            stage_start = time.perf_counter()
            ranked_results = self._rank_results(precondition_results, request)
            latency_breakdown["ranking_ms"] = (
                time.perf_counter() - stage_start
            ) * 1000

            # Stage 6: Top-k selection
            final_results = ranked_results[: request.k]

            # Total latency
            total_latency_ms = (time.perf_counter() - start_time) * 1000

            # Update stats
            self.total_searches += 1
            self.total_latency_ms += total_latency_ms

            # Build response
            response = SearchResponse(
                results=final_results,
                total_candidates=total_candidates,
                total_filtered=total_filtered,
                total_returned=len(final_results),
                search_latency_ms=total_latency_ms,
                breakdown=latency_breakdown,
                collection=request.collection,
                query_embedding_dimension=len(query_embedding),
                searched_at=datetime.now(timezone.utc),
            )

            logger.info(
                f"Search completed: {len(final_results)} results, "
                f"{total_latency_ms:.2f}ms total "
                f"(embed: {latency_breakdown['embedding_ms']:.2f}ms, "
                f"search: {latency_breakdown['search_ms']:.2f}ms, "
                f"filter: {latency_breakdown['filtering_ms']:.2f}ms, "
                f"precond: {latency_breakdown['precondition_ms']:.2f}ms, "
                f"rank: {latency_breakdown['ranking_ms']:.2f}ms)"
            )

            return response

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            raise

    async def _fetch_candidates(
        self,
        query_embedding: list[float],
        customer_id: str,
        collection: str,
        k: int,
        min_similarity: float,
    ) -> list[tuple[Episode, float]]:
        """
        Fetch candidate episodes from KyroDB with customer namespace isolation.

        Multi-tenancy: Only searches within customer's namespaced collection.

        Args:
            query_embedding: Query embedding vector
            customer_id: Customer ID for namespace isolation
            collection: Collection name (failures, skills, rules)
            k: Number of candidates to fetch
            min_similarity: Minimum similarity threshold

        Returns:
            list[tuple[Episode, float]]: List of (episode, similarity_score) pairs

        Raises:
            ValueError: If customer_id is None or empty
        """
        # Security: Verify customer_id is provided (prevent data leakage)
        if not customer_id:
            raise ValueError(
                "customer_id is required for search - multi-tenancy violation detected"
            )

        # Search text instance with customer namespace isolation
        search_results = await self.kyrodb_router.search_text(
            query_embedding=query_embedding,
            k=k,
            customer_id=customer_id,
            collection=collection,
            min_score=min_similarity,
        )

        # Parse episode metadata from KyroDB results
        candidates = []
        for result in search_results.results:
            try:
                # Deserialize episode from metadata
                episode = Episode.from_metadata_dict(
                    doc_id=result.doc_id, metadata=result.metadata
                )
                similarity_score = result.score

                candidates.append((episode, similarity_score))

            except Exception as e:
                logger.warning(
                    f"Failed to parse episode from search result {result.doc_id}: {e}",
                    exc_info=True,
                )
                # Skip malformed results

        return candidates

    def _apply_metadata_filters(
        self,
        candidates: list[tuple[Episode, float]],
        request: SearchRequest,
    ) -> list[tuple[Episode, float]]:
        """
        Apply metadata filters to candidates.

        Filters:
        - Tool filter (exact match on primary tool)
        - Timestamp range (min_timestamp, max_timestamp)
        - Tags (all required tags must be present)

        Args:
            candidates: List of (episode, score) pairs
            request: Search request with filter criteria

        Returns:
            list[tuple[Episode, float]]: Filtered candidates
        """
        filtered = candidates

        # Filter by tool (primary tool in tool_chain)
        if request.tool_filter:
            tool_filter_lower = request.tool_filter.lower()
            filtered = [
                (ep, score)
                for ep, score in filtered
                if ep.create_data.tool_chain
                and ep.create_data.tool_chain[0].lower() == tool_filter_lower
            ]

        # Filter by timestamp range
        if request.min_timestamp is not None:
            min_dt = datetime.fromtimestamp(request.min_timestamp, tz=timezone.utc)
            filtered = [
                (ep, score) for ep, score in filtered if ep.created_at >= min_dt
            ]

        if request.max_timestamp is not None:
            max_dt = datetime.fromtimestamp(request.max_timestamp, tz=timezone.utc)
            filtered = [
                (ep, score) for ep, score in filtered if ep.created_at <= max_dt
            ]

        # Filter by tags (all required tags must be present)
        if request.tags:
            required_tags = set(tag.lower() for tag in request.tags)
            filtered = [
                (ep, score)
                for ep, score in filtered
                if required_tags.issubset(
                    set(tag.lower() for tag in ep.create_data.tags)
                )
            ]

        return filtered

    async def _check_preconditions(
        self,
        candidates: list[tuple[Episode, float]],
        request: SearchRequest,
    ) -> list[tuple[Episode, float, float, list[str]]]:
        """
        Check preconditions for all candidates.

        Args:
            candidates: List of (episode, similarity_score) pairs
            request: Search request with current_state

        Returns:
            list[tuple[Episode, similarity_score, precondition_score, matched_list]]:
                List of candidates with precondition scores
        """
        results = []

        for episode, similarity_score in candidates:
            # Check preconditions against current state
            precond_result = self.precondition_matcher.check_preconditions(
                episode=episode,
                current_state=request.current_state,
                threshold=request.precondition_threshold,
            )

            # Only include episodes that meet precondition threshold
            if precond_result.matched:
                results.append(
                    (
                        episode,
                        similarity_score,
                        precond_result.match_score,
                        precond_result.matched_preconditions,
                    )
                )
            else:
                logger.debug(
                    f"Episode {episode.episode_id} filtered by preconditions: "
                    f"{precond_result.explanation}"
                )

        return results

    def _rank_results(
        self,
        precondition_results: list[tuple[Episode, float, float, list[str]]],
        request: SearchRequest,
    ) -> list[SearchResult]:
        """
        Rank results using weighted scoring.

        Args:
            precondition_results: List of (episode, similarity, precondition_score, matched) tuples
            request: Search request with ranking weights

        Returns:
            list[SearchResult]: Ranked search results
        """
        if not precondition_results:
            return []

        # Separate components for ranker
        episodes = [ep for ep, _, _, _ in precondition_results]
        similarity_scores = [sim for _, sim, _, _ in precondition_results]
        precondition_scores = [precond for _, _, precond, _ in precondition_results]
        matched_preconditions_list = [
            matched for _, _, _, matched in precondition_results
        ]

        # Rank using weighted scoring
        ranked_results = self.ranker.rank_episodes(
            episodes=episodes,
            similarity_scores=similarity_scores,
            precondition_scores=precondition_scores,
            matched_preconditions_list=matched_preconditions_list,
            weights=request.ranking_weights,
            current_time=datetime.now(timezone.utc),
        )

        return ranked_results

    def get_stats(self) -> dict[str, float]:
        """
        Get search statistics.

        Returns:
            dict: Stats (total_searches, avg_latency_ms)
        """
        avg_latency = (
            self.total_latency_ms / self.total_searches
            if self.total_searches > 0
            else 0.0
        )

        return {
            "total_searches": self.total_searches,
            "avg_latency_ms": avg_latency,
        }
