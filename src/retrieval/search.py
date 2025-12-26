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

import logging
import time
from datetime import datetime, timezone
from threading import Lock

from src.config import get_settings
from src.ingestion.embedding import EmbeddingService
from src.kyrodb.router import KyroDBRouter
from src.models.episode import Episode
from src.models.search import SearchRequest, SearchResponse, SearchResult
from src.retrieval.preconditions import (
    AdvancedPreconditionMatcher,
    PreconditionMatcher,
    get_advanced_precondition_matcher,
)
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



    def __init__(
        self,
        kyrodb_router: KyroDBRouter,
        embedding_service: EmbeddingService,
        precondition_matcher: PreconditionMatcher | None = None,
        advanced_precondition_matcher: AdvancedPreconditionMatcher | None = None,
        ranker: EpisodeRanker | None = None,
    ):
        """
        Initialize search pipeline.

        Args:
            kyrodb_router: KyroDB router for dual-instance search
            embedding_service: Multi-modal embedding service
            precondition_matcher: Optional basic precondition matcher (creates default if None)
            advanced_precondition_matcher: Optional LLM-based precondition matcher
            ranker: Optional ranker (creates default if None)
        """
        self.kyrodb_router = kyrodb_router
        self.embedding_service = embedding_service
        self.precondition_matcher = precondition_matcher or PreconditionMatcher()
        self.advanced_precondition_matcher = advanced_precondition_matcher
        self.ranker = ranker or EpisodeRanker()

        # Initialize from config if advanced matcher not provided
        settings = get_settings()
        if settings.search.enable_llm_validation and self.advanced_precondition_matcher is None:
            try:
                self.advanced_precondition_matcher = get_advanced_precondition_matcher(
                    openrouter_api_key=settings.llm.openrouter_api_key,
                    enable_llm=True
                )
                logger.info("LLM-based semantic validation enabled for search pipeline")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM validation: {e}. Falling back to basic matcher.")
                self.advanced_precondition_matcher = None

        # Thread-safe metrics tracking
        self._metrics_lock = Lock()
        self.total_searches = 0
        self.total_latency_ms = 0.0
        self.llm_validation_calls = 0
        self.llm_rejections = 0
        self.llm_latency_ms = 0.0

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
            latency_breakdown["embedding_ms"] = (time.perf_counter() - stage_start) * 1000

            # Stage 2: KyroDB search with server-side filtering (~1-3ms)
            stage_start = time.perf_counter()
            kyrodb_filters = self._build_kyrodb_filters(request)
            # Fetch 2x buffer: headroom for incomplete server-side filtering + precondition matching
            # Validate fetch_k to prevent memory issues with very large k values
            settings = get_settings()
            max_fetch_k = settings.search.max_k * 2
            fetch_k = min(request.k * 2, max_fetch_k)
            
            if request.k * 2 > max_fetch_k:
                logger.warning(
                    f"Requested k={request.k} would fetch {request.k * 2} candidates, "
                    f"capping at max_fetch_k={max_fetch_k} to prevent memory issues"
                )
            
            candidates = await self._fetch_candidates(
                query_embedding=query_embedding,
                customer_id=request.customer_id,
                collection=request.collection,
                k=fetch_k,
                min_similarity=request.min_similarity,
                metadata_filters=kyrodb_filters,
            )
            latency_breakdown["search_ms"] = (time.perf_counter() - stage_start) * 1000

            total_candidates = len(candidates)
            logger.debug(f"Fetched {total_candidates} candidates from KyroDB")

            # Stage 3: Metadata validation (safety check)
            stage_start = time.perf_counter()
            filtered_candidates = self._validate_metadata_filters(candidates, request)
            latency_breakdown["validation_ms"] = (time.perf_counter() - stage_start) * 1000

            total_filtered = len(filtered_candidates)
            if total_filtered < total_candidates:
                logger.warning(
                    f"Server-side filtering incomplete: {total_candidates - total_filtered} "
                    f"candidates removed by client validation. Check KyroDB metadata filters."
                )
            logger.debug(f"Validated {total_filtered} candidates")

            # Stage 4: Precondition matching (~5-15ms for 25 candidates)
            stage_start = time.perf_counter()
            precondition_results = await self._check_preconditions(filtered_candidates, request)
            latency_breakdown["precondition_ms"] = (time.perf_counter() - stage_start) * 1000

            # Stage 5: Ranking (~1-2ms)
            stage_start = time.perf_counter()
            ranked_results = self._rank_results(precondition_results, request)
            latency_breakdown["ranking_ms"] = (time.perf_counter() - stage_start) * 1000

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
                searched_at=datetime.now(UTC),
            )

            logger.info(
                f"Search completed: {len(final_results)} results, "
                f"{total_latency_ms:.2f}ms total "
                f"(embed: {latency_breakdown['embedding_ms']:.2f}ms, "
                f"search: {latency_breakdown['search_ms']:.2f}ms, "
                f"validation: {latency_breakdown['validation_ms']:.2f}ms, "
                f"precond: {latency_breakdown['precondition_ms']:.2f}ms, "
                f"rank: {latency_breakdown['ranking_ms']:.2f}ms)"
            )

            return response

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            raise

    def _build_kyrodb_filters(self, request: SearchRequest) -> dict[str, str]:
        """
        Build KyroDB metadata filters from search request.
        
        Converts SearchRequest filters to KyroDB-compatible string-string map.
        
        Args:
            request: Search request with filter criteria
            
        Returns:
            dict[str, str]: Metadata filters for KyroDB
        """
        filters = {}
        
        if request.tool_filter:
            filters["tool"] = request.tool_filter.lower()
            
        if request.min_timestamp is not None:
            filters["min_timestamp"] = str(request.min_timestamp)
            
        if request.max_timestamp is not None:
            filters["max_timestamp"] = str(request.max_timestamp)
            
        if request.tags:
            filters["tags"] = ",".join(sorted(request.tags))
            
        return filters

    async def _fetch_candidates(
        self,
        query_embedding: list[float],
        customer_id: str,
        collection: str,
        k: int,
        min_similarity: float,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[tuple[Episode, float]]:
        """
        Fetch candidate episodes from KyroDB with server-side filtering.

        Args:
            query_embedding: Query embedding vector
            customer_id: Customer ID for namespace isolation
            collection: Collection name (failures, skills, rules)
            k: Number of candidates to fetch
            min_similarity: Minimum similarity threshold
            metadata_filters: Server-side metadata filters

        Returns:
            list[tuple[Episode, float]]: List of (episode, similarity_score) pairs
        """
        # Security: Verify customer_id is provided (prevent data leakage)
        if not customer_id:
            raise ValueError(
                "customer_id is required for search - multi-tenancy violation detected"
            )

        # Search with server-side metadata filtering
        search_results = await self.kyrodb_router.search_text(
            query_embedding=query_embedding,
            k=k,
            customer_id=customer_id,
            collection=collection,
            min_score=min_similarity,
            metadata_filters=metadata_filters,
        )

        # Parse episode metadata from KyroDB results
        candidates = []
        for result in search_results.results:
            try:
                # Deserialize episode from metadata
                episode = Episode.from_metadata_dict(doc_id=result.doc_id, metadata=result.metadata)
                similarity_score = result.score

                candidates.append((episode, similarity_score))

            except Exception as e:
                logger.warning(
                    f"Failed to parse episode from search result {result.doc_id}: {e}",
                    exc_info=True,
                )
                # Skip malformed results

        return candidates

    def _validate_metadata_filters(
        self,
        candidates: list[tuple[Episode, float]],
        request: SearchRequest,
    ) -> list[tuple[Episode, float]]:
        """
        Validate server-side metadata filtering (safety check).
        
        This is a defensive check to ensure KyroDB's metadata filtering worked correctly.
        Under normal operation, this should not filter out any candidates.
        
        Args:
            candidates: List of (episode, score) pairs from KyroDB
            request: Search request with filter criteria
            
        Returns:
            list[tuple[Episode, float]]: Validated candidates
        """
        original_count = len(candidates)
        filtered = candidates

        # Validate tool filter
        if request.tool_filter:
            tool_filter_lower = request.tool_filter.lower()
            filtered = [
                (ep, score)
                for ep, score in filtered
                if ep.create_data.tool_chain
                and ep.create_data.tool_chain[0].lower() == tool_filter_lower
            ]

        # Validate timestamp range
        if request.min_timestamp is not None:
            min_dt = datetime.fromtimestamp(request.min_timestamp, tz=UTC)
            filtered = [(ep, score) for ep, score in filtered if ep.created_at >= min_dt]

        if request.max_timestamp is not None:
            max_dt = datetime.fromtimestamp(request.max_timestamp, tz=UTC)
            filtered = [(ep, score) for ep, score in filtered if ep.created_at <= max_dt]

        # Validate tags
        if request.tags:
            required_tags = {tag.lower() for tag in request.tags}
            filtered = [
                (ep, score)
                for ep, score in filtered
                if required_tags.issubset({tag.lower() for tag in ep.create_data.tags})
            ]

        # Log if validation removed candidates (indicates KyroDB filtering incomplete)
        removed_count = original_count - len(filtered)
        if removed_count > 0:
            logger.warning(
                f"Validation removed {removed_count}/{original_count} candidates. "
                f"KyroDB metadata filtering may be incomplete. Filters: {request.tool_filter}, "
                f"timestamps: [{request.min_timestamp}, {request.max_timestamp}], tags: {request.tags}"
            )

        return filtered

    async def _check_preconditions(
        self,
        candidates: list[tuple[Episode, float]],
        request: SearchRequest,
    ) -> list[tuple[Episode, float, float, list[str]]]:
        """
        Check preconditions for all candidates with optional LLM semantic validation.
        
        Two-stage approach:
        1. Basic heuristic matching (fast)
        2. LLM semantic validation for high-similarity candidates (if enabled)

        Args:
            candidates: List of (episode, similarity_score) pairs
            request: Search request with current_state

        Returns:
            list[tuple[Episode, similarity_score, precondition_score, matched_list]]:
                List of candidates with precondition scores
        """
        results = []
        settings = get_settings()
        llm_similarity_threshold = settings.search.llm_similarity_threshold

        for episode, similarity_score in candidates:
            # Stage 1: Basic heuristic precondition matching
            precond_result = self.precondition_matcher.check_preconditions(
                episode=episode,
                current_state=request.current_state,
                threshold=request.precondition_threshold,
            )

            # Only include episodes that meet basic precondition threshold
            if precond_result.matched:
                # Stage 2: LLM semantic validation (if enabled and high similarity)
                should_validate_with_llm = (
                    self.advanced_precondition_matcher is not None
                    and similarity_score >= llm_similarity_threshold
                )
                
                if should_validate_with_llm:
                    try:
                        llm_start = time.perf_counter()
                        llm_result = await self.advanced_precondition_matcher.check_preconditions_with_llm(
                            candidate_episode=episode,
                            current_query=request.goal,
                            current_state=request.current_state,
                            similarity_score=similarity_score,
                            threshold=request.precondition_threshold,
                        )
                        llm_latency = (time.perf_counter() - llm_start) * 1000
                        
                        # Track LLM metrics (thread-safe)
                        with self._metrics_lock:
                            self.llm_validation_calls += 1
                            self.llm_latency_ms += llm_latency
                        
                        if not llm_result.matched:
                            with self._metrics_lock:
                                self.llm_rejections += 1
                            logger.debug(
                                f"Episode {episode.episode_id} rejected by LLM validation "
                                f"(similarity={similarity_score:.3f}): {llm_result.explanation}"
                            )
                            continue  # Skip this candidate
                        
                    except Exception as e:
                        logger.warning(
                            f"LLM validation failed for episode {episode.episode_id}: {e}. "
                            f"Falling back to basic match."
                        )
                        # Gracefully degrade to basic match
                
                # Accept candidate (passed both stages or LLM not enabled)
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
                    f"Episode {episode.episode_id} filtered by basic preconditions: "
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
        matched_preconditions_list = [matched for _, _, _, matched in precondition_results]

        # Rank using weighted scoring
        ranked_results = self.ranker.rank_episodes(
            episodes=episodes,
            similarity_scores=similarity_scores,
            precondition_scores=precondition_scores,
            matched_preconditions_list=matched_preconditions_list,
            weights=request.ranking_weights,
            current_time=datetime.now(UTC),
        )

        return ranked_results

    def get_stats(self) -> dict[str, float]:
        """
        Get search statistics including LLM validation metrics.

        Returns:
            dict: Stats including total searches, latency, and LLM validation metrics
        """
        avg_latency = (
            self.total_latency_ms / self.total_searches if self.total_searches > 0 else 0.0
        )
        
        # Thread-safe metrics read
        with self._metrics_lock:
            llm_calls = self.llm_validation_calls
            llm_latency = self.llm_latency_ms
            llm_rejects = self.llm_rejections
        
        avg_llm_latency = (
            llm_latency / llm_calls if llm_calls > 0 else 0.0
        )
        llm_rejection_rate = (
            llm_rejects / llm_calls if llm_calls > 0 else 0.0
        )

        stats = {
            "total_searches": self.total_searches,
            "avg_latency_ms": avg_latency,
            "llm_validation_calls": llm_calls,
            "llm_rejections": llm_rejects,
            "llm_rejection_rate": llm_rejection_rate,
            "avg_llm_latency_ms": avg_llm_latency,
        }
        
        # Add advanced matcher stats if available
        if self.advanced_precondition_matcher:
            advanced_stats = self.advanced_precondition_matcher.get_stats()
            stats["llm_cache_hits"] = advanced_stats.get("cache_hits", 0)
            stats["llm_cache_hit_rate"] = advanced_stats.get("cache_hit_rate", 0.0)
            stats["llm_total_cost_usd"] = advanced_stats.get("total_cost_usd", 0.0)
        
        return stats
