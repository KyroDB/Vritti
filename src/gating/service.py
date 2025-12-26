"""
Gating Service for Pre-Action Validation.

Analyzes proposed actions against historical episodes to prevent repeat failures.
"""

import logging
import time
from typing import Any, NamedTuple, Optional

from src.kyrodb.router import KyroDBRouter
from src.models.gating import ActionRecommendation, ReflectRequest, ReflectResponse
from src.models.search import SearchRequest, SearchResult
from src.models.skill import Skill
from src.observability.metrics import (
    track_gating_decision,
    track_repeat_error_prevented,
)
from src.retrieval.search import SearchPipeline

logger = logging.getLogger(__name__)


class GatingRecommendationResult(NamedTuple):
    """Result of gating recommendation analysis.
    
    Attributes:
        recommendation: The action recommendation (PROCEED, BLOCK, REWRITE, HINT)
        confidence: Confidence score (0.0-1.0) in the recommendation
        rationale: Human-readable explanation of the decision
        suggested_action: Optional alternative action to suggest
        hints: List of helpful hints or warnings
    """
    recommendation: ActionRecommendation
    confidence: float
    rationale: str
    suggested_action: Optional[str]
    hints: list[str]


class GatingService:
    """
    Service for pre-action gating.
    
    Orchestrates search for failures and skills, then applies logic to
    determine if an action should be blocked, rewritten, or allowed.
    """

    # Confidence thresholds for gating decisions
    MIN_SIMILARITY_SEARCH = 0.6  # Minimum similarity to consider a match
    SKILL_HIGH_CONFIDENCE = 0.85  # Skill similarity to suggest using it
    BLOCK_SIMILARITY = 0.9  # Similarity threshold for blocking
    BLOCK_PRECONDITION = 0.7  # Precondition match threshold for blocking
    REWRITE_SIMILARITY = 0.8  # Similarity threshold for rewrite suggestion
    REWRITE_PRECONDITION = 0.5  # Precondition match threshold for rewrite
    HINT_SIMILARITY = 0.7  # Similarity threshold for showing hints

    def __init__(self, search_pipeline: SearchPipeline, kyrodb_router: KyroDBRouter):
        self.search_pipeline = search_pipeline
        self.kyrodb_router = kyrodb_router

    async def reflect_before_action(
        self, request: ReflectRequest, customer_id: str
    ) -> ReflectResponse:
        """
        Reflect before executing action.

        Args:
            request: The reflection request containing goal and proposed action.
            customer_id: The authenticated customer ID.

        Returns:
            ReflectResponse with recommendation.
        """
        start_time = time.perf_counter()
        search_latency_ms = 0.0

        try:
            # 1. Search for similar failures
            # We use the proposed action + goal as the query for better context
            search_query = f"{request.goal} {request.proposed_action}"
            
            search_req = SearchRequest(
                customer_id=customer_id,
                goal=search_query,
                current_state=request.current_state,
                k=5,  # We only need top matches for gating
                min_similarity=self.MIN_SIMILARITY_SEARCH
            )

            search_start = time.perf_counter()
            search_response = await self.search_pipeline.search(search_req)
            
            # 2. Search for relevant skills (if any)
            # We need to embed the query first - using search pipeline's embedding service
            # This assumes search_pipeline has access to embedding service
            query_embedding = self.search_pipeline.embedding_service.embed_text(search_query)
            
            matched_skills = await self.kyrodb_router.search_skills(
                query_embedding=query_embedding,
                customer_id=customer_id,
                k=3,
                min_score=0.7
            )
            
            search_latency_ms = (time.perf_counter() - search_start) * 1000

            # 3. Determine recommendation
            result = self._determine_gating_recommendation(
                request.proposed_action,
                search_response.results,
                matched_skills,
                request.current_state
            )

            total_latency_ms = (time.perf_counter() - start_time) * 1000

            # Convert skills to dict for response
            skills_dicts = [
                skill.to_metadata_dict() for skill, _ in matched_skills
            ]

            # Track gating decision metrics
            track_gating_decision(
                recommendation=result.recommendation.value,
                customer_tier="default",  # TODO: Pass customer tier from context
                confidence=result.confidence,
                latency_seconds=total_latency_ms / 1000.0,
                matched_failures=len(search_response.results),
                matched_skills=len(matched_skills),
            )

            # Track repeat error prevention
            if (result.recommendation in [ActionRecommendation.BLOCK, ActionRecommendation.REWRITE, ActionRecommendation.HINT] and
                len(search_response.results) > 0):
                top_match = search_response.results[0]
                error_class = "unknown"
                
                # Defensive extraction of error_class with proper null checks
                try:
                    if (top_match.episode and 
                        top_match.episode.create_data and 
                        hasattr(top_match.episode.create_data, 'error_class') and
                        top_match.episode.create_data.error_class is not None):
                        
                        error_class_obj = top_match.episode.create_data.error_class
                        
                        # Check if it's an Enum with .value attribute
                        if hasattr(error_class_obj, 'value'):
                            error_class = str(error_class_obj.value)
                        else:
                            # Coerce to string if not an Enum
                            error_class = str(error_class_obj)
                except Exception as e:
                    # Log but don't fail - metrics are critical
                    logger.debug(f"Could not extract error_class from match: {e}, using 'unknown'")
                    error_class = "unknown"

                track_repeat_error_prevented(
                    customer_id=customer_id,
                    customer_tier="default",
                    error_class=error_class,
                    recommendation=result.recommendation.value,
                )

            return ReflectResponse(
                recommendation=result.recommendation,
                confidence=result.confidence,
                rationale=result.rationale,
                matched_failures=search_response.results,
                suggested_action=result.suggested_action,
                hints=result.hints,
                relevant_skills=skills_dicts,
                search_latency_ms=search_latency_ms,
                total_latency_ms=total_latency_ms
            )

        except Exception as e:
            logger.error(f"Pre-action gating failed: {e}", exc_info=True)
            # Fail open (PROCEED) but with low confidence
            return ReflectResponse(
                recommendation=ActionRecommendation.PROCEED,
                confidence=0.0,
                rationale=f"Gating service error: {str(e)}",
                matched_failures=[],
                search_latency_ms=0.0,
                total_latency_ms=(time.perf_counter() - start_time) * 1000
            )

    def _determine_gating_recommendation(
        self,
        _proposed_action: str,
        matched_failures: list[SearchResult],
        matched_skills: list[tuple[Skill, float]],
        _current_state: dict[str, Any]
    ) -> GatingRecommendationResult:
        """
        Determine gating recommendation based on matched failures and skills.

        Priority order:
        1. Skills (proven solutions) - suggest REWRITE if high confidence
        2. Failures - BLOCK/REWRITE/HINT based on confidence
        3. Default - PROCEED if no matches
        
        Args:
            _proposed_action: The proposed action (unused - reserved for future context-aware gating)
            matched_failures: List of matched failure episodes from search
            matched_skills: List of matched skills with confidence scores
            _current_state: Current state dict (unused - reserved for future state-aware gating)
        
        Returns:
            GatingRecommendationResult with recommendation details
        """
        
        # 1. Check for high-confidence Skills first (proven solutions)
        # Skills take priority because they represent successful patterns
        if matched_skills:
            top_skill, score = matched_skills[0]
            if score >= self.SKILL_HIGH_CONFIDENCE:
                # We have a proven solution with high confidence
                # Suggest using the skill instead of the proposed action
                return GatingRecommendationResult(
                    recommendation=ActionRecommendation.REWRITE,
                    confidence=0.9,
                    rationale=f"Found proven solution: '{top_skill.name}' (used {top_skill.usage_count}× successfully)",
                    suggested_action=top_skill.code,  # Suggest the skill's code
                    hints=[
                        f"Success rate: {top_skill.success_rate * 100:.0f}%",
                        f"Skill documentation: {top_skill.docstring}"
                    ]
                )

        # 2. No high-confidence skills found, check failures
        if not matched_failures:
            # No failures and no skills - safe to proceed
            hints = []
            if matched_skills:
                # Low-confidence skill exists, mention as a hint
                skill, score = matched_skills[0]
                hints.append(f"Related skill available: {skill.name} (confidence: {score:.2f})")
            
            return GatingRecommendationResult(
                recommendation=ActionRecommendation.PROCEED,
                confidence=1.0,
                rationale="No similar past failures found.",
                suggested_action=None,
                hints=hints
            )

        top_match = matched_failures[0]
        similarity_score = top_match.scores.get("similarity", 0.0)
        precondition_score = top_match.scores.get("precondition", 0.0)
        
        # Extract reflection data safely
        root_cause = "Unknown"
        resolution = None
        if top_match.episode.reflection:
            root_cause = top_match.episode.reflection.root_cause
            resolution = top_match.episode.reflection.resolution_strategy

        # 3. Check for BLOCK (highest confidence failure match)
        if (similarity_score >= self.BLOCK_SIMILARITY and
            precondition_score >= self.BLOCK_PRECONDITION):
            
            rationale = (
                f"High risk: Similar action failed previously ({similarity_score:.2f} similarity, "
                f"{precondition_score:.2f} precondition match). Root cause: {root_cause}"
            )
            
            return GatingRecommendationResult(
                recommendation=ActionRecommendation.BLOCK,
                confidence=0.95,
                rationale=rationale,
                suggested_action=resolution,  # Suggest the fix from the failure
                hints=[f"Previous failure: {top_match.episode.create_data.goal}"]
            )

        # 4. Check for REWRITE (medium-high confidence)
        elif (similarity_score >= self.REWRITE_SIMILARITY and
              precondition_score >= self.REWRITE_PRECONDITION and
              resolution is not None):
            
            rationale = (
                f"Action likely to fail ({similarity_score:.2f} similarity). "
                f"Suggested alternative available. Root cause: {root_cause}"
            )
            
            return GatingRecommendationResult(
                recommendation=ActionRecommendation.REWRITE,
                confidence=0.85,
                rationale=rationale,
                suggested_action=resolution,
                hints=[f"Based on failure: {top_match.episode.create_data.goal}"]
            )

        # 5. Check for HINT (medium confidence)
        elif similarity_score >= self.HINT_SIMILARITY:
            hints = [
                f"Watch out for: {root_cause}",
                f"Previous resolution: {resolution if resolution else 'N/A'}"
            ]
            
            return GatingRecommendationResult(
                recommendation=ActionRecommendation.HINT,
                confidence=0.7,
                rationale="Similar failures detected, proceed with caution.",
                suggested_action=None,
                hints=hints
            )

        # 6. PROCEED (low confidence / weak match)
        else:
            return GatingRecommendationResult(
                recommendation=ActionRecommendation.PROCEED,
                confidence=0.8,
                rationale="Low similarity to past failures.",
                suggested_action=None,
                hints=[]
            )
