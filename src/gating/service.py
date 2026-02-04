"""
Gating Service for Pre-Action Validation.

Analyzes proposed actions against historical episodes to prevent repeat failures.
"""

import asyncio
import logging
import time
from typing import Any, NamedTuple


from src.kyrodb.router import KyroDBRouter
from src.models.gating import ActionRecommendation, ReflectRequest, ReflectResponse
from src.models.search import SearchRequest, SearchResult
from src.models.skill import Skill
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
    suggested_action: str | None
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
    
    # Display limits for hints and context
    MAX_ACTION_HINT_LENGTH = 100  # Maximum characters for action hints
    MAX_ENV_FACTORS_IN_HINTS = 3  # Maximum environment factors to show in hints

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
            embed_start = time.perf_counter()
            query_embedding = await self.search_pipeline.embedding_service.embed_text_async(
                search_query
            )
            embedding_ms = (time.perf_counter() - embed_start) * 1000

            search_task = asyncio.create_task(
                self.search_pipeline.search_with_embedding(
                    search_req,
                    query_embedding,
                    embedding_ms=embedding_ms,
                )
            )
            skills_task = asyncio.create_task(
                self.kyrodb_router.search_skills(
                    query_embedding=query_embedding,
                    customer_id=customer_id,
                    k=3,
                    min_score=0.7,
                )
            )

            search_response, matched_skills = await asyncio.gather(search_task, skills_task)
            
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
        proposed_action: str,
        matched_failures: list[SearchResult],
        matched_skills: list[tuple[Skill, float]],
        current_state: dict[str, Any]
    ) -> GatingRecommendationResult:
        """
        Determine gating recommendation based on matched failures and skills.

        Priority order:
        1. Skills (proven solutions) - suggest REWRITE if high confidence
        2. Failures - BLOCK/REWRITE/HINT based on confidence
        3. Default - PROCEED if no matches
        
        Args:
            proposed_action: The proposed action to evaluate
            matched_failures: List of matched failure episodes from search
            matched_skills: List of matched skills with confidence scores
            current_state: Current environment state for context-aware hints
        
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
                action_hint = proposed_action
                if len(action_hint) > self.MAX_ACTION_HINT_LENGTH:
                    action_hint = action_hint[: self.MAX_ACTION_HINT_LENGTH] + "..."
                return GatingRecommendationResult(
                    recommendation=ActionRecommendation.REWRITE,
                    confidence=0.9,
                    rationale=f"Found proven solution: '{top_skill.name}' (used {top_skill.usage_count}× successfully)",
                    suggested_action=top_skill.code,  # Suggest the skill's code
                    hints=[
                        f"Original action: {action_hint}",
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
        environment_factors = []
        if top_match.episode.reflection:
            root_cause = top_match.episode.reflection.root_cause
            resolution = top_match.episode.reflection.resolution_strategy
            environment_factors = top_match.episode.reflection.environment_factors
        
        # Check if current environment matches failure's environment factors
        environment_match = self._check_environment_match(current_state, environment_factors)

        # 3. Check for BLOCK (highest confidence failure match)
        if (similarity_score >= self.BLOCK_SIMILARITY and
            precondition_score >= self.BLOCK_PRECONDITION):
            
            rationale = (
                f"High risk: Similar action failed previously ({similarity_score:.2f} similarity, "
                f"{precondition_score:.2f} precondition match). Root cause: {root_cause}"
            )
            
            hints = [
                f"Previous failure: {top_match.episode.create_data.goal}",
                f"Blocked action: {proposed_action}",
            ]
            if environment_factors:
                if environment_match:
                    hints.append("Environment matches prior failure context")
                else:
                    hints.append("Environment differs from prior failure context")

            return GatingRecommendationResult(
                recommendation=ActionRecommendation.BLOCK,
                confidence=0.95,
                rationale=rationale,
                suggested_action=resolution,  # Suggest the fix from the failure
                hints=hints
            )

        # 4. Check for REWRITE (medium-high confidence)
        elif (similarity_score >= self.REWRITE_SIMILARITY and
              precondition_score >= self.REWRITE_PRECONDITION and
              resolution is not None):
            
            rationale = (
                f"Action likely to fail ({similarity_score:.2f} similarity). "
                f"Suggested alternative available. Root cause: {root_cause}"
            )
            
            hints = [f"Based on failure: {top_match.episode.create_data.goal}"]
            if environment_factors:
                if environment_match:
                    hints.append("Environment matches prior failure context")
                else:
                    hints.append("Environment differs from prior failure context")

            return GatingRecommendationResult(
                recommendation=ActionRecommendation.REWRITE,
                confidence=0.85,
                rationale=rationale,
                suggested_action=resolution,
                hints=hints
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

    def _check_environment_match(
        self, current_state: dict[str, Any], environment_factors: list[str]
    ) -> bool:
        """
        Check if current environment state matches the failure's environment factors.

        Args:
            current_state: Current environment state (OS, versions, tools, etc.)
            environment_factors: List of environment factors from the failure

        Returns:
            True if there's a reasonable match, False otherwise
        """
        if not current_state or not environment_factors:
            # If either is missing, we can't make a determination
            # Return True (assume match) to be conservative
            return True

        # Check if any environment factor appears in current state values
        # We check each value individually to avoid false positives from string concatenation
        matches = 0
        for factor in environment_factors:
            factor_lower = factor.lower()
            for value in current_state.values():
                if value is not None and factor_lower in str(value).lower():
                    matches += 1
                    break  # Found a match for this factor, move to next

        # Consider it a match if at least one factor is found
        return matches > 0
