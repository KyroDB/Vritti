"""
Data models for Pre-Action Gating system.

Defines the schema for the /reflect endpoint which acts as a safety gate
before agents execute actions.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.models.search import SearchResult


class ActionRecommendation(str, Enum):
    """Recommendation for proposed action."""
    
    BLOCK = "block"      # High confidence this will fail, don't execute
    REWRITE = "rewrite"  # Likely to fail, suggest alternative
    HINT = "hint"        # Might fail, show hints
    PROCEED = "proceed"  # No known issues


class ReflectRequest(BaseModel):
    """
    Request to reflect before action.
    
    Agents call this BEFORE executing potentially risky actions.
    """
    
    goal: str = Field(..., description="The high-level goal the agent is trying to achieve")
    proposed_action: str = Field(..., description="The specific action/command the agent wants to take")
    tool: str = Field(..., description="Tool being used (e.g., 'kubectl', 'docker', 'git')")
    current_state: dict[str, Any] = Field(default_factory=dict, description="Current environment state (OS, versions, etc.)")
    context: str | None = Field(default=None, description="Additional context or previous steps")


class ReflectResponse(BaseModel):
    """
    Response with action recommendation.
    """
    
    recommendation: ActionRecommendation
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the recommendation")
    rationale: str = Field(..., description="Why this recommendation was made")

    # Matched failures that led to this recommendation
    matched_failures: list[SearchResult] = Field(default_factory=list)

    # Suggested alternatives (if REWRITE)
    suggested_action: str | None = Field(default=None, description="Better alternative action if available")

    # Hints (if HINT)
    hints: list[str] = Field(default_factory=list, description="Helpful hints to avoid failure")

    # Matched skills (if any relevant skills found)
    # Using dict for now to avoid circular imports if Skill is not fully ready or if we want a simplified view
    relevant_skills: list[dict[str, Any]] = Field(default_factory=list)

    # Latency breakdown
    search_latency_ms: float
    total_latency_ms: float
