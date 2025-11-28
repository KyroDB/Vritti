"""
Structured Metadata Extraction for Goals.

Extracts structured metadata from natural language goals to enable fast
pre-filtering before expensive LLM validation.

Security:
- Input sanitization to prevent injection
- Max input length enforcement
- Validation of extracted metadata
"""

import logging
import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """Type of action in a goal."""
    DELETE = "delete"
    CREATE = "create"
    MODIFY = "modify"
    READ = "read"
    DEPLOY = "deploy"
    BUILD = "build"
    TEST = "test"
    UPLOAD = "upload"
    DOWNLOAD = "download"
    INSTALL = "install"
    UNINSTALL = "uninstall"
    START = "start"
    STOP = "stop"
    RESTART = "restart"
    UNKNOWN = "unknown"


class TimeCondition(BaseModel):
    """Time-based condition extracted from goal."""
    operator: str  # "older_than", "newer_than", "exactly", "between"
    value: int  # numeric value
    unit: str  # "days", "hours", "minutes", "months", "years"
    negated: bool = False  # if preceded by NOT/EXCEPT
    
    @validator('operator')
    def validate_operator(cls, v):
        """Validate operator is one of the allowed values."""
        allowed = {"older_than", "newer_than", "exactly", "between"}
        if v not in allowed:
            raise ValueError(f"Invalid operator: {v}. Must be one of {allowed}")
        return v
    
    @validator('unit')
    def validate_unit(cls, v):
        """Validate unit is one of the allowed time units."""
        allowed = {"day", "days", "hour", "hours", "minute", "minutes", "month", "months", "year", "years"}
        if v not in allowed:
            raise ValueError(f"Invalid unit: {v}. Must be one of {allowed}")
        return v
    
    @validator('value')
    def validate_value(cls, v):
        """Validate value is positive."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v


class StructuredGoal(BaseModel):
    """Structured representation of a goal."""
    
    # Core components
    action: ActionType
    target: str  # What's being acted on (e.g., "files", "deployment", "database")
    
    # Modifiers
    has_negation: bool = False  # If goal contains NOT/EXCEPT/WITHOUT
    environment: Optional[str] = None  # "production", "staging", "dev", etc.
    time_condition: Optional[TimeCondition] = None
    
    # Metadata
    original_text: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)  # Parser confidence
    
    @validator('target')
    def validate_target(cls, v):
        """Ensure target is not empty and reasonably sized."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Target cannot be empty")
        if len(v) > 200:
            raise ValueError("Target too long (max 200 chars)")
        return v.strip().lower()


class GoalParser:
    """
    Parse natural language goals into structured metadata.
    
    This provides fast heuristic-based extraction to pre-filter before
    expensive LLM validation. Not meant to be perfect, but catch obvious
    incompatibilities.
    
    Security:
    - Max input length: 500 chars
    - Input sanitization
    - No regex injection vulnerabilities
    """
    
    # Max input length for security
    MAX_INPUT_LENGTH = 500
    
    # Action detection patterns (order matters - more specific first)
    ACTION_PATTERNS = [
        (ActionType.DOWNLOAD, r'\b(download|fetch|pull|retrieve)\b'),
        (ActionType.UPLOAD, r'\b(upload|push|put|send)\b'),
        (ActionType.DEPLOY, r'\b(deploy|release|rollout|ship)\b'),  # Before CREATE/START
        (ActionType.UNINSTALL, r'\b(uninstall|remove|purge)\b'),
        (ActionType.INSTALL, r'\b(install|setup|provision)\b'),
        (ActionType.DELETE, r'\b(delete|drop|destroy|erase|rm|clean)\b'),  # Removed 'remove'
        (ActionType.BUILD, r'\b(build|compile|package)\b'),
        (ActionType.TEST, r'\b(test|verify|validate|check)\b'),
        (ActionType.RESTART, r'\b(restart|reboot|reload)\b'),
        (ActionType.MODIFY, r'\b(modify|update|change|edit|patch|alter)\b'),
        (ActionType.CREATE, r'\b(create|add|make|new|generate|init)\b'),
        (ActionType.START, r'\b(start|launch|run|execute|begin)\b'),
        (ActionType.STOP, r'\b(stop|halt|kill|terminate|end)\b'),
        (ActionType.READ, r'\b(read|view|show|list|get)\b'),
    ]
    
    # Negation patterns
    NEGATION_PATTERNS = [
        r'\bexcept\b',
        r'\bexcluding\b',
        r'\bnot\b',
        r'\bwithout\b',
        r'\bother than\b',
        r'\baside from\b',
        r'\bignore\b',
        r'\bskip\b',
    ]
    
    # Environment patterns
    ENVIRONMENT_PATTERNS = [
        (r'\b(prod|production)\b', 'production'),
        (r'\b(staging|stage|stg)\b', 'staging'),
        (r'\b(dev|development)\b', 'development'),
        (r'\b(test|testing|qa)\b', 'test'),
        (r'\b(local|localhost)\b', 'local'),
    ]
    
    # Time condition patterns
    TIME_PATTERNS = [
        # "older than X days"
        (r'older\s+than\s+(\d+)\s+(day|hour|minute|month|year)s?', 'older_than'),
        # "newer than X days"
        (r'newer\s+than\s+(\d+)\s+(day|hour|minute|month|year)s?', 'newer_than'),
        # "more than X days old"
        (r'more\s+than\s+(\d+)\s+(day|hour|minute|month|year)s?\s+old', 'older_than'),
        # "less than X days old"
        (r'less\s+than\s+(\d+)\s+(day|hour|minute|month|year)s?\s+old', 'newer_than'),
        # "X days ago"
        (r'(\d+)\s+(day|hour|minute|month|year)s?\s+ago', 'older_than'),
        # "within X days"
        (r'within\s+(\d+)\s+(day|hour|minute|month|year)s?', 'newer_than'),
    ]
    
    def parse(self, goal: str) -> StructuredGoal:
        """
        Parse goal into structured metadata.
        
        Args:
            goal: Natural language goal text
            
        Returns:
            StructuredGoal with extracted metadata
            
        Raises:
            ValueError: If input too long or invalid
        """
        # Security: Validate input length
        if not goal:
            raise ValueError("Goal cannot be empty")
        
        if len(goal) > self.MAX_INPUT_LENGTH:
            raise ValueError(f"Goal too long (max {self.MAX_INPUT_LENGTH} chars)")
        
        # Sanitize input (basic)
        goal_lower = goal.lower().strip()
        
        # Extract components
        action = self._extract_action(goal_lower)
        target = self._extract_target(goal_lower, action)
        has_negation = self._detect_negation(goal_lower)
        environment = self._extract_environment(goal_lower)
        time_condition = self._extract_time_condition(goal_lower)
        
        # Calculate confidence (simple heuristic)
        confidence = self._calculate_confidence(
            action, target, has_negation, environment, time_condition
        )
        
        return StructuredGoal(
            action=action,
            target=target,
            has_negation=has_negation,
            environment=environment,
            time_condition=time_condition,
            original_text=goal,
            confidence=confidence
        )
    
    def _extract_action(self, goal: str) -> ActionType:
        """Extract action type from goal."""
        for action_type, pattern in self.ACTION_PATTERNS:
            if re.search(pattern, goal, re.IGNORECASE):
                return action_type
        return ActionType.UNKNOWN
    
    def _extract_target(self, goal: str, action: ActionType) -> str:
        """
        Extract target of the action.
        
        Simple heuristic: nouns following the action verb.
        """
        # Common targets
        target_patterns = [
            r'\b(file|files)\b',
            r'\b(directory|directories|folder|folders)\b',
            r'\b(database|db|table|tables)\b',
            r'\b(deployment|deploy)\b',
            r'\b(container|containers|pod|pods)\b',
            r'\b(service|services)\b',
            r'\b(package|packages)\b',
            r'\b(image|images)\b',
            r'\b(user|users|account|accounts)\b',
            r'\b(config|configuration|settings)\b',
        ]
        
        for pattern in target_patterns:
            match = re.search(pattern, goal, re.IGNORECASE)
            if match:
                return match.group(0).lower()
        
        # Fallback: generic target
        return "resource"
    
    def _detect_negation(self, goal: str) -> bool:
        """Detect if goal contains negation keywords."""
        for pattern in self.NEGATION_PATTERNS:
            if re.search(pattern, goal, re.IGNORECASE):
                return True
        return False
    
    def _extract_environment(self, goal: str) -> Optional[str]:
        """Extract environment from goal."""
        for pattern, env_name in self.ENVIRONMENT_PATTERNS:
            if re.search(pattern, goal, re.IGNORECASE):
                return env_name
        return None
    
    def _extract_time_condition(self, goal: str) -> Optional[TimeCondition]:
        """Extract time-based conditions."""
        for pattern, operator in self.TIME_PATTERNS:
            match = re.search(pattern, goal, re.IGNORECASE)
            if match:
                value = int(match.group(1))
                unit = match.group(2).lower()
                
                # Check if negated (EXCEPT/NOT before time condition)
                # Look backwards from match position
                pre_text = goal[:match.start()].lower()
                negated = any(
                    re.search(neg_pattern, pre_text[-50:])  # Last 50 chars
                    for neg_pattern in self.NEGATION_PATTERNS
                )
                
                return TimeCondition(
                    operator=operator,
                    value=value,
                    unit=unit,
                    negated=negated
                )
        
        return None
    
    def _calculate_confidence(
        self,
        action: ActionType,
        target: str,
        has_negation: bool,
        environment: Optional[str],
        time_condition: Optional[TimeCondition]
    ) -> float:
        """
        Calculate parser confidence score.
        
        Higher confidence when we detect specific components.
        """
        confidence = 0.2  # Base confidence (lowered)
        
        if action != ActionType.UNKNOWN:
            confidence += 0.2  # Lowered from 0.3
        
        if target != "resource":  # Specific target detected
            confidence += 0.15  # Lowered from 0.2
        
        if environment is not None:
            confidence += 0.1
        
        if time_condition is not None:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def goals_compatible(
        self,
        goal1: StructuredGoal,
        goal2: StructuredGoal,
        strict: bool = False
    ) -> bool:
        """
        Fast heuristic check if two goals are compatible.
        
        This is a pre-filter before expensive LLM validation.
        
        Args:
            goal1: First goal
            goal2: Second goal
            strict: If True, require exact matches
            
        Returns:
            True if goals appear compatible, False if obviously incompatible
        """
        # Check 1: Opposite actions
        opposite_actions = [
            (ActionType.DELETE, ActionType.CREATE),
            (ActionType.START, ActionType.STOP),
            (ActionType.INSTALL, ActionType.UNINSTALL),
            (ActionType.UPLOAD, ActionType.DOWNLOAD),
        ]
        
        for action_a, action_b in opposite_actions:
            if (goal1.action == action_a and goal2.action == action_b) or \
               (goal1.action == action_b and goal2.action == action_a):
                logger.debug(
                    f"Goals incompatible: opposite actions {goal1.action} vs {goal2.action}"
                )
                return False
        
        # Check 2: Different environments (if both specified)
        if goal1.environment and goal2.environment:
            if goal1.environment != goal2.environment:
                logger.debug(
                    f"Goals incompatible: different environments "
                    f"{goal1.environment} vs {goal2.environment}"
                )
                return False
        
        # Check 3: Negation mismatch
        if goal1.has_negation != goal2.has_negation:
            logger.debug(
                f"Goals potentially incompatible: negation mismatch "
                f"({goal1.has_negation} vs {goal2.has_negation})"
            )
            # Don't reject yet - let LLM validate
            # This is a warning flag
        
        # Check 4: Time conditions (if both present)
        if goal1.time_condition and goal2.time_condition:
            tc1, tc2 = goal1.time_condition, goal2.time_condition
            
            # Opposite operators
            if (tc1.operator == "older_than" and tc2.operator == "newer_than") or \
               (tc1.operator == "newer_than" and tc2.operator == "older_than"):
                logger.debug(
                    f"Goals incompatible: opposite time conditions "
                    f"{tc1.operator} vs {tc2.operator}"
                )
                return False
            
            # Negation mismatch on same operator
            if tc1.operator == tc2.operator and tc1.negated != tc2.negated:
                logger.debug(
                    "Goals incompatible: time condition negation mismatch"
                )
                return False
        
        # Passed all checks - appears compatible
        return True
