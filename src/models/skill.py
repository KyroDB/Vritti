"""
Skill data model for promoted successful fixes.

Skills are reusable code patterns extracted from episodes with high success rates.
A skill is promoted when:
- 3+ episodes have same/similar fix
- Fix success rate > 90%
- Fix contains executable code or clear procedure

Security:
- All fields strictly validated
- Customer namespace isolation enforced
- Source episode tracking for audit
"""

import json
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class Skill(BaseModel):
    """
    Reusable code pattern or procedure extracted from successful fixes.

    Security: Strict validation prevents skill poisoning attacks.
    """

    skill_id: int = Field(..., description="Unique skill ID (KyroDB doc_id)")
    customer_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Customer namespace for multi-tenancy"
    )

    # Descriptive metadata
    name: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description="Auto-generated descriptive name"
    )

    docstring: str = Field(
        ...,
        min_length=20,
        max_length=2000,
        description="Human-readable description of what this skill does"
    )

    # The actual fix (one of these must be present)
    code: Optional[str] = Field(
        default=None,
        max_length=10000,
        description="Executable code snippet"
    )

    procedure: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Step-by-step procedure"
    )

    language: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Programming language (python, bash, yaml, etc.)"
    )

    # Success tracking
    source_episodes: list[int] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Episode IDs this skill was promoted from"
    )

    usage_count: int = Field(
        default=0,
        ge=0,
        description="How many times this skill was retrieved"
    )

    success_count: int = Field(
        default=0,
        ge=0,
        description="How many times applying this skill succeeded"
    )

    failure_count: int = Field(
        default=0,
        ge=0,
        description="How many times applying this skill failed"
    )

    # Categorization
    tags: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Searchable tags (deployment, kubernetes, etc.)"
    )

    error_class: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="What type of error this skill fixes"
    )

    tools: list[str] = Field(
        default_factory=list,
        max_length=15,
        description="Tools this skill applies to (kubectl, docker, etc.)"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    promoted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this skill was promoted from episodes"
    )

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total

    @field_validator("code", "procedure")
    @classmethod
    def validate_at_least_one_fix(cls, v: Optional[str], info) -> Optional[str]:
        """Ensure at least one of code or procedure is provided."""
        if v is None:
            other_field = "code" if info.field_name == "procedure" else "procedure"
            if other_field not in info.data or info.data[other_field] is None:
                raise ValueError("At least one of 'code' or 'procedure' must be provided")
        return v

    @field_validator("name", "docstring")
    @classmethod
    def sanitize_text_fields(cls, v: str) -> str:
        """Sanitize text fields to prevent injection."""
        v = v.replace("\x00", "")
        v = " ".join(v.split())
        return v.strip()

    @field_validator("source_episodes")
    @classmethod
    def validate_unique_episodes(cls, v: list[int]) -> list[int]:
        """Ensure no duplicate source episodes."""
        if len(v) != len(set(v)):
            raise ValueError("Duplicate episode IDs in source_episodes")
        return v

    @field_validator("success_count", "failure_count")
    @classmethod
    def validate_usage_consistency(cls, v: int, info) -> int:
        """Ensure success + failure doesn't exceed usage."""
        if "usage_count" in info.data:
            usage = info.data["usage_count"]
            success = info.data.get("success_count", 0)
            failure = info.data.get("failure_count", 0)

            if info.field_name == "success_count":
                if v + failure > usage:
                    raise ValueError(
                        f"success_count ({v}) + failure_count ({failure}) "
                        f"cannot exceed usage_count ({usage})"
                    )
            elif info.field_name == "failure_count":
                if success + v > usage:
                    raise ValueError(
                        f"success_count ({success}) + failure_count ({v}) "
                        f"cannot exceed usage_count ({usage})"
                    )
        return v

    def to_metadata_dict(self) -> dict[str, str]:
        """
        Convert to KyroDB metadata format (map<string, string>).

        Returns:
            dict: Metadata compatible with KyroDB InsertRequest
        """
        metadata = {
            "customer_id": self.customer_id,
            "name": self.name,
            "docstring": self.docstring,
            "code": self.code or "",
            "procedure": self.procedure or "",
            "language": self.language or "",
            "source_episodes": json.dumps(self.source_episodes),
            "usage_count": str(self.usage_count),
            "success_count": str(self.success_count),
            "failure_count": str(self.failure_count),
            "success_rate": str(self.success_rate),
            "tags": json.dumps(self.tags),
            "error_class": self.error_class,
            "tools": json.dumps(self.tools),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "promoted_at": self.promoted_at.isoformat(),
            "skill_json": json.dumps(self.model_dump(mode="json"), default=str),
        }

        return metadata

    @classmethod
    def from_metadata_dict(cls, doc_id: int, metadata: dict[str, str]) -> "Skill":
        """
        Reconstruct Skill from KyroDB metadata.

        Args:
            doc_id: KyroDB document ID
            metadata: Metadata dict from SearchResult

        Returns:
            Skill: Reconstructed skill instance
        """
        if "skill_json" in metadata:
            data = json.loads(metadata["skill_json"])
            skill = cls.model_validate(data)
            skill.skill_id = doc_id
            return skill
        else:
            return cls(
                skill_id=doc_id,
                customer_id=metadata["customer_id"],
                name=metadata["name"],
                docstring=metadata["docstring"],
                code=metadata.get("code") or None,
                procedure=metadata.get("procedure") or None,
                language=metadata.get("language") or None,
                source_episodes=json.loads(metadata["source_episodes"]),
                usage_count=int(metadata.get("usage_count", 0)),
                success_count=int(metadata.get("success_count", 0)),
                failure_count=int(metadata.get("failure_count", 0)),
                tags=json.loads(metadata.get("tags", "[]")),
                error_class=metadata["error_class"],
                tools=json.loads(metadata.get("tools", "[]")),
                created_at=datetime.fromisoformat(metadata["created_at"]),
                updated_at=datetime.fromisoformat(metadata["updated_at"]),
                promoted_at=datetime.fromisoformat(metadata["promoted_at"]),
            )
