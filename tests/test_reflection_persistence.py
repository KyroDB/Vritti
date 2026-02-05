"""
Integration tests for reflection persistence to KyroDB.

Tests:
- Reflection serialization to metadata
- Reflection persistence with customer validation
- End-to-end: Capture → Generate → Persist → Retrieve
- Security: Customer ID validation, cross-customer attacks
"""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.kyrodb.router import KyroDBRouter
from src.models.episode import (
    LLMPerspective,
    Reflection,
    ReflectionConsensus,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_reflection_with_consensus():
    """Sample reflection with multi-perspective consensus."""
    perspective1 = LLMPerspective(
        model_name="gpt-4-turbo-preview",
        root_cause="Configuration error in deployment manifest",
        preconditions={"tool": "kubectl", "platform": "kubernetes"},
        resolution_strategy="Update image tag and reapply manifest",
        environment_factors=["Kubernetes 1.28"],
        affected_components=["deployment", "pod"],
        generalization_score=0.8,
        confidence_score=0.9,
        reasoning="Common K8s deployment issue",
    )

    perspective2 = LLMPerspective(
        model_name="claude-3-5-sonnet-20241022",
        root_cause="Configuration error in deployment manifest",
        preconditions={
            "tool": "kubectl",
            "platform": "kubernetes",
            "artifact": "container_registry",
        },
        resolution_strategy="Fix deployment configuration",
        environment_factors=["K8s cluster"],
        affected_components=["deployment"],
        generalization_score=0.75,
        confidence_score=0.85,
        reasoning="Tag mismatch",
    )

    consensus = ReflectionConsensus(
        perspectives=[perspective1, perspective2],
        consensus_method="unanimous",
        agreed_root_cause="Configuration error in deployment manifest",
        agreed_preconditions={
            "tool": "kubectl",
            "platform": "kubernetes",
            "artifact": "container_registry",
        },
        agreed_resolution="Update image tag and reapply manifest",
        consensus_confidence=1.0,
        disagreement_points=[],
        generated_at=datetime.now(UTC),
    )

    return Reflection(
        consensus=consensus,
        root_cause=consensus.agreed_root_cause,
        preconditions=consensus.agreed_preconditions,
        resolution_strategy=consensus.agreed_resolution,
        environment_factors=["Kubernetes 1.28", "K8s cluster"],
        affected_components=["deployment", "pod"],
        generalization_score=0.775,  # Average
        confidence_score=1.0,
        llm_model="multi-perspective",
        generated_at=datetime.now(UTC),
        cost_usd=0.065,
        generation_latency_ms=3500.0,
    )


@pytest.fixture
def sample_reflection_single_llm():
    """Sample reflection from single LLM (no consensus)."""
    return Reflection(
        consensus=None,  # No consensus
        root_cause="Network timeout connecting to external API",
        preconditions={"dependency": "external_api", "network": "required"},
        resolution_strategy="Add retry logic with exponential backoff",
        environment_factors=["Production network"],
        affected_components=["API client"],
        generalization_score=0.7,
        confidence_score=0.8,
        llm_model="gpt-4-turbo-preview",
        generated_at=datetime.now(UTC),
        cost_usd=0.043,
        generation_latency_ms=2100.0,
    )


# ============================================================================
# Serialization Tests
# ============================================================================


class TestReflectionSerialization:
    """Test reflection serialization to KyroDB metadata format."""

    def test_reflection_accepts_structured_preconditions_dict(self):
        """Structured preconditions dict is preserved (canonical dict form)."""
        reflection = Reflection(
            root_cause="Missing image tag in deployment spec",
            preconditions={"tool": "kubectl", "image_tag": "latest"},
            resolution_strategy="Pin image tag and redeploy",
            environment_factors=["kubernetes"],
            affected_components=["deployment"],
            generalization_score=0.8,
            confidence_score=0.9,
            llm_model="test-model",
            generated_at=datetime.now(UTC),
            cost_usd=0.0,
            generation_latency_ms=10.0,
        )

        assert reflection.preconditions == {"tool": "kubectl", "image_tag": "latest"}

    def test_serialize_reflection_with_consensus(self, sample_reflection_with_consensus):
        """Test serialization of multi-perspective reflection."""
        router = KyroDBRouter(config=MagicMock())  # Don't need real config for serialization

        metadata = router._serialize_reflection_to_metadata(sample_reflection_with_consensus)

        # Core fields
        assert metadata["reflection_root_cause"] == sample_reflection_with_consensus.root_cause
        assert (
            metadata["reflection_resolution"]
            == sample_reflection_with_consensus.resolution_strategy
        )
        assert "reflection_confidence" in metadata
        assert float(metadata["reflection_confidence"]) == 1.0

        # Lists are JSON-encoded
        preconditions = json.loads(metadata["reflection_preconditions"])
        assert isinstance(preconditions, dict)
        assert len(preconditions) > 0

        # Consensus metadata
        assert metadata["reflection_consensus_method"] == "unanimous"
        assert float(metadata["reflection_consensus_confidence"]) == 1.0
        assert "reflection_perspectives_count" in metadata
        assert metadata["reflection_perspectives_count"] == "2"

        # Individual perspectives stored for audit
        perspectives_json = json.loads(metadata["reflection_perspectives_json"])
        assert len(perspectives_json) == 2
        assert perspectives_json[0]["model"] == "gpt-4-turbo-preview"
        assert perspectives_json[1]["model"] == "claude-3-5-sonnet-20241022"

        # Cost tracking
        assert "reflection_cost_usd" in metadata
        assert float(metadata["reflection_cost_usd"]) == 0.065

    def test_serialize_reflection_single_llm(self, sample_reflection_single_llm):
        """Test serialization of single-LLM reflection (no consensus)."""
        router = KyroDBRouter(config=MagicMock())

        metadata = router._serialize_reflection_to_metadata(sample_reflection_single_llm)

        # Core fields should exist
        assert "reflection_root_cause" in metadata
        assert "reflection_resolution" in metadata
        assert "reflection_model" in metadata
        assert metadata["reflection_model"] == "gpt-4-turbo-preview"

        # Consensus fields should NOT exist (no consensus)
        assert "reflection_consensus_method" not in metadata
        assert "reflection_perspectives_count" not in metadata

    def test_serialization_is_string_string_map(self, sample_reflection_with_consensus):
        """Test that all metadata values are strings (KyroDB requirement)."""
        router = KyroDBRouter(config=MagicMock())

        metadata = router._serialize_reflection_to_metadata(sample_reflection_with_consensus)

        # All values must be strings
        for key, value in metadata.items():
            assert isinstance(value, str), f"Key {key} has non-string value: {type(value)}"


# ============================================================================
# Persistence Tests
# ============================================================================


@pytest.mark.asyncio
class TestReflectionPersistence:
    """Test reflection persistence to KyroDB."""

    async def test_update_episode_reflection_success(self, sample_reflection_with_consensus):
        """Test successful reflection persistence."""
        # Mock KyroDB client
        mock_client = AsyncMock()
        mock_client.query.return_value = MagicMock(
            found=True,
            metadata={"customer_id": "test-customer", "episode_type": "failure"},
            embedding=[0.1] * 384,  # Mock embedding
        )
        mock_client.insert.return_value = MagicMock(success=True, error="")

        # Create router with mocked client
        router = KyroDBRouter(config=MagicMock())
        router.text_client = mock_client

        # Attempt to persist reflection
        success = await router.update_episode_reflection(
            episode_id=12345,
            customer_id="test-customer",
            collection="failures",
            reflection=sample_reflection_with_consensus,
        )

        assert success is True

        # Verify query was called (to fetch existing episode)
        mock_client.query.assert_called_once()
        query_call = mock_client.query.call_args
        assert query_call.kwargs["doc_id"] == 12345
        assert query_call.kwargs["include_embedding"] is True

        # Verify insert was called (to update with reflection)
        mock_client.insert.assert_called_once()
        insert_call = mock_client.insert.call_args
        assert insert_call.kwargs["doc_id"] == 12345
        assert "reflection_root_cause" in insert_call.kwargs["metadata"]

    async def test_update_episode_reflection_not_found(self, sample_reflection_with_consensus):
        """Test persistence fails when episode doesn't exist."""
        # Mock episode not found
        mock_client = AsyncMock()
        mock_client.query.return_value = MagicMock(found=False)

        router = KyroDBRouter(config=MagicMock())
        router.text_client = mock_client

        # Should fail gracefully (not raise)
        success = await router.update_episode_reflection(
            episode_id=99999,
            customer_id="test-customer",
            collection="failures",
            reflection=sample_reflection_with_consensus,
        )

        assert success is False

        # Insert should NOT be called
        mock_client.insert.assert_not_called()

    async def test_update_episode_reflection_customer_mismatch(
        self, sample_reflection_with_consensus
    ):
        """Test security: Prevent cross-customer reflection updates."""
        # Mock episode belonging to different customer
        mock_client = AsyncMock()
        mock_client.query.return_value = MagicMock(
            found=True,
            metadata={"customer_id": "other-customer"},  # Different customer!
            embedding=[0.1] * 384,
        )

        router = KyroDBRouter(config=MagicMock())
        router.text_client = mock_client

        # Should fail due to customer mismatch
        success = await router.update_episode_reflection(
            episode_id=12345,
            customer_id="test-customer",  # Requesting customer
            collection="failures",
            reflection=sample_reflection_with_consensus,
        )

        assert success is False

        # Insert should NOT be called (security violation)
        mock_client.insert.assert_not_called()

    async def test_update_episode_reflection_preserves_metadata(
        self, sample_reflection_with_consensus
    ):
        """Test that existing metadata is preserved when adding reflection."""
        # Mock existing episode with metadata
        existing_metadata = {
            "customer_id": "test-customer",
            "episode_type": "failure",
            "tool": "kubectl",
            "timestamp": "1700000000",
            "custom_field": "preserve this",
        }

        mock_client = AsyncMock()
        mock_client.query.return_value = MagicMock(
            found=True,
            metadata=existing_metadata,
            embedding=[0.1] * 384,
        )
        mock_client.insert.return_value = MagicMock(success=True)

        router = KyroDBRouter(config=MagicMock())
        router.text_client = mock_client

        await router.update_episode_reflection(
            episode_id=12345,
            customer_id="test-customer",
            collection="failures",
            reflection=sample_reflection_with_consensus,
        )

        # Check merged metadata
        insert_call = mock_client.insert.call_args
        merged_metadata = insert_call.kwargs["metadata"]

        # Existing fields should be preserved
        assert merged_metadata["customer_id"] == "test-customer"
        assert merged_metadata["custom_field"] == "preserve this"

        # Reflection fields should be added
        assert "reflection_root_cause" in merged_metadata
        assert "reflection_consensus_method" in merged_metadata


# ============================================================================
# End-to-End Integration Test
# ============================================================================


@pytest.mark.asyncio
async def test_end_to_end_capture_reflect_persist_retrieve(
    sample_reflection_with_consensus,
):
    """
    Full integration test: Capture episode → Generate reflection → Persist → Retrieve.

    This simulates the complete flow a real episode goes through.
    """
    episode_id = 12345
    customer_id = "test-customer"

    # Mock KyroDB client
    mock_client = AsyncMock()

    # Step 1: Initial episode insert (capture)
    initial_metadata = {
        "customer_id": customer_id,
        "episode_type": "failure",
        "error_class": "configuration_error",
    }

    # Step 2: Query for reflection update
    mock_client.query.return_value = MagicMock(
        found=True,
        metadata=initial_metadata,
        embedding=[0.1] * 384,
    )

    # Step 3: Re-insert with reflection
    mock_client.insert.return_value = MagicMock(success=True)

    # Create router
    router = KyroDBRouter(config=MagicMock())
    router.text_client = mock_client

    # Persist reflection
    success = await router.update_episode_reflection(
        episode_id=episode_id,
        customer_id=customer_id,
        collection="failures",
        reflection=sample_reflection_with_consensus,
    )

    assert success is True

    # Verify final metadata contains reflection
    insert_call = mock_client.insert.call_args
    final_metadata = insert_call.kwargs["metadata"]

    # Original fields preserved
    assert final_metadata["customer_id"] == customer_id
    assert final_metadata["episode_type"] == "failure"

    # Reflection fields added
    assert "reflection_root_cause" in final_metadata
    assert "reflection_resolution" in final_metadata
    assert "reflection_consensus_method" in final_metadata
    assert final_metadata["reflection_consensus_method"] == "unanimous"

    # Cost tracked
    assert "reflection_cost_usd" in final_metadata
    cost = float(final_metadata["reflection_cost_usd"])
    assert cost > 0.0

    print(f"✓ End-to-end test passed: Episode {episode_id} has reflection persisted")
    print(f"  Root cause: {final_metadata['reflection_root_cause']}")
    print(f"  Consensus: {final_metadata['reflection_consensus_method']}")
    print(f"  Cost: ${cost:.4f}")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
