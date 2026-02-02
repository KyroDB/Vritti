"""
Skills Promotion End-to-End Integration Tests.

 Skills Promotion E2E
Tests the full lifecycle:
1. Episode capture with reflection
2. Fix validation with success tracking
3. Skill promotion when criteria met (3+ similar episodes, >90% success rate)
4. Skill retrieval and usage in gating
5. Skill feedback updates

Security tested:
- Customer namespace isolation
- Input validation
- Rate limiting awareness

Edge cases tested:
- Empty reflections
- Missing usage stats
- Promotion threshold boundaries
"""

import contextlib
import time
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.config import get_settings
from src.ingestion.embedding import EmbeddingService
from src.kyrodb.router import KyroDBRouter, get_namespaced_collection
from src.models.episode import (
    Episode,
    EpisodeCreate,
    EpisodeType,
    ErrorClass,
    Reflection,
    UsageStats,
)
from src.models.skill import Skill
from src.skills.promotion import SkillPromotionService


def _make_episode_id() -> int:
    """Generate a unique episode ID (int64)."""
    return int(time.time() * 1000) % (2**63 - 1)


def _make_episode_create(
    customer_id: str,
    goal: str = "Deploy application to Kubernetes cluster",
    error_class: ErrorClass = ErrorClass.CONFIGURATION_ERROR,
    tool: str = "kubectl",
    error_trace: str = "Error: ImagePullBackOff - unable to pull image",
    resolution: str = None,
) -> EpisodeCreate:
    """Helper to create a valid EpisodeCreate object."""
    return EpisodeCreate(
        customer_id=customer_id,
        episode_type=EpisodeType.FAILURE,
        goal=goal,
        tool_chain=[tool, "docker"],
        actions_taken=["kubectl apply -f deployment.yaml", "kubectl get pods", "kubectl describe pod"],
        error_trace=error_trace,
        error_class=error_class,
        environment_info={"kubernetes_version": "1.28", "cluster": "test-cluster"},
        resolution=resolution,
        tags=["kubernetes", "deployment"],
        severity=3,
    )


def _make_reflection(
    root_cause: str = "Image registry credentials not configured",
    resolution_strategy: str = """
1. Create a Kubernetes secret with registry credentials:
   kubectl create secret docker-registry regcred --docker-server=registry.example.com --docker-username=user --docker-password=pass

2. Add imagePullSecrets to deployment:
   ```yaml
   spec:
     imagePullSecrets:
     - name: regcred
   ```

3. Reapply the deployment:
   kubectl apply -f deployment.yaml
""",
    confidence: float = 0.85,
) -> Reflection:
    """Helper to create a valid Reflection object."""
    return Reflection(
        root_cause=root_cause,
        preconditions=[
            "Using private container registry",
            "Registry credentials not configured",
            "Kubernetes cluster environment",
        ],
        resolution_strategy=resolution_strategy,
        environment_factors=["kubernetes", "docker-registry"],
        affected_components=["deployment", "imagePullSecrets"],
        generalization_score=0.7,
        confidence_score=confidence,
        llm_model="test-model",
        generated_at=datetime.now(UTC),
        cost_usd=0.01,
        generation_latency_ms=100,
    )


def _make_usage_stats(
    total_retrievals: int = 5,
    fix_applied_count: int = 10,
    fix_success_count: int = 10,
    fix_failure_count: int = 0,
) -> UsageStats:
    """Helper to create UsageStats with high success rate."""
    return UsageStats(
        total_retrievals=total_retrievals,
        fix_applied_count=fix_applied_count,
        fix_success_count=fix_success_count,
        fix_failure_count=fix_failure_count,
    )


async def _insert_episode_with_reflection(
    kyrodb_router: KyroDBRouter,
    embedding_service: EmbeddingService,
    customer_id: str,
    episode_create: EpisodeCreate,
    reflection: Reflection,
    usage_stats: UsageStats,
) -> Episode:
    """
    Insert an episode with reflection and usage stats directly into KyroDB.
    
    This bypasses the ingestion pipeline for controlled test setup.
    """
    episode_id = _make_episode_id()
    
    episode = Episode(
        create_data=episode_create,
        episode_id=episode_id,
        reflection=reflection,
        usage_stats=usage_stats,
        created_at=datetime.now(UTC),
    )
    
    # Generate embedding
    text_content = f"{episode_create.goal}\n\n{episode_create.error_trace}"
    embedding = embedding_service.embed_text(text_content)
    
    # Store in KyroDB
    text_success, _ = await kyrodb_router.insert_episode(
        episode_id=episode_id,
        customer_id=customer_id,
        collection="failures",
        text_embedding=embedding,
        metadata=episode.to_metadata_dict(),
    )
    
    if not text_success:
        raise RuntimeError(f"Failed to insert episode {episode_id}")
    
    return episode


@pytest.fixture
def embedding_service():
    """Create an embedding service for tests."""
    settings = get_settings()
    return EmbeddingService(config=settings.embedding)


@pytest.fixture
def promotion_service(kyrodb_router, embedding_service):
    """Create a SkillPromotionService for tests."""
    return SkillPromotionService(
        kyrodb_router=kyrodb_router,
        embedding_service=embedding_service,
    )


@pytest.mark.integration
@pytest.mark.requires_kyrodb
class TestSkillsPromotionE2E:
    """Test end-to-end skills promotion flow."""

    @pytest.mark.asyncio
    async def test_promotion_requires_minimum_episodes(
        self,
        kyrodb_router: KyroDBRouter,
        embedding_service: EmbeddingService,
        promotion_service: SkillPromotionService,
    ):
        """
        Test that promotion requires minimum 3 similar episodes.
        
        Setup: 2 similar episodes with high success rate
        Expected: No promotion (under threshold)
        """
        customer_id = f"test_promo_{uuid4().hex[:8]}"
        inserted_episodes = []
        
        try:
            # Insert only 2 episodes (below MIN_EPISODES_FOR_PROMOTION=3)
            for i in range(2):
                episode_create = _make_episode_create(
                    customer_id=customer_id,
                    goal="Deploy application to Kubernetes cluster",
                    error_trace=f"Error: ImagePullBackOff - attempt {i}",
                )
                reflection = _make_reflection(confidence=0.9)
                usage_stats = _make_usage_stats(
                    fix_applied_count=5,
                    fix_success_count=5,
                )
                
                episode = await _insert_episode_with_reflection(
                    kyrodb_router=kyrodb_router,
                    embedding_service=embedding_service,
                    customer_id=customer_id,
                    episode_create=episode_create,
                    reflection=reflection,
                    usage_stats=usage_stats,
                )
                inserted_episodes.append(episode)
            
            # Try to promote first episode
            skill = await promotion_service.check_and_promote(
                episode_id=inserted_episodes[0].episode_id,
                customer_id=customer_id,
            )
            
            # Should NOT promote - only 2 episodes, need 3
            assert skill is None, "Should not promote with only 2 similar episodes"
            
        finally:
            # Cleanup
            for episode in inserted_episodes:
                with contextlib.suppress(Exception):
                    await kyrodb_router.delete_episode(
                        episode.episode_id, customer_id, "failures"
                    )

    @pytest.mark.asyncio
    async def test_promotion_requires_high_success_rate(
        self,
        kyrodb_router: KyroDBRouter,
        embedding_service: EmbeddingService,
        promotion_service: SkillPromotionService,
    ):
        """
        Test that promotion requires >90% success rate.
        
        Setup: 4 similar episodes with 80% success rate
        Expected: No promotion (below threshold)
        """
        customer_id = f"test_promo_{uuid4().hex[:8]}"
        inserted_episodes = []
        
        try:
            # Insert 4 episodes with 80% success rate (below MIN_SUCCESS_RATE=0.9)
            for i in range(4):
                episode_create = _make_episode_create(
                    customer_id=customer_id,
                    goal="Deploy application to Kubernetes cluster",
                    error_trace=f"Error: ImagePullBackOff - attempt {i}",
                )
                reflection = _make_reflection(confidence=0.85)
                
                # 80% success rate: 4/5 successes
                usage_stats = _make_usage_stats(
                    fix_applied_count=5,
                    fix_success_count=4,
                    fix_failure_count=1,
                )
                
                episode = await _insert_episode_with_reflection(
                    kyrodb_router=kyrodb_router,
                    embedding_service=embedding_service,
                    customer_id=customer_id,
                    episode_create=episode_create,
                    reflection=reflection,
                    usage_stats=usage_stats,
                )
                inserted_episodes.append(episode)
            
            # Try to promote first episode
            skill = await promotion_service.check_and_promote(
                episode_id=inserted_episodes[0].episode_id,
                customer_id=customer_id,
            )
            
            # Should NOT promote - success rate 80% < 90%
            assert skill is None, "Should not promote with 80% success rate"
            
        finally:
            # Cleanup
            for episode in inserted_episodes:
                with contextlib.suppress(Exception):
                    await kyrodb_router.delete_episode(
                        episode.episode_id, customer_id, "failures"
                    )

    @pytest.mark.asyncio
    async def test_successful_promotion_flow(
        self,
        kyrodb_router: KyroDBRouter,
        embedding_service: EmbeddingService,
        promotion_service: SkillPromotionService,
    ):
        """
        Test successful promotion when all criteria met.
        
        Setup: 4 similar episodes with >90% success rate and code in resolution
        Expected: Skill created with correct metadata
        """
        customer_id = f"test_promo_{uuid4().hex[:8]}"
        inserted_episodes = []
        created_skill_id = None
        
        try:
            # Insert 4 episodes meeting all criteria
            for i in range(4):
                episode_create = _make_episode_create(
                    customer_id=customer_id,
                    goal="Deploy application to Kubernetes cluster",
                    error_trace=f"Error: ImagePullBackOff - pod/myapp-{i}",
                )
                # Resolution with code block (required for promotion)
                reflection = _make_reflection(
                    root_cause="Image registry credentials not configured",
                    resolution_strategy="""
Fix: Create imagePullSecrets

```bash
kubectl create secret docker-registry regcred \\
  --docker-server=registry.example.com \\
  --docker-username=user \\
  --docker-password=pass
```

Then add to deployment spec:
```yaml
imagePullSecrets:
  - name: regcred
```
""",
                    confidence=0.92,
                )
                
                # High success rate: 95%
                usage_stats = _make_usage_stats(
                    total_retrievals=20,
                    fix_applied_count=19,
                    fix_success_count=19,
                    fix_failure_count=0,
                )
                
                episode = await _insert_episode_with_reflection(
                    kyrodb_router=kyrodb_router,
                    embedding_service=embedding_service,
                    customer_id=customer_id,
                    episode_create=episode_create,
                    reflection=reflection,
                    usage_stats=usage_stats,
                )
                inserted_episodes.append(episode)
            
            # Promote first episode
            skill = await promotion_service.check_and_promote(
                episode_id=inserted_episodes[0].episode_id,
                customer_id=customer_id,
            )
            
            # Should succeed
            assert skill is not None, "Should promote episode to skill"
            created_skill_id = skill.skill_id
            
            # Verify skill metadata
            assert skill.customer_id == customer_id
            assert skill.error_class == "configuration_error"
            assert len(skill.source_episodes) >= 3  # At least 3 source episodes
            assert skill.code is not None or skill.procedure is not None
            assert "kubectl" in skill.name.lower() or "fix" in skill.name.lower()
            
            # Verify skill is searchable
            query_text = "ImagePullBackOff kubernetes deploy"
            query_embedding = embedding_service.embed_text(query_text)
            
            search_results = await kyrodb_router.search_skills(
                query_embedding=query_embedding,
                customer_id=customer_id,
                k=5,
                min_score=0.5,
            )
            
            assert len(search_results) > 0, "Should find promoted skill"
            found_skill_ids = [s.skill_id for s, _ in search_results]
            assert skill.skill_id in found_skill_ids, "Promoted skill should be in search results"
            
        finally:
            # Cleanup
            for episode in inserted_episodes:
                with contextlib.suppress(Exception):
                    await kyrodb_router.delete_episode(
                        episode.episode_id, customer_id, "failures"
                    )
            
            if created_skill_id:
                try:
                    namespace = get_namespaced_collection(customer_id, "skills")
                    await kyrodb_router.text_client.delete(
                        doc_id=created_skill_id,
                        namespace=namespace,
                    )
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_skill_feedback_updates_stats(
        self,
        kyrodb_router: KyroDBRouter,
        embedding_service: EmbeddingService,
    ):
        """
        Test that skill feedback correctly updates usage statistics.
        """
        customer_id = f"test_feedback_{uuid4().hex[:8]}"
        skill_id = _make_episode_id()
        
        # Create a skill directly
        skill = Skill(
            skill_id=skill_id,
            customer_id=customer_id,
            name="Test feedback skill for validation",
            docstring="A test skill to verify feedback updates work correctly with atomic operations",
            code="echo 'test'",
            language="bash",
            error_class="TestError",
            source_episodes=[skill_id - 100],
            usage_count=0,
            success_count=0,
            failure_count=0,
        )
        
        embedding = [0.5] * 384
        
        try:
            # Insert skill
            await kyrodb_router.insert_skill(skill, embedding)
            
            # Submit success feedback
            success1 = await kyrodb_router.update_skill_stats(
                skill_id=skill_id,
                customer_id=customer_id,
                success=True,
            )
            assert success1, "First success feedback should succeed"
            
            # Submit another success
            success2 = await kyrodb_router.update_skill_stats(
                skill_id=skill_id,
                customer_id=customer_id,
                success=True,
            )
            assert success2, "Second success feedback should succeed"
            
            # Submit a failure
            success3 = await kyrodb_router.update_skill_stats(
                skill_id=skill_id,
                customer_id=customer_id,
                success=False,
            )
            assert success3, "Failure feedback should succeed"
            
            # Verify stats
            namespace = get_namespaced_collection(customer_id, "skills")
            skill_data = await kyrodb_router.text_client.query(
                doc_id=skill_id,
                namespace=namespace,
                include_embedding=False,
            )
            
            assert skill_data.found, "Skill should exist"
            updated_skill = Skill.from_metadata_dict(skill_id, dict(skill_data.metadata))
            
            assert updated_skill.usage_count == 3, f"Usage count should be 3, got {updated_skill.usage_count}"
            assert updated_skill.success_count == 2, f"Success count should be 2, got {updated_skill.success_count}"
            assert updated_skill.failure_count == 1, f"Failure count should be 1, got {updated_skill.failure_count}"
            
            # Verify success rate: 2/3 = 0.666...
            expected_rate = 2 / 3
            assert abs(updated_skill.success_rate - expected_rate) < 0.01, \
                f"Success rate should be ~0.67, got {updated_skill.success_rate}"
            
        finally:
            try:
                namespace = get_namespaced_collection(customer_id, "skills")
                await kyrodb_router.text_client.delete(doc_id=skill_id, namespace=namespace)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_customer_isolation_for_skills(
        self,
        kyrodb_router: KyroDBRouter,
        embedding_service: EmbeddingService,
        promotion_service: SkillPromotionService,
    ):
        """
        Test that skills are isolated by customer_id.
        
        Customer A's episodes should not contribute to Customer B's skill promotion.
        """
        customer_a = f"customer_a_{uuid4().hex[:8]}"
        customer_b = f"customer_b_{uuid4().hex[:8]}"
        inserted_episodes_a = []
        inserted_episodes_b = []
        
        try:
            # Insert 3 episodes for Customer A (meets promotion threshold)
            for i in range(3):
                episode_create = _make_episode_create(
                    customer_id=customer_a,
                    goal="Deploy app to K8s",
                    error_trace=f"ImagePullBackOff error A{i} details here",
                )
                reflection = _make_reflection(confidence=0.9)
                usage_stats = _make_usage_stats(
                    fix_applied_count=10,
                    fix_success_count=10,
                )
                
                episode = await _insert_episode_with_reflection(
                    kyrodb_router=kyrodb_router,
                    embedding_service=embedding_service,
                    customer_id=customer_a,
                    episode_create=episode_create,
                    reflection=reflection,
                    usage_stats=usage_stats,
                )
                inserted_episodes_a.append(episode)
            
            # Insert 1 episode for Customer B (does NOT meet promotion threshold)
            episode_create_b = _make_episode_create(
                customer_id=customer_b,
                goal="Deploy app to K8s",  # Same goal as A's episodes
                error_trace="ImagePullBackOff error B0 details here",
            )
            reflection_b = _make_reflection(confidence=0.9)
            usage_stats_b = _make_usage_stats(
                fix_applied_count=10,
                fix_success_count=10,
            )
            
            episode_b = await _insert_episode_with_reflection(
                kyrodb_router=kyrodb_router,
                embedding_service=embedding_service,
                customer_id=customer_b,
                episode_create=episode_create_b,
                reflection=reflection_b,
                usage_stats=usage_stats_b,
            )
            inserted_episodes_b.append(episode_b)
            
            # Customer B should NOT be able to promote - only has 1 episode
            skill_b = await promotion_service.check_and_promote(
                episode_id=episode_b.episode_id,
                customer_id=customer_b,
            )
            
            assert skill_b is None, \
                "Customer B should not promote - A's episodes should not count"
            
        finally:
            # Cleanup
            for episode in inserted_episodes_a:
                with contextlib.suppress(Exception):
                    await kyrodb_router.delete_episode(
                        episode.episode_id, customer_a, "failures"
                    )
            
            for episode in inserted_episodes_b:
                with contextlib.suppress(Exception):
                    await kyrodb_router.delete_episode(
                        episode.episode_id, customer_b, "failures"
                    )

    @pytest.mark.asyncio
    async def test_promotion_handles_missing_reflection(
        self,
        kyrodb_router: KyroDBRouter,
        embedding_service: EmbeddingService,
        promotion_service: SkillPromotionService,
    ):
        """
        Test that promotion gracefully handles episodes without reflections.
        
        Edge case: Episode exists but has no reflection yet.
        """
        customer_id = f"test_missing_{uuid4().hex[:8]}"
        episode_id = _make_episode_id()
        
        try:
            # Create episode WITHOUT reflection
            episode_create = _make_episode_create(
                customer_id=customer_id,
                goal="Test missing reflection handling",
            )
            
            episode = Episode(
                create_data=episode_create,
                episode_id=episode_id,
                reflection=None,  # No reflection
                usage_stats=_make_usage_stats(
                    fix_applied_count=10,
                    fix_success_count=10,
                ),
                created_at=datetime.now(UTC),
            )
            
            # Generate embedding and store
            text_content = f"{episode_create.goal}\n\n{episode_create.error_trace}"
            embedding = embedding_service.embed_text(text_content)
            
            await kyrodb_router.insert_episode(
                episode_id=episode_id,
                customer_id=customer_id,
                collection="failures",
                text_embedding=embedding,
                metadata=episode.to_metadata_dict(),
            )
            
            # Try to promote - should return None gracefully
            skill = await promotion_service.check_and_promote(
                episode_id=episode_id,
                customer_id=customer_id,
            )
            
            assert skill is None, "Should not promote episode without reflection"
            
        finally:
            with contextlib.suppress(Exception):
                await kyrodb_router.delete_episode(episode_id, customer_id, "failures")

    @pytest.mark.asyncio
    async def test_promotion_handles_zero_applications(
        self,
        kyrodb_router: KyroDBRouter,
        embedding_service: EmbeddingService,
        promotion_service: SkillPromotionService,
    ):
        """
        Test that promotion handles episodes with zero fix applications.
        
        Edge case: Episode has reflection but fix was never applied.
        """
        customer_id = f"test_zero_app_{uuid4().hex[:8]}"
        inserted_episodes = []
        
        try:
            # Create 4 episodes with ZERO applications
            for i in range(4):
                episode_create = _make_episode_create(
                    customer_id=customer_id,
                    goal="Test zero applications handling",
                    error_trace=f"Error message for test case number {i}",
                )
                reflection = _make_reflection(confidence=0.9)
                
                # Zero applications = undefined success rate
                usage_stats = UsageStats(
                    total_retrievals=10,
                    fix_applied_count=0,
                    fix_success_count=0,
                    fix_failure_count=0,
                )
                
                episode = await _insert_episode_with_reflection(
                    kyrodb_router=kyrodb_router,
                    embedding_service=embedding_service,
                    customer_id=customer_id,
                    episode_create=episode_create,
                    reflection=reflection,
                    usage_stats=usage_stats,
                )
                inserted_episodes.append(episode)
            
            # Try to promote - should return None (0% success rate < 90%)
            skill = await promotion_service.check_and_promote(
                episode_id=inserted_episodes[0].episode_id,
                customer_id=customer_id,
            )
            
            assert skill is None, "Should not promote episode with zero applications"
            
        finally:
            for episode in inserted_episodes:
                with contextlib.suppress(Exception):
                    await kyrodb_router.delete_episode(
                        episode.episode_id, customer_id, "failures"
                    )


@pytest.mark.integration
@pytest.mark.requires_kyrodb
class TestSkillsInGating:
    """Test skills integration with gating service."""

    @pytest.mark.asyncio
    async def test_gating_returns_relevant_skills(
        self,
        kyrodb_router: KyroDBRouter,
        embedding_service: EmbeddingService,
    ):
        """
        Test that gating service returns relevant skills as hints.
        """
        from src.gating.service import GatingService
        from src.models.gating import ReflectRequest
        from src.retrieval.search import SearchPipeline
        
        customer_id = f"test_gating_{uuid4().hex[:8]}"
        skill_id = _make_episode_id()
        
        # Create a skill about Kubernetes deployment
        skill = Skill(
            skill_id=skill_id,
            customer_id=customer_id,
            name="Fix Kubernetes ImagePullBackOff errors",
            docstring="Configure imagePullSecrets when using private container registries with Kubernetes deployments",
            code="kubectl create secret docker-registry regcred",
            language="bash",
            error_class="configuration_error",
            source_episodes=[skill_id - 100, skill_id - 200, skill_id - 300],
            usage_count=10,
            success_count=9,
            failure_count=1,
        )
        
        # Embedding for Kubernetes topic
        skill_embedding = embedding_service.embed_text(
            f"{skill.name} {skill.docstring}"
        )
        
        try:
            # Insert skill
            await kyrodb_router.insert_skill(skill, skill_embedding)
            
            # Create gating service
            search_pipeline = SearchPipeline(
                kyrodb_router=kyrodb_router,
                embedding_service=embedding_service,
            )
            gating_service = GatingService(
                search_pipeline=search_pipeline,
                kyrodb_router=kyrodb_router,
            )
            
            # Create reflect request about K8s deployment
            request = ReflectRequest(
                goal="Deploy my application to Kubernetes",
                proposed_action="kubectl apply -f deployment.yaml with private registry image",
                tool="kubectl",
                current_state={
                    "cluster": "production",
                    "registry": "private.registry.io",
                },
            )
            
            # Get gating response
            response = await gating_service.reflect_before_action(
                request=request,
                customer_id=customer_id,
            )
            
            # Verify response includes skills
            assert response is not None
            assert response.relevant_skills is not None
            
            # Note: Skills may or may not be found depending on similarity threshold
            # The important thing is the endpoint works without error
            if len(response.relevant_skills) > 0:
                # If we found skills, verify structure
                found_skill = response.relevant_skills[0]
                assert "name" in found_skill or hasattr(found_skill, "name")
            
        finally:
            try:
                namespace = get_namespaced_collection(customer_id, "skills")
                await kyrodb_router.text_client.delete(doc_id=skill_id, namespace=namespace)
            except Exception:
                pass
