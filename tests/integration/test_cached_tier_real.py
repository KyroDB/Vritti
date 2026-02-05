"""
End-to-end cached tier + clustering job path with real KyroDB.

Validates that:
- Episodes are enumerable via the local episode index (SQLite)
- EpisodeClusterer can fetch embeddings from KyroDB and produce clusters
- TemplateGenerator persists cluster templates to KyroDB
- TieredReflectionService can select and serve the cached tier from those templates

This test is offline (no OpenRouter calls) and requires KyroDB locally:
  - Text: localhost:50051 (384-dim)
  - Image: localhost:50052 (512-dim)
"""

from __future__ import annotations

import socket
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import KyroDBConfig, LLMConfig
from src.hygiene.clustering import EpisodeClusterer
from src.hygiene.templates import TemplateGenerator
from src.ingestion.tiered_reflection import ReflectionTier, TieredReflectionService
from src.kyrodb.router import KyroDBRouter, get_namespaced_collection
from src.models.customer import CustomerCreate, SubscriptionTier
from src.models.episode import Episode, EpisodeCreate, ErrorClass, Reflection
from src.storage.database import get_customer_db


def _is_port_open(host: str, port: int) -> bool:
    candidates = ["127.0.0.1", "::1", host] if host == "localhost" else [host]

    for target in dict.fromkeys(candidates):
        try:
            for info in socket.getaddrinfo(target, port, type=socket.SOCK_STREAM):
                family, socktype, proto, _, sockaddr = info
                with socket.socket(family, socktype, proto) as sock:
                    sock.settimeout(1)
                    if sock.connect_ex(sockaddr) == 0:
                        return True
        except Exception:
            continue
    return False


pytestmark = [pytest.mark.integration, pytest.mark.requires_kyrodb]


@pytest.fixture
def skip_if_no_kyrodb():
    if not _is_port_open("localhost", 50051):
        pytest.skip("KyroDB text instance not running on localhost:50051")
    if not _is_port_open("localhost", 50052):
        pytest.skip("KyroDB image instance not running on localhost:50052")


@pytest.mark.asyncio
async def test_cached_tier_clustering_end_to_end(skip_if_no_kyrodb, tmp_path, monkeypatch):
    # Isolate customer DB for this test process.
    db_path = Path(tmp_path) / "customers.db"
    monkeypatch.setenv("STORAGE_CUSTOMER_DB_PATH", str(db_path))

    # Enable clustering (cached tier).
    monkeypatch.setenv("CLUSTERING_ENABLED", "true")
    monkeypatch.setenv("CLUSTERING_MIN_CLUSTER_SIZE", "3")
    monkeypatch.setenv("CLUSTERING_MIN_SAMPLES", "1")
    monkeypatch.setenv("CLUSTERING_TEMPLATE_MATCH_MIN_SIMILARITY", "0.70")
    monkeypatch.setenv("CLUSTERING_TEMPLATE_MATCH_K", "5")

    # Reset singleton caches so env changes take effect.
    import src.config as config_module

    config_module._settings = None

    import src.storage.database as customer_db_module

    customer_db_module._db = None

    customer_id = f"cached-tier-e2e-{int(time.time())}"

    db = await get_customer_db()
    await db.create_customer(
        CustomerCreate(
            customer_id=customer_id,
            organization_name="Cached Tier E2E",
            email="cached-tier-e2e@vritti.local",
            subscription_tier=SubscriptionTier.PRO,
        )
    )

    router = KyroDBRouter(
        config=KyroDBConfig(
            text_host="localhost",
            text_port=50051,
            image_host="localhost",
            image_port=50052,
            enable_tls=False,
            request_timeout_seconds=30,
        )
    )
    await router.connect()

    episode_ids: list[int] = []
    template_ids: list[int] = []
    try:
        # Seed 2 separable embedding clusters so HDBSCAN reliably forms at least one cluster.
        # HDBSCAN tends to label "single dense blob" datasets as all-noise unless there is
        # density separation; two blobs avoids flakes across environments.
        rng = __import__("random").Random(0)

        def _make_embedding(*, center_dim: int, noise: float = 0.01) -> list[float]:
            vec = [0.0] * 384
            vec[center_dim] = 1.0
            # Add small noise in a stable subset of dims to avoid identical points.
            for j in range(16):
                vec[j] += rng.uniform(-noise, noise)
            norm = sum(v * v for v in vec) ** 0.5
            return [v / (norm + 1e-8) for v in vec]

        cluster_a = [_make_embedding(center_dim=0) for _ in range(10)]
        cluster_b = [_make_embedding(center_dim=1) for _ in range(10)]

        # Seed episodes (without reflection), index them, then persist reflections.
        for i, embedding in enumerate(cluster_a + cluster_b):
            episode_id = await db.allocate_doc_id()
            episode_ids.append(episode_id)

            create = EpisodeCreate(
                customer_id=customer_id,
                goal=f"Deploy application to production (attempt {i})",
                tool_chain=["kubectl"],
                actions_taken=["kubectl apply -f deployment.yaml"],
                error_trace="ImagePullBackOff: image not found in registry",
                error_class=ErrorClass.RESOURCE_ERROR,
                environment_info={"env": "production"},
                tags=["kubernetes", "deployment"],
            )

            episode = Episode(episode_id=episode_id, create_data=create)
            metadata = episode.to_metadata_dict()

            ok, _ = await router.insert_episode(
                episode_id=episode_id,
                customer_id=customer_id,
                collection="failures",
                text_embedding=embedding,
                metadata=metadata,
            )
            assert ok is True

            await db.index_episode(
                episode_id=episode_id,
                customer_id=customer_id,
                collection="failures",
                created_at=datetime.now(UTC),
                has_image=False,
            )

            reflection = Reflection(
                root_cause="Image tag does not exist in registry",
                preconditions=["tool=kubectl", "env=production"],
                resolution_strategy="Use an existing image tag or push the image before deploy",
                confidence_score=0.92,
                generalization_score=0.70,
                environment_factors=["cluster=production"],
                affected_components=["kubernetes", "registry"],
                llm_model="unit-test-reflection",
                generated_at=datetime.now(UTC),
                cost_usd=0.0,
                generation_latency_ms=1.0,
                tier=ReflectionTier.CHEAP.value,
            )

            persisted = await router.update_episode_reflection(
                episode_id=episode_id,
                customer_id=customer_id,
                collection="failures",
                reflection=reflection,
            )
            assert persisted is True

        # Cluster episodes and generate templates (offline).
        clusterer = EpisodeClusterer(
            kyrodb_router=router,
            episode_index=db,
            min_cluster_size=3,
            min_samples=1,
            metric="cosine",
        )
        clusters = await clusterer.cluster_customer_episodes(
            customer_id=customer_id,
            collection="failures",
        )
        assert clusters, "Expected at least one cluster for the seeded episodes"

        template_generator = TemplateGenerator(kyrodb_router=router, reflection_service=None)
        for info in clusters.values():
            template = await template_generator.generate_cluster_template(
                customer_id=customer_id,
                cluster_info=info,
            )
            template_ids.append(template.cluster_id)

        # Serve cached tier for a new episode based on the generated templates.
        embedding_service = MagicMock()
        embedding_service.embed_text_async = AsyncMock(return_value=cluster_a[0])

        tiered = TieredReflectionService(
            config=LLMConfig(openrouter_api_key=""),
            kyrodb_router=router,
            embedding_service=embedding_service,
        )

        new_episode = EpisodeCreate(
            customer_id=customer_id,
            goal="Deploy application to production",
            tool_chain=["kubectl"],
            actions_taken=["kubectl apply -f deployment.yaml"],
            error_trace="ImagePullBackOff: image not found in registry",
            error_class=ErrorClass.RESOURCE_ERROR,
            environment_info={"env": "production"},
            tags=["kubernetes", "deployment"],
        )

        selected = await tiered._select_tier(new_episode)
        assert selected == ReflectionTier.CACHED

        new_episode_id = await db.allocate_doc_id()
        cached = await tiered.generate_reflection(
            new_episode,
            episode_id=new_episode_id,
            tier=ReflectionTier.CACHED,
        )
        assert cached.tier == ReflectionTier.CACHED.value
        assert cached.cost_usd == 0.0
        assert cached.llm_model.startswith("cluster-template-")

        # Ensure usage_count increments for the matched template (best-effort).
        matched_cluster_id = int(cached.llm_model.split("cluster-template-", 1)[1])
        namespace = get_namespaced_collection(customer_id, "cluster_templates")
        template_doc = await router.text_client.query(
            doc_id=matched_cluster_id,
            namespace=namespace,
            include_embedding=False,
        )
        assert template_doc.found is True
        usage_count = int(dict(template_doc.metadata).get("usage_count", "0") or 0)
        assert usage_count >= 1

    finally:
        # Cleanup KyroDB namespaces to avoid leaking test data.
        try:
            failures_ns = get_namespaced_collection(customer_id, "failures")
            templates_ns = get_namespaced_collection(customer_id, "cluster_templates")
            if episode_ids:
                with suppress(Exception):
                    await router.text_client.batch_delete(
                        doc_ids=episode_ids, namespace=failures_ns
                    )
            if template_ids:
                with suppress(Exception):
                    await router.text_client.batch_delete(
                        doc_ids=template_ids, namespace=templates_ns
                    )
        finally:
            await router.close()
