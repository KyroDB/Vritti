#!/usr/bin/env python3
"""
Offline clustering + cached-tier template generation job.

Why this exists:
- KyroDB does not expose a server-side scan/list API.
- Vritti maintains a local episode index in the customer DB for enumeration.
- This job clusters a customer's episodes, persists cluster labels, and generates
  reusable cached-tier reflection templates.

Usage:
  python scripts/run_clustering_job.py --customer-id demo-customer

Notes:
- Requires a running KyroDB text+image setup per `docs/KYRODB_SETUP.md`.
- Template generation may call OpenRouter (PREMIUM tier) if a cluster has no
  high-quality reflection to reuse.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add repo root to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


logger = logging.getLogger(__name__)


async def run_job(customer_id: str, collection: str) -> int:
    from src.config import get_settings
    from src.hygiene.clustering import EpisodeClusterer
    from src.hygiene.templates import TemplateGenerator
    from src.ingestion.embedding import EmbeddingService
    from src.ingestion.tiered_reflection import get_tiered_reflection_service
    from src.kyrodb.router import KyroDBRouter
    from src.storage.database import get_customer_db

    settings = get_settings()

    if not settings.llm.has_any_api_key:
        logger.warning(
            "No LLM API key configured. Clustering can run, but template generation may fail "
            "if clusters do not already contain reflections."
        )

    # Connect KyroDB
    kyrodb_router = KyroDBRouter(config=settings.kyrodb)
    await kyrodb_router.connect()

    try:
        # Episode index lives in the customer DB.
        customer_db = await get_customer_db()

        # Tiered reflection service used by template generator (PREMIUM tier for templates).
        embedding_service = EmbeddingService(config=settings.embedding)
        reflection_service = (
            get_tiered_reflection_service(
                config=settings.llm,
                kyrodb_router=kyrodb_router,
                embedding_service=embedding_service,
            )
            if settings.llm.has_any_api_key
            else None
        )

        clusterer = EpisodeClusterer(
            kyrodb_router=kyrodb_router,
            episode_index=customer_db,
            min_cluster_size=settings.clustering.min_cluster_size,
            min_samples=settings.clustering.min_samples,
            metric=settings.clustering.metric,
            cluster_cache_ttl_seconds=settings.clustering.cluster_cache_ttl_seconds,
        )

        clusters = await clusterer.cluster_customer_episodes(
            customer_id=customer_id,
            collection=collection,
        )

        if not clusters:
            logger.info("No clusters generated (insufficient episodes or all noise).")
            return 0

        template_generator = TemplateGenerator(
            kyrodb_router=kyrodb_router,
            reflection_service=reflection_service,
        )

        generated = 0
        for cluster_info in clusters.values():
            try:
                await template_generator.generate_cluster_template(
                    customer_id=customer_id,
                    cluster_info=cluster_info,
                )
                generated += 1
            except Exception as e:
                logger.error(
                    f"Template generation failed for cluster {cluster_info.cluster_id}: {e}",
                    exc_info=True,
                )

        logger.info(
            f"Clustering job finished for {customer_id}: "
            f"{len(clusters)} clusters, {generated} templates generated"
        )
        return 0 if generated == len(clusters) else 2

    finally:
        await kyrodb_router.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run offline clustering + cached-tier template generation",
    )
    parser.add_argument(
        "--customer-id",
        required=True,
        help="Customer ID to cluster",
    )
    parser.add_argument(
        "--collection",
        default="failures",
        help="Collection to cluster (default: failures)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    exit_code = asyncio.run(run_job(customer_id=args.customer_id, collection=args.collection))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
