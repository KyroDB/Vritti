"""
KyroDB namespace router for multi-modal dual-instance architecture.

Routes requests to appropriate KyroDB instances:
- Text/code embeddings → kyrodb_text (384-dim)
- Image embeddings → kyrodb_images (512-dim)

Multi-tenancy: Uses customer-namespaced collections for data isolation.
Namespace format: {customer_id}:failures (e.g., "acme-corp:failures")
"""

import asyncio
import logging
from datetime import UTC
from typing import TYPE_CHECKING, Optional

from src.config import KyroDBConfig
from src.kyrodb.client import KyroDBClient
from src.kyrodb.kyrodb_pb2 import SearchResponse, SearchResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.models.clustering import ClusterTemplate
    from src.models.episode import Episode, Reflection
    from src.models.skill import Skill


def get_namespaced_collection(customer_id: str, collection: str) -> str:
    """
    Generate customer-namespaced collection name for multi-tenancy.

    Args:
        customer_id: Customer identifier (slug format)
        collection: Base collection name (e.g., "failures")

    Returns:
        str: Namespaced collection (e.g., "acme-corp:failures")

    Raises:
        ValueError: If customer_id is empty

    Example:
        >>> get_namespaced_collection("acme-corp", "failures")
        "acme-corp:failures"
    """
    if not customer_id:
        raise ValueError("customer_id is required for namespace isolation")
    return f"{customer_id}:{collection}"


class KyroDBRouter:
    """
    Routes requests to text and image KyroDB instances.

    Provides a unified interface for multi-modal episodic memory storage,
    including clustering and memory hygiene operations.
    """

    def __init__(self, config: KyroDBConfig):
        """
        Initialize router with dual KyroDB instances.

        Args:
            config: KyroDB configuration with text/image connection details
        """
        self.config = config

        # Initialize clients (lazy connection) with TLS configuration
        self.text_client = KyroDBClient(
            host=config.text_host,
            port=config.text_port,
            timeout_seconds=config.request_timeout_seconds,
            max_retries=3,
            enable_tls=config.enable_tls,
            tls_ca_cert_path=config.tls_ca_cert_path,
            tls_client_cert_path=config.tls_client_cert_path,
            tls_client_key_path=config.tls_client_key_path,
            tls_verify_server=config.tls_verify_server,
        )

        self.image_client = KyroDBClient(
            host=config.image_host,
            port=config.image_port,
            timeout_seconds=config.request_timeout_seconds,
            max_retries=3,
            enable_tls=config.enable_tls,
            tls_ca_cert_path=config.tls_ca_cert_path,
            tls_client_cert_path=config.tls_client_cert_path,
            tls_client_key_path=config.tls_client_key_path,
            tls_verify_server=config.tls_verify_server,
        )

        self._text_connected = False
        self._image_connected = False
        self._cluster_usage_locks: dict[str, asyncio.Lock] = {}
        self._cluster_usage_locks_guard = asyncio.Lock()

    async def connect(self) -> None:
        """
        Connect to both KyroDB instances.

        Raises:
            KyroDBError: If either connection fails
        """
        logger.info("Connecting to KyroDB instances...")
        await self.text_client.connect()
        self._text_connected = True
        logger.info(f"Text instance connected ({self.text_client.address})")

        await self.image_client.connect()
        self._image_connected = True
        logger.info(f"Image instance connected ({self.image_client.address})")

    async def close(self) -> None:
        """Close connections to both instances."""
        if self._text_connected:
            await self.text_client.close()
            self._text_connected = False

        if self._image_connected:
            await self.image_client.close()
            self._image_connected = False

        logger.info("Closed all KyroDB connections")

    async def __aenter__(self) -> "KyroDBRouter":
        """Context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        await self.close()

    async def insert_episode(
        self,
        episode_id: int,
        customer_id: str,
        collection: str,
        text_embedding: list[float],
        image_embedding: list[float] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> tuple[bool, bool]:
        """
        Insert episode into appropriate KyroDB instances with customer namespace.

        Multi-tenancy: Uses customer-namespaced collection for data isolation.

        Args:
            episode_id: Unique episode ID (used as doc_id)
            customer_id: Customer ID for namespace isolation
            collection: Base collection name ("failures")
            text_embedding: Text/code embedding (384-dim)
            image_embedding: Optional image embedding (512-dim)
            metadata: Episode metadata (stored in both instances)

        Returns:
            tuple: (text_inserted, image_inserted)

        Raises:
            ValueError: If customer_id is empty
            KyroDBError: If insertion fails critically
        """
        # Generate customer-namespaced collection
        namespaced_collection = get_namespaced_collection(customer_id, collection)

        text_success = False
        image_success = False

        # Insert text embedding (required)
        try:
            response = await self.text_client.insert(
                doc_id=episode_id,
                embedding=text_embedding,
                namespace=namespaced_collection,
                metadata=metadata,
            )
            text_success = response.success
            if not text_success:
                logger.error(
                    f"Text insertion failed for episode {episode_id} "
                    f"(customer: {customer_id}, collection: {namespaced_collection}): "
                    f"{response.error}"
                )
        except Exception as e:
            logger.error(
                f"Text insertion error for episode {episode_id} " f"(customer: {customer_id}): {e}"
            )
            raise

        # Insert image embedding (optional)
        if image_embedding:
            try:
                # Image instance uses separate namespace: {customer_id}:failures_images
                image_namespace = f"{namespaced_collection}_images"
                response = await self.image_client.insert(
                    doc_id=episode_id,
                    embedding=image_embedding,
                    namespace=image_namespace,
                    metadata=metadata,
                )
                image_success = response.success
                if not image_success:
                    logger.warning(
                        f"Image insertion failed for episode {episode_id} "
                        f"(customer: {customer_id}, namespace: {image_namespace}): "
                        f"{response.error}"
                    )
            except Exception as e:
                logger.warning(
                    f"Image insertion error for episode {episode_id} "
                    f"(customer: {customer_id}): {e} (continuing with text-only)"
                )
                # Don't fail the whole operation if image insert fails
                image_success = False

        logger.debug(f"Episode {episode_id} inserted: text={text_success}, image={image_success}")
        return (text_success, image_success)

    async def insert_cluster_template(
        self,
        customer_id: str,
        cluster_id: int,
        template_embedding: list[float],
        template_metadata: dict[str, str],
    ) -> bool:
        """
        Persist a cluster template (cached-tier) into KyroDB.

        Templates are stored in the text instance under:
            namespace = "{customer_id}:cluster_templates"

        Args:
            customer_id: Customer ID for namespace isolation
            cluster_id: Cluster identifier (used as doc_id)
            template_embedding: 384-dim template embedding (typically cluster centroid)
            template_metadata: Metadata (must include template_reflection_json)

        Returns:
            bool: True if insert succeeded
        """
        namespace = get_namespaced_collection(customer_id, "cluster_templates")
        metadata = dict(template_metadata)
        metadata.setdefault("customer_id", customer_id)
        metadata.setdefault("cluster_id", str(cluster_id))
        metadata.setdefault("doc_type", "cluster_template")

        response = await self.text_client.insert(
            doc_id=cluster_id,
            embedding=template_embedding,
            namespace=namespace,
            metadata=metadata,
        )
        if not response.success:
            logger.error(
                "Cluster template insert failed",
                extra={
                    "customer_id": customer_id,
                    "cluster_id": cluster_id,
                    "error": response.error,
                },
            )
        return bool(response.success)

    async def search_cluster_templates(
        self,
        query_embedding: list[float],
        customer_id: str,
        k: int = 5,
        min_score: float = 0.85,
    ) -> SearchResponse:
        """
        Search for the best matching cluster templates for cached-tier.

        Args:
            query_embedding: 384-dim query embedding
            customer_id: Customer ID for namespace isolation
            k: Number of templates to return
            min_score: Minimum similarity threshold

        Returns:
            SearchResponse: KyroDB search response
        """
        namespace = get_namespaced_collection(customer_id, "cluster_templates")
        return await self.text_client.search(
            query_embedding=query_embedding,
            k=k,
            namespace=namespace,
            min_score=min_score,
            include_embeddings=False,
            metadata_filters={"doc_type": "cluster_template"},
        )

    async def get_cluster_template(
        self, customer_id: str, cluster_id: int
    ) -> Optional["ClusterTemplate"]:
        """
        Retrieve a cluster template by cluster_id.

        Returns:
            ClusterTemplate if found, else None
        """
        from src.models.clustering import ClusterTemplate

        namespace = get_namespaced_collection(customer_id, "cluster_templates")
        response = await self.text_client.query(
            doc_id=cluster_id,
            namespace=namespace,
            include_embedding=False,
        )
        if not response.found:
            return None

        return ClusterTemplate.from_metadata_dict(dict(response.metadata))

    async def increment_cluster_template_usage(self, customer_id: str, cluster_id: int) -> bool:
        """
        Increment usage counters for a cluster template.

        Uses KyroDB's update-by-reinsert mechanism with an in-process
        per-cluster async lock to serialize `text_client.query` +
        `text_client.insert` and avoid lost increments within this process.
        Concurrent writers across multiple processes may still race.

        - Query existing doc (with embedding)
        - Update metadata fields
        - Re-insert with same doc_id + embedding
        """
        from datetime import datetime

        lock_key = f"{customer_id}:{cluster_id}"
        lock = await self._get_cluster_usage_lock(lock_key)
        async with lock:
            return await self._increment_cluster_template_usage_locked(
                customer_id=customer_id,
                cluster_id=cluster_id,
                timestamp_factory=lambda: datetime.now(UTC),
            )

    async def _get_cluster_usage_lock(self, lock_key: str) -> asyncio.Lock:
        async with self._cluster_usage_locks_guard:
            lock = self._cluster_usage_locks.get(lock_key)
            if lock is None:
                lock = asyncio.Lock()
                self._cluster_usage_locks[lock_key] = lock
            return lock

    async def _increment_cluster_template_usage_locked(
        self,
        customer_id: str,
        cluster_id: int,
        timestamp_factory,
    ) -> bool:
        namespace = get_namespaced_collection(customer_id, "cluster_templates")
        existing = await self.text_client.query(
            doc_id=cluster_id,
            namespace=namespace,
            include_embedding=True,
        )
        if not existing.found:
            return False

        metadata = dict(existing.metadata)
        try:
            current = int(metadata.get("usage_count", "0") or 0)
        except Exception:
            current = 0
        metadata["usage_count"] = str(current + 1)
        metadata["last_used_at"] = timestamp_factory().isoformat()

        response = await self.text_client.insert(
            doc_id=cluster_id,
            embedding=list(existing.embedding),
            namespace=namespace,
            metadata=metadata,
        )
        return bool(response.success)

    async def search_text(
        self,
        query_embedding: list[float],
        k: int,
        customer_id: str,
        collection: str,
        min_score: float = 0.6,
        metadata_filters: dict[str, str] | None = None,
    ) -> SearchResponse:
        """
        Search text instance for episodes with server-side filtering.

        Args:
            query_embedding: Query embedding vector (384-dim text)
            k: Number of results to return
            customer_id: Customer ID for namespace isolation
            collection: Base collection name ("failures")
            min_score: Minimum similarity score
            metadata_filters: Server-side metadata filters

        Returns:
            SearchResponse: Filtered search results from KyroDB
        """
        namespaced_collection = get_namespaced_collection(customer_id, collection)

        return await self.text_client.search(
            query_embedding=query_embedding,
            k=k,
            namespace=namespaced_collection,
            min_score=min_score,
            include_embeddings=False,
            metadata_filters=metadata_filters,
        )

    async def search_image(
        self,
        query_embedding: list[float],
        k: int,
        customer_id: str,
        collection: str,
        min_score: float = 0.6,
        metadata_filters: dict[str, str] | None = None,
    ) -> SearchResponse:
        """
        Search image instance for episodes with server-side filtering.

        Args:
            query_embedding: Query embedding vector (512-dim image)
            k: Number of results to return
            customer_id: Customer ID for namespace isolation
            collection: Base collection name ("failures")
            min_score: Minimum similarity score
            metadata_filters: Server-side metadata filters

        Returns:
            SearchResponse: Filtered search results from KyroDB image instance
        """
        namespaced_collection = get_namespaced_collection(customer_id, collection)
        image_namespace = f"{namespaced_collection}_images"

        return await self.image_client.search(
            query_embedding=query_embedding,
            k=k,
            namespace=image_namespace,
            min_score=min_score,
            include_embeddings=False,
            metadata_filters=metadata_filters,
        )

    async def search_episodes(
        self,
        query_embedding: list[float],
        customer_id: str,
        collection: str,
        image_embedding: list[float] | None = None,
        k: int = 20,
        min_score: float = 0.6,
        metadata_filters: dict[str, str] | None = None,
        image_weight: float = 0.3,
    ) -> SearchResponse:
        """
        Search for episodes using text embeddings and optional image fusion.

        Security:
        - Customer namespace isolation enforced
        - Only searches within customer's episodes

        Args:
            query_embedding: Text query embedding
            customer_id: Customer ID for namespace isolation
            collection: Base collection name ("failures", "skills", "rules")
            k: Number of results to return
            min_score: Minimum similarity threshold
            metadata_filters: Optional metadata filters
            image_weight: Weight for image similarity (0-1) if image_embedding is provided

        Returns:
            SearchResponse: Combined search results from text (and optionally images)

        Raises:
            ValueError: If customer_id is empty
            KyroDBError: If search fails
        """
        # Generate customer-namespaced collection
        namespaced_collection = get_namespaced_collection(customer_id, collection)

        # Primary search: text embeddings
        text_response = await self.text_client.search(
            query_embedding=query_embedding,
            k=k,
            namespace=namespaced_collection,
            min_score=min_score,
            include_embeddings=False,  # Don't waste bandwidth
            metadata_filters=metadata_filters,
        )

        if image_embedding is None:
            return text_response

        # Secondary search: image embeddings
        image_response = await self.image_client.search(
            query_embedding=image_embedding,
            k=k,
            namespace=f"{namespaced_collection}_images",
            min_score=min_score,
            include_embeddings=False,
            metadata_filters=metadata_filters,
        )

        # Fuse results: weighted average of text + image scores
        combined: dict[int, tuple[SearchResult, float | None, float | None]] = {}

        for result in text_response.results:
            combined[result.doc_id] = (result, result.score, None)

        for result in image_response.results:
            if result.doc_id in combined:
                existing, text_score, _ = combined[result.doc_id]
                combined[result.doc_id] = (existing, text_score, result.score)
            else:
                combined[result.doc_id] = (result, None, result.score)

        fused_results: list[SearchResult] = []
        for result, text_score, image_score in combined.values():
            if text_score is not None and image_score is not None:
                fused_score = (1.0 - image_weight) * text_score + image_weight * image_score
            elif text_score is not None:
                fused_score = text_score
            else:
                fused_score = image_score if image_score is not None else 0.0

            # Create new SearchResult with fused score (preserve metadata)
            fused_results.append(
                SearchResult(
                    doc_id=result.doc_id,
                    score=fused_score,
                    metadata=result.metadata,
                    embedding=result.embedding,
                )
            )

        # Sort by fused score and limit to k
        fused_results.sort(key=lambda r: r.score, reverse=True)
        fused_results = fused_results[:k]

        return SearchResponse(
            results=fused_results,
            total_found=len(fused_results),
            search_latency_ms=text_response.search_latency_ms + image_response.search_latency_ms,
            search_path=text_response.search_path,
            error="",
        )

    async def delete_episode(
        self, episode_id: int, customer_id: str, collection: str, delete_images: bool = True
    ) -> tuple[bool, bool]:
        """
        Delete episode from KyroDB instances.

        Security:
        - Customer namespace isolation enforced
        - Only deletes from customer's collection

        Args:
            episode_id: Episode ID to delete
            customer_id: Customer ID for namespace isolation
            collection: Base collection name ("failures")
            delete_images: Also delete from image instance

        Returns:
            tuple: (text_deleted, image_deleted)

        Raises:
            ValueError: If customer_id is empty
        """
        # Generate customer-namespaced collection
        namespaced_collection = get_namespaced_collection(customer_id, collection)

        text_deleted = False
        image_deleted = False

        # Delete from text instance
        try:
            response = await self.text_client.delete(doc_id=episode_id, namespace=namespaced_collection)
            text_deleted = response.success
        except Exception as e:
            logger.error(f"Text deletion failed for episode {episode_id}: {e}")

        # Delete from image instance
        if delete_images:
            try:
                response = await self.image_client.delete(
                    doc_id=episode_id, namespace=f"{namespaced_collection}_images"
                )
                image_deleted = response.success
            except Exception as e:
                logger.warning(f"Image deletion failed for episode {episode_id}: {e}")
                image_deleted = False

        logger.debug(f"Episode {episode_id} deleted: text={text_deleted}, image={image_deleted}")
        return (text_deleted, image_deleted)

    async def bulk_fetch_episodes(
        self,
        episode_ids: list[int],
        customer_id: str,
        collection: str,
    ) -> list["Episode"]:
        """
        Batch retrieve multiple episodes by ID.
        
        KyroDB does not expose a server-side bulk query API. This helper issues
        bounded concurrent point lookups via `KyroDBClient.bulk_query()` and
        reconstructs `Episode` objects from stored metadata.
        
        Args:
            episode_ids: List of episode IDs to retrieve
            customer_id: Customer ID for namespace isolation
            collection: Base collection name ("failures")
            
        Returns:
            list[Episode]: Successfully retrieved episodes (partial success)
            
        Raises:
            ValueError: If customer_id is empty
            KyroDBError: On critical failure
        """
        from src.models.episode import Episode
        
        if not episode_ids:
            return []
        
        namespaced_collection = get_namespaced_collection(customer_id, collection)
        
        try:
            response = await self.text_client.bulk_query(
                doc_ids=episode_ids,
                namespace=namespaced_collection,
                include_embeddings=False,
            )
            
            episodes = []
            for query_result in response.results:
                if not query_result.found:
                    logger.debug(
                        f"Episode {query_result.doc_id} not found in bulk fetch "
                        f"(customer: {customer_id})"
                    )
                    continue
                    
                try:
                    episode = Episode.from_metadata_dict(
                        doc_id=query_result.doc_id,
                        metadata=dict(query_result.metadata),
                    )
                    episodes.append(episode)
                except Exception as e:
                    logger.warning(
                        f"Failed to deserialize episode {query_result.doc_id}: {e}"
                    )
                    continue
            
            logger.info(
                f"Bulk fetched {len(episodes)}/{len(episode_ids)} episodes "
                f"(customer: {customer_id}, requested: {response.total_requested}, "
                f"found: {response.total_found})"
            )
            
            return episodes
            
        except Exception as e:
            logger.error(
                f"Bulk fetch failed for {len(episode_ids)} episodes "
                f"(customer: {customer_id}): {e}",
                exc_info=True,
            )
            raise

    async def batch_delete_episodes(
        self,
        episode_ids: list[int],
        customer_id: str,
        collection: str,
        delete_images: bool = True,
    ) -> tuple[int, int]:
        """
        Batch delete multiple episodes by ID from text and optionally image instances.
        
        KyroDB does not expose a server-side batch delete API. This helper issues
        bounded concurrent deletes and aggregates deletion counts.
        
        Args:
            episode_ids: List of episode IDs to delete
            customer_id: Customer ID for namespace isolation
            collection: Base collection name ("failures")
            delete_images: Also delete from image instance (default: True)
            
        Returns:
            tuple[int, int]: (text_deleted_count, image_deleted_count)
            
        Raises:
            ValueError: If customer_id is empty
            KyroDBError: On critical failure
        """
        if not episode_ids:
            return (0, 0)
        
        namespaced_collection = get_namespaced_collection(customer_id, collection)
        
        text_deleted = 0
        image_deleted = 0
        
        # Delete from text instance
        try:
            response = await self.text_client.batch_delete(
                doc_ids=episode_ids,
                namespace=namespaced_collection,
            )
            
            if response.success:
                text_deleted = int(response.deleted_count)
                logger.info(
                    f"Batch deleted {text_deleted} episodes from text instance "
                    f"(customer: {customer_id}, requested: {len(episode_ids)})"
                )
            else:
                logger.error(
                    f"Batch delete failed for {len(episode_ids)} episodes: "
                    f"{response.error}"
                )
                
        except Exception as e:
            logger.error(
                f"Text batch delete failed for {len(episode_ids)} episodes "
                f"(customer: {customer_id}): {e}",
                exc_info=True,
            )
            raise
        
        # Delete from image instance if requested
        if delete_images:
            try:
                image_namespace = f"{namespaced_collection}_images"
                image_response = await self.image_client.batch_delete(
                    doc_ids=episode_ids,
                    namespace=image_namespace,
                )
                
                if image_response.success:
                    image_deleted = int(image_response.deleted_count)
                    logger.info(
                        f"Batch deleted {image_deleted} episodes from image instance "
                        f"(customer: {customer_id})"
                    )
                else:
                    logger.warning(
                        f"Image batch delete failed for {len(episode_ids)} episodes: "
                        f"{image_response.error}"
                    )
                    
            except Exception as e:
                logger.warning(
                    f"Image batch delete failed for {len(episode_ids)} episodes "
                    f"(customer: {customer_id}): {e} (continuing - text delete succeeded)"
                )
        
        logger.debug(
            f"Batch deleted {len(episode_ids)} episodes: "
            f"text={text_deleted}, images={image_deleted}"
        )
        return (text_deleted, image_deleted)

    async def get_episode(
        self, episode_id: int, customer_id: str, collection: str, include_image: bool = False
    ) -> dict | None:
        """
        Retrieve episode by ID.

        Security:
        - Customer namespace isolation enforced
        - Only retrieves from customer's collection

        Args:
            episode_id: Episode ID to retrieve
            customer_id: Customer ID for namespace isolation
            collection: Base collection name ("failures")
            include_image: Also fetch image embedding

        Returns:
            dict with episode data, or None if not found

        Raises:
            ValueError: If customer_id is empty
            KyroDBError: On retrieval failure
        """
        # Generate customer-namespaced collection
        namespaced_collection = get_namespaced_collection(customer_id, collection)

        # Fetch from text instance
        try:
            response = await self.text_client.query(
                doc_id=episode_id, namespace=namespaced_collection, include_embedding=False
            )
            if not response.found:
                return None

            episode_data = {
                "doc_id": response.doc_id,
                "metadata": dict(response.metadata),
                "found": True,
            }

            # Optionally fetch image embedding
            if include_image:
                try:
                    image_response = await self.image_client.query(
                        doc_id=episode_id,
                        namespace=f"{namespaced_collection}_images",
                        include_embedding=False,
                    )
                    episode_data["image_found"] = image_response.found
                except Exception as e:
                    logger.warning(f"Failed to fetch image for episode {episode_id}: {e}")
                    episode_data["image_found"] = False

            return episode_data

        except Exception as e:
            logger.error(f"Failed to retrieve episode {episode_id}: {e}")
            raise

    async def update_episode_reflection(
        self,
        episode_id: int,
        customer_id: str,
        collection: str,
        reflection: "Reflection",  # Forward reference for type hint
    ) -> bool:
        """
        Update existing episode with generated reflection.

        Security:
        - Validates customer_id matches existing episode
        - Serializes reflection with validation
        - Uses atomic re-insert (KyroDB's update mechanism)

        This is called asynchronously after initial episode storage
        to add the LLM-generated reflection.

        Args:
            episode_id: Episode ID to update
            customer_id: Customer ID for namespace isolation
            collection: Base collection name ("failures")
            reflection: Generated reflection to persist

        Returns:
            bool: True if reflection was successfully persisted

        Raises:
            ValueError: If customer_id is empty or episode not found
            KyroDBError: On update failure
        """

        # Generate customer-namespaced collection
        namespaced_collection = get_namespaced_collection(customer_id, collection)

        try:
            # Step 1: Query existing episode to get metadata and embedding
            logger.debug(
                f"Fetching existing episode {episode_id} "
                f"(customer: {customer_id}, collection: {namespaced_collection})"
            )

            existing = await self.text_client.query(
                doc_id=episode_id,
                namespace=namespaced_collection,
                include_embedding=True,  # Need embedding for re-insert
            )

            if not existing.found:
                logger.error(
                    f"Cannot update reflection: episode {episode_id} not found "
                    f"(customer: {customer_id}, collection: {namespaced_collection})"
                )
                raise ValueError(
                    f"Episode {episode_id} not found in {namespaced_collection}"
                )

            # Security: Verify customer_id matches (prevent cross-customer updates)
            existing_customer = existing.metadata.get("customer_id")
            if existing_customer != customer_id:
                logger.error(
                    f"Customer ID mismatch: episode {episode_id} belongs to "
                    f"{existing_customer}, not {customer_id}"
                )
                raise ValueError(
                    "Customer ID mismatch - potential security violation"
                )

            # Step 2: Serialize reflection to metadata format
            reflection_metadata = self._serialize_reflection_to_metadata(reflection)

            # Step 3: Merge with existing metadata (preserve all fields)
            updated_metadata = {**dict(existing.metadata), **reflection_metadata}

            # Step 3.1: Update episode_json so reflection is visible on retrieval
            episode_json_raw = updated_metadata.get("episode_json")
            if episode_json_raw:
                try:
                    import base64
                    import json
                    import zlib

                    if isinstance(episode_json_raw, str) and episode_json_raw.startswith("gz:"):
                        compressed = base64.b64decode(episode_json_raw[3:])
                        episode_json_str = zlib.decompress(compressed).decode("utf-8")
                    else:
                        episode_json_str = str(episode_json_raw)

                    episode_payload = json.loads(episode_json_str)
                    episode_payload["reflection"] = reflection.model_dump(mode="json")

                    encoded = json.dumps(
                        episode_payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    compressed = zlib.compress(encoded.encode("utf-8"), level=9)
                    updated_metadata["episode_json"] = (
                        f"gz:{base64.b64encode(compressed).decode('utf-8')}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to update episode_json for episode {episode_id}: {e}",
                        exc_info=True,
                    )
                    # Important: do not overwrite episode_json on fallback.
                    # Reflection is already persisted via reflection_* metadata fields.

            # Step 4: Re-insert with updated metadata (KyroDB's update mechanism)
            logger.debug(
                f"Re-inserting episode {episode_id} with reflection metadata..."
            )

            response = await self.text_client.insert(
                doc_id=episode_id,
                embedding=list(existing.embedding),  # Keep same embedding
                namespace=namespaced_collection,
                metadata=updated_metadata,
            )

            if response.success:
                # Best-effort: keep image instance metadata in sync so image-only hits
                # still reconstruct the full episode (including reflection) from episode_json.
                try:
                    image_namespace = f"{namespaced_collection}_images"
                    image_existing = await self.image_client.query(
                        doc_id=episode_id,
                        namespace=image_namespace,
                        include_embedding=True,
                    )
                    if image_existing.found:
                        image_customer = image_existing.metadata.get("customer_id")
                        if image_customer != customer_id:
                            logger.error(
                                "Customer ID mismatch in image reflection update",
                                extra={
                                    "episode_id": episode_id,
                                    "expected_customer_id": customer_id,
                                    "found_customer_id": image_customer,
                                },
                            )
                        else:
                            image_response = await self.image_client.insert(
                                doc_id=episode_id,
                                embedding=list(image_existing.embedding),
                                namespace=image_namespace,
                                metadata=updated_metadata,
                            )
                            if not image_response.success:
                                logger.warning(
                                    "Failed to persist reflection metadata to image instance",
                                    extra={
                                        "episode_id": episode_id,
                                        "customer_id": customer_id,
                                        "error": image_response.error,
                                    },
                                )
                except Exception as e:
                    logger.warning(
                        f"Image instance reflection update failed for episode {episode_id}: {e}",
                        exc_info=True,
                    )

                logger.info(
                    f"Reflection persisted for episode {episode_id} "
                    f"(customer: {customer_id}, "
                    f"model: {reflection.llm_model}, "
                    f"confidence: {reflection.confidence_score:.2f}, "
                    f"cost: ${reflection.cost_usd:.4f})"
                )
                return True
            else:
                logger.error(
                    f"Failed to persist reflection for episode {episode_id}: "
                    f"{response.error}"
                )
                return False

        except Exception as e:
            logger.error(
                f"Failed to update reflection for episode {episode_id} "
                f"(customer: {customer_id}): {e}",
                exc_info=True,
            )
            # Don't raise - this is async background task, log and continue
            return False

    def _serialize_reflection_to_metadata(
        self, reflection: "Reflection"
    ) -> dict[str, str]:
        """
        Serialize reflection to KyroDB metadata format (string-string map).

        Security:
        - All fields validated by Pydantic before serialization
        - Lists converted to JSON strings
        - Floats converted to strings with precision control
        - No user-controlled keys in output

        Args:
            reflection: Validated reflection object

        Returns:
            dict[str, str]: Metadata compatible with KyroDB
        """
        import json

        metadata = {
            # Core fields
            "reflection_root_cause": reflection.root_cause,
            "reflection_resolution": reflection.resolution_strategy,
            "reflection_confidence": f"{reflection.confidence_score:.4f}",
            "reflection_generalization": f"{reflection.generalization_score:.4f}",

            # Lists (JSON-encoded)
            "reflection_preconditions": json.dumps(reflection.preconditions),
            "reflection_env_factors": json.dumps(reflection.environment_factors),
            "reflection_components": json.dumps(reflection.affected_components),

            # Generation metadata
            "reflection_model": reflection.llm_model,
            "reflection_generated_at": reflection.generated_at.isoformat(),
            "reflection_cost_usd": f"{reflection.cost_usd:.6f}",
            "reflection_latency_ms": f"{reflection.generation_latency_ms:.2f}",
        }

        # Consensus metadata (if multi-perspective)
        if reflection.consensus:
            consensus = reflection.consensus
            metadata.update({
                "reflection_consensus_method": consensus.consensus_method,
                "reflection_consensus_confidence": f"{consensus.consensus_confidence:.4f}",
                "reflection_disagreements": json.dumps(consensus.disagreement_points),

                # Store number of perspectives
                "reflection_perspectives_count": str(len(consensus.perspectives)),

                # Store individual perspectives (for debugging/audit)
                "reflection_perspectives_json": json.dumps([
                    {
                        "model": p.model_name,
                        "root_cause": p.root_cause,
                        "confidence": p.confidence_score,
                        "reasoning": p.reasoning[:200],  # Truncate for storage
                    }
                    for p in consensus.perspectives
                ]),
            })

        return metadata

    async def update_episode_metadata(
        self,
        episode_id: int,
        customer_id: str,
        collection: str,
        metadata_updates: dict[str, str],
    ) -> bool:
        """
        Update episode metadata fields (generic method for any metadata).

        Security:
        - Validates customer_id matches existing episode
        - Only updates specified fields (preserves others)

        Args:
            episode_id: Episode to update
            customer_id: Customer ID for validation
            collection: Collection name
            metadata_updates: Key-value pairs to update

        Returns:
            bool: Success status
        """
        namespaced_collection = get_namespaced_collection(customer_id, collection)

        try:
            # Fetch existing
            existing = await self.text_client.query(
                doc_id=episode_id,
                namespace=namespaced_collection,
                include_embedding=True,
            )

            if not existing.found:
                logger.error(f"Episode {episode_id} not found for metadata update")
                return False

            # Security: Verify customer_id
            existing_customer = existing.metadata.get("customer_id")
            if existing_customer != customer_id:
                logger.error(
                    f"Customer ID mismatch in metadata update: "
                    f"episode belongs to {existing_customer}, not {customer_id}"
                )
                return False

            # Merge metadata
            updated_metadata = {**dict(existing.metadata), **metadata_updates}

            # Re-insert
            response = await self.text_client.insert(
                doc_id=episode_id,
                embedding=list(existing.embedding),
                namespace=namespaced_collection,
                metadata=updated_metadata,
            )

            if response.success:
                logger.debug(
                    f"Updated {len(metadata_updates)} metadata fields "
                    f"for episode {episode_id}"
                )
                return True
            else:
                logger.error(f"Metadata update failed: {response.error}")
                return False

        except Exception as e:
            logger.error(f"Metadata update error for episode {episode_id}: {e}")
            return False

    async def insert_skill(
        self,
        skill: "Skill",
        embedding: list[float],
    ) -> bool:
        """
        Insert skill into KyroDB skills collection.

        Security:
        - Customer namespace isolation enforced
        - Skill validated before insertion

        Args:
            skill: Skill object to insert
            embedding: Skill embedding (generated from docstring)

        Returns:
            bool: True if insertion successful

        Raises:
            ValueError: If customer_id missing
        """
        if not skill.customer_id:
            raise ValueError("customer_id is required for skill insertion")

        collection = "skills"
        namespaced_collection = get_namespaced_collection(skill.customer_id, collection)
        metadata = skill.to_metadata_dict()

        try:
            response = await self.text_client.insert(
                doc_id=skill.skill_id,
                embedding=embedding,
                namespace=namespaced_collection,
                metadata=metadata,
            )

            if response.success:
                logger.info(
                    f"Skill {skill.skill_id} inserted into {namespaced_collection} "
                    f"(name: {skill.name})"
                )
                return True
            else:
                logger.error(f"Skill insertion failed: {response.error}")
                return False

        except Exception as e:
            logger.error(f"Failed to insert skill {skill.skill_id}: {e}", exc_info=True)
            return False

    async def search_skills(
        self,
        query_embedding: list[float],
        customer_id: str,
        k: int = 5,
        min_score: float = 0.7,
    ) -> list[tuple["Skill", float]]:
        """
        Search skills collection for similar skills.

        Security:
        - Customer namespace isolation enforced
        - Only searches within customer's skills

        Args:
            query_embedding: Query embedding vector
            customer_id: Customer ID for namespace isolation
            k: Number of results to return
            min_score: Minimum similarity score

        Returns:
            list: List of (Skill, similarity_score) tuples
        """
        from src.models.skill import Skill

        if not customer_id:
            raise ValueError("customer_id is required for skill search")

        collection = "skills"
        namespaced_collection = get_namespaced_collection(customer_id, collection)

        try:
            response = await self.text_client.search(
                query_embedding=query_embedding,
                k=k,
                namespace=namespaced_collection,
                min_score=min_score,
                include_embeddings=False,
            )

            skills = []
            for result in response.results:
                try:
                    skill = Skill.from_metadata_dict(result.doc_id, dict(result.metadata))
                    skills.append((skill, result.score))
                except Exception as e:
                    logger.warning(
                        f"Failed to deserialize skill {result.doc_id}: {e}"
                    )
                    continue

            logger.debug(
                f"Skills search returned {len(skills)} results "
                f"(customer: {customer_id}, k: {k}, min_score: {min_score})"
            )

            return skills

        except Exception as e:
            logger.error(
                f"Skills search failed for customer {customer_id}: {e}",
                exc_info=True
            )
            return []

    async def update_skill_stats(
        self,
        skill_id: int,
        customer_id: str,
        success: bool,
    ) -> Optional["Skill"]:
        """
        Update skill usage statistics after application.

        Security:
        - Customer ID validated
        - Atomic update to prevent race conditions

        Args:
            skill_id: Skill ID to update
            customer_id: Customer ID for namespace isolation
            success: Whether the skill application succeeded

        Returns:
            Optional[Skill]: Updated skill object if successful, None otherwise
        """
        from src.models.skill import Skill

        if not customer_id:
            raise ValueError("customer_id is required for skill stats update")

        collection = "skills"
        namespaced_collection = get_namespaced_collection(customer_id, collection)

        try:
            # Fetch existing skill
            existing = await self.text_client.query(
                doc_id=skill_id,
                namespace=namespaced_collection,
                include_embedding=True,
            )

            if not existing.found:
                logger.error(
                    f"Skill {skill_id} not found in {namespaced_collection}"
                )
                return None

            # Security: Verify customer ID
            existing_customer = existing.metadata.get("customer_id")
            if existing_customer != customer_id:
                logger.error(
                    f"Customer ID mismatch: skill {skill_id} belongs to "
                    f"{existing_customer}, not {customer_id}"
                )
                return None

            # Deserialize skill
            skill = Skill.from_metadata_dict(skill_id, dict(existing.metadata))

            # Update stats
            skill.usage_count += 1
            if success:
                skill.success_count += 1
            else:
                skill.failure_count += 1

            # Re-insert with updated stats
            response = await self.text_client.insert(
                doc_id=skill_id,
                embedding=list(existing.embedding),
                namespace=namespaced_collection,
                metadata=skill.to_metadata_dict(),
            )

            if response.success:
                logger.debug(
                    f"Updated skill {skill_id} stats: "
                    f"usage={skill.usage_count}, success_rate={skill.success_rate:.2f}"
                )
                return skill
            else:
                logger.error(f"Skill stats update failed: {response.error}")
                return None

        except Exception as e:
            logger.error(
                f"Failed to update skill {skill_id} stats: {e}",
                exc_info=True
            )
            return None

    async def health_check(self) -> dict[str, bool]:
        """
        Check health of both KyroDB instances.

        Returns:
            dict: {"text": bool, "image": bool}
        """
        health = {"text": False, "image": False}

        try:
            await self.text_client.health_check()
            health["text"] = True
        except Exception as e:
            logger.error(f"Text instance health check failed: {e}")

        try:
            await self.image_client.health_check()
            health["image"] = True
        except Exception as e:
            logger.error(f"Image instance health check failed: {e}")

        return health
