"""
Episode clustering service using HDBSCAN for memory consolidation.

Implements density-based clustering to identify common failure patterns
and generate reusable reflection templates.

Security:
- Customer isolation enforced
- Input validation on all parameters
- Thread-safe cluster cache
- Audit logging for all operations

Performance:
- Cached cluster centroids
- Batch operations for KyroDB
- Async I/O throughout
- Optimized numpy operations
"""

import asyncio
import logging
import threading
import time
from typing import Protocol, cast

import numpy as np
from hdbscan import HDBSCAN

from src.kyrodb.router import KyroDBRouter, get_namespaced_collection
from src.models.clustering import ClusterInfo, ClusterTemplate

logger = logging.getLogger(__name__)


class EpisodeIndex(Protocol):
    async def allocate_doc_id(self, *, scope: str = "kyrodb_text") -> int: ...

    async def list_episode_ids(
        self,
        *,
        customer_id: str,
        collection: str,
        include_archived: bool = False,
        include_deleted: bool = False,
        limit: int | None = None,
    ) -> list[int]: ...


class EpisodeClusterer:
    """
    Cluster similar episodes using HDBSCAN for cached reflection templates.

    Strategy:
    - Group episodes by semantic similarity (cosine distance on embeddings)
    - Generate one high-quality reflection per cluster (premium tier)
    - Reuse template for all new episodes matching the cluster

    Performance:
    - Centroid cache (in-memory, per-customer)
    - Batch KyroDB operations
    - Async throughout

    Thread Safety:
    - Lock for cache updates
    - Safe for concurrent clustering jobs
    """

    def __init__(
        self,
        kyrodb_router: KyroDBRouter,
        episode_index: EpisodeIndex | None = None,
        min_cluster_size: int = 5,
        min_samples: int = 3,
        metric: str = "cosine",
        cluster_cache_ttl_seconds: int = 3600,
    ):
        """
        Initialize episode clusterer.

        Args:
            kyrodb_router: KyroDB router for data access
            min_cluster_size: Minimum episodes to form a cluster
            min_samples: HDBSCAN min_samples parameter
            metric: Distance metric (cosine recommended for embeddings)
            cluster_cache_ttl_seconds: Cache TTL for cluster centroids

        Security:
            - min_cluster_size >= 3 (prevents overfitting)
            - metric must be supported by HDBSCAN backend

        Note:
            We cluster on **L2-normalized** embeddings. Cosine distance is therefore
            equivalent to Euclidean distance on the unit hypersphere. In practice:
            - metric="cosine" is implemented as HDBSCAN(metric="euclidean") on normalized vectors
        """
        if min_cluster_size < 3:
            raise ValueError("min_cluster_size must be >= 3 for robust clusters")

        metric_clean = (metric or "").strip().lower()
        if not metric_clean:
            raise ValueError("metric is required")

        # HDBSCAN uses sklearn's DistanceMetric via BallTree/KDTree; "cosine" is not a
        # supported fast metric there. Since our embeddings are normalized, Euclidean
        # distance is a correct proxy for cosine distance.
        if metric_clean in {"cosine", "euclidean", "l2"}:
            hdbscan_metric = "euclidean"
        else:
            raise ValueError(
                f"Unsupported clustering metric '{metric_clean}'. "
                "Use 'cosine' (recommended) or 'euclidean'."
            )

        self.kyrodb_router = kyrodb_router
        self.episode_index = episode_index
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric_clean
        self._hdbscan_metric = hdbscan_metric
        self.cache_ttl = cluster_cache_ttl_seconds

        # Thread-safe cluster cache: {customer_id: (timestamp, {cluster_id: centroid})}
        self._cluster_cache: dict[str, tuple[float, dict[int, np.ndarray]]] = {}
        self._cache_lock = threading.Lock()

        logger.info(
            f"EpisodeClusterer initialized "
            f"(min_cluster_size={min_cluster_size}, "
            f"min_samples={min_samples}, "
            f"metric={metric_clean}, "
            f"hdbscan_metric={hdbscan_metric})"
        )

    async def cluster_customer_episodes(
        self, customer_id: str, collection: str = "failures", invalidate_cache: bool = True
    ) -> dict[int, ClusterInfo]:
        """
        Cluster all active episodes for a customer.

        Args:
            customer_id: Customer to cluster
            collection: Episode collection
            invalidate_cache: Clear cache after clustering

        Returns:
            Dict mapping cluster_id → ClusterInfo

        Security:
            - customer_id validated
            - Only active (non-archived) episodes

        Performance:
            - Async episode fetching
            - Batch numpy operations
            - ~1-2s for 1000 episodes
        """
        logger.info(f"Starting clustering for customer {customer_id}...")
        start_time = time.perf_counter()

        # Validate customer_id
        if not customer_id or len(customer_id) > 100:
            raise ValueError(f"Invalid customer_id: {customer_id}")

        # Fetch all active episode embeddings (requires episode enumeration).
        episode_embeddings = await self._fetch_active_episode_embeddings(customer_id, collection)

        if len(episode_embeddings) < self.min_cluster_size:
            logger.info(
                f"Not enough episodes to cluster for {customer_id}: "
                f"{len(episode_embeddings)} < {self.min_cluster_size}"
            )
            return {}

        # Validate embeddings before clustering
        embedding_dim = len(episode_embeddings[0][1])
        if embedding_dim == 0:
            raise ValueError("Episode embedding dimension cannot be 0")
        for episode_id, embedding in episode_embeddings:
            if not embedding:
                raise ValueError(f"Episode {episode_id} missing embedding")
            if len(embedding) != embedding_dim:
                raise ValueError(
                    f"Episode {episode_id} has inconsistent embedding dimension: "
                    f"{len(embedding)} vs {embedding_dim}"
                )

        # Extract embeddings (numpy array for HDBSCAN)
        episode_ids = [episode_id for episode_id, _ in episode_embeddings]
        embeddings = np.array([embedding for _, embedding in episode_embeddings], dtype=np.float32)

        # Normalize embeddings so Euclidean distance is a valid proxy for cosine distance.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        logger.info(
            f"Clustering {len(episode_embeddings)} episodes for {customer_id} using HDBSCAN "
            f"(min_cluster_size={self.min_cluster_size}, min_samples={self.min_samples})"
        )

        # Run HDBSCAN clustering
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self._hdbscan_metric,
            cluster_selection_method="eom",  # Excess of Mass for stable clusters
        )

        cluster_labels = clusterer.fit_predict(embeddings)

        # Count clusters (excluding noise = -1)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        logger.info(f"HDBSCAN found {n_clusters} clusters, {n_noise} noise points")

        # Build cluster info
        clusters: dict[int, ClusterInfo] = {}
        label_to_cluster_id: dict[int, int] = {}

        for label in set(cluster_labels):
            if label == -1:  # Noise points
                continue

            # Get episodes in this cluster
            cluster_mask = cluster_labels == label
            cluster_episode_ids = [
                episode_ids[i] for i in range(len(episode_ids)) if cluster_mask[i]
            ]
            cluster_embeddings = embeddings[cluster_mask]

            # Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0)

            # Calculate avg intra-cluster similarity
            avg_similarity = self._calculate_avg_similarity(cluster_embeddings)

            # Allocate a globally unique KyroDB doc_id for this cluster template.
            # KyroDB namespaces are metadata-only; doc_id collisions across namespaces/types
            # would overwrite data. Cluster IDs must therefore come from the same global
            # allocator used for episodes/skills.
            label_int = int(label)
            episode_index = self.episode_index
            if episode_index is None:
                raise RuntimeError("episode_index is required for cluster ID allocation")
            cluster_id = await episode_index.allocate_doc_id()
            label_to_cluster_id[label_int] = cluster_id
            clusters[cluster_id] = ClusterInfo(
                cluster_id=cluster_id,
                customer_id=customer_id,
                episode_ids=cluster_episode_ids,
                centroid_embedding=centroid.tolist(),
                avg_intra_cluster_similarity=avg_similarity,
            )

        # Persist cluster labels to KyroDB
        await self._persist_cluster_labels(
            customer_id,
            collection,
            episode_ids,
            cluster_labels,
            label_to_cluster_id=label_to_cluster_id,
        )

        # Update cache
        if invalidate_cache:
            self._update_cache(customer_id, clusters)

        duration = time.perf_counter() - start_time

        # Log statistics
        total_clustered = sum(len(c.episode_ids) for c in clusters.values())
        noise_count = np.sum(cluster_labels == -1)

        logger.info(
            f"Clustering complete for {customer_id}: "
            f"{len(clusters)} clusters, "
            f"{total_clustered}/{len(episode_embeddings)} episodes clustered, "
            f"{noise_count} noise, "
            f"duration: {duration:.1f}s"
        )

        return clusters

    async def find_matching_cluster(
        self,
        episode_embedding: list[float],
        customer_id: str,
        similarity_threshold: float = 0.85,
        k: int = 5,
    ) -> ClusterTemplate | None:
        """
        Find cluster that best matches the episode embedding.

        Args:
            episode_embedding: Episode embedding to match
            customer_id: Customer ID for namespace isolation
            similarity_threshold: Minimum similarity to match

        Returns:
            ClusterTemplate if match found, else None

        Raises:
            ValueError: If episode_embedding is invalid
        """
        # Validate episode_embedding
        if not episode_embedding:
            raise ValueError("episode_embedding cannot be empty")
        if not isinstance(episode_embedding, list | np.ndarray):
            raise ValueError("episode_embedding must be a list or numpy array")
        if not all(isinstance(x, int | float | np.number) for x in episode_embedding):
            raise ValueError("episode_embedding must contain only numbers")

        # Preferred path: search persisted cluster templates in KyroDB (authoritative).
        try:
            search_fn = getattr(self.kyrodb_router, "search_cluster_templates", None)
            if search_fn is not None:
                response = await search_fn(
                    query_embedding=episode_embedding,
                    customer_id=customer_id,
                    k=k,
                    min_score=similarity_threshold,
                )
                raw_results = getattr(response, "results", None)
                try:
                    results = list(raw_results) if raw_results else []
                except TypeError:
                    results = []
                if results:
                    best = max(results, key=lambda r: float(getattr(r, "score", 0.0)))
                    template = ClusterTemplate.from_metadata_dict(dict(best.metadata))
                    template.match_similarity = float(best.score)
                    # Ensure cluster_id matches KyroDB doc_id (authoritative).
                    doc_id = getattr(best, "doc_id", None)
                    if doc_id is not None:
                        try:
                            template.cluster_id = int(doc_id)
                        except (TypeError, ValueError) as exc:
                            logger.warning(
                                "Failed to coerce cluster template doc_id to int",
                                extra={
                                    "customer_id": customer_id,
                                    "doc_id": doc_id,
                                    "error": str(exc),
                                },
                            )
                    return template
        except Exception as e:
            logger.warning(
                f"Cluster template search failed for customer {customer_id}: {e}. "
                "Falling back to in-memory centroid cache.",
                exc_info=True,
            )

        # Fallback: in-memory centroid cache matching (only if cache is populated).
        centroids = self._get_cached_centroids(customer_id)
        if not centroids:
            logger.debug(f"No clusters cached for {customer_id}")
            return None

        # Convert to numpy for vectorized ops
        query_vec = np.array(episode_embedding)

        # Compute cosine similarity to all centroids
        best_cluster_id = None
        best_similarity = 0.0

        for cluster_id, centroid in centroids.items():
            similarity = self._cosine_similarity(query_vec, centroid)

            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster_id = cluster_id

        # Check threshold
        if best_similarity < similarity_threshold:
            logger.debug(
                f"No cluster match for episode (best: {best_similarity:.2f} < {similarity_threshold:.2f})"
            )
            return None

        logger.info(
            f"Episode matches cluster {best_cluster_id} " f"(similarity: {best_similarity:.2f})"
        )
        if best_cluster_id is None:
            return None

        # Retrieve cluster template
        matched_template = await self._get_cluster_template(
            customer_id=customer_id, cluster_id=best_cluster_id
        )

        return matched_template

    def _calculate_avg_similarity(self, embeddings: np.ndarray) -> float:
        """
        Calculate average pairwise cosine similarity within cluster.

        Args:
            embeddings: NxD array of embeddings

        Returns:
            Average similarity (0.0-1.0)
        """
        if len(embeddings) < 2:
            return 1.0

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)

        # Compute pairwise similarities
        similarity_matrix = np.dot(normalized, normalized.T)

        # Get upper triangle (avoid diagonal and duplicates)
        n = len(embeddings)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        similarities = similarity_matrix[mask]

        return float(np.mean(similarities))

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Returns:
            Similarity in [0, 1] (higher = more similar)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _update_cache(self, customer_id: str, clusters: dict[int, ClusterInfo]) -> None:
        """
        Update cluster centroid cache (thread-safe).

        Args:
            customer_id: Customer ID
            clusters: Cluster info dict
        """
        with self._cache_lock:
            centroids_dict = {
                cluster_id: np.array(info.centroid_embedding)
                for cluster_id, info in clusters.items()
            }
            self._cluster_cache[customer_id] = (time.time(), centroids_dict)

        logger.debug(f"Updated cluster cache for {customer_id}: {len(clusters)} clusters")

    def _get_cached_centroids(self, customer_id: str) -> dict[int, np.ndarray]:
        """
        Get cached centroids for customer (thread-safe).

        Returns:
            Dict mapping cluster_id → centroid (numpy array)
        """
        with self._cache_lock:
            timestamp, centroids = self._cluster_cache.get(customer_id, (0, {}))

            # Check if cache is expired
            if time.time() - timestamp > self.cache_ttl:
                if customer_id in self._cluster_cache:
                    logger.debug(
                        f"Cluster cache expired for {customer_id} (age: {time.time() - timestamp:.0f}s)"
                    )
                    del self._cluster_cache[customer_id]
                return {}

            # Deep copy centroids to prevent cache corruption
            # Each numpy array is copied to avoid shared references
            return {cluster_id: centroid.copy() for cluster_id, centroid in centroids.items()}

    async def _fetch_active_episode_embeddings(
        self, customer_id: str, collection: str
    ) -> list[tuple[int, list[float]]]:
        """
        Fetch all active (non-archived) episode embeddings.

        Args:
            customer_id: Customer ID
            collection: Collection name

        Returns:
            List of (episode_id, embedding) tuples.

        Raises:
            RuntimeError: If episode_index is not configured.
        """
        if self.episode_index is None:
            raise RuntimeError(
                "EpisodeClusterer requires an episode index for full enumeration. "
                "Pass CustomerDatabase (or compatible) as episode_index."
            )

        # Enumerate active episodes from the local index.
        episode_ids = await self.episode_index.list_episode_ids(
            customer_id=customer_id,
            collection=collection,
            include_archived=False,
            include_deleted=False,
        )
        if not episode_ids:
            return []

        namespaced_collection = get_namespaced_collection(customer_id, collection)

        # Fetch embeddings from KyroDB in bounded chunks.
        chunk_size = 256
        max_in_flight = 8
        semaphore = asyncio.Semaphore(max_in_flight)

        async def _fetch_chunk(doc_ids: list[int]) -> list[tuple[int, list[float]]]:
            async with semaphore:
                response = await self.kyrodb_router.text_client.bulk_query(
                    doc_ids=doc_ids,
                    namespace=namespaced_collection,
                    include_embeddings=True,
                )
            items: list[tuple[int, list[float]]] = []
            for result in response.results:
                if not result.found:
                    continue
                embedding = list(getattr(result, "embedding", []) or [])
                if not embedding:
                    logger.warning(
                        "Episode missing embedding during clustering fetch",
                        extra={
                            "customer_id": customer_id,
                            "episode_id": int(getattr(result, "doc_id", 0) or 0),
                        },
                    )
                    continue
                items.append((int(result.doc_id), embedding))
            return items

        tasks: list[asyncio.Task[list[tuple[int, list[float]]]]] = []
        for i in range(0, len(episode_ids), chunk_size):
            chunk = episode_ids[i : i + chunk_size]
            tasks.append(asyncio.create_task(_fetch_chunk(chunk)))

        embeddings: list[tuple[int, list[float]]] = []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for item in results:
            if isinstance(item, Exception):
                raise item
            embeddings.extend(cast(list[tuple[int, list[float]]], item))

        return embeddings

    async def _persist_cluster_labels(
        self,
        customer_id: str,
        collection: str,
        episode_ids: list[int],
        cluster_labels: np.ndarray,
        *,
        label_to_cluster_id: dict[int, int],
    ) -> None:
        """
        Persist cluster labels to KyroDB metadata.

        Updates each episode's metadata with cluster_id.

        Args:
            customer_id: Customer ID
            collection: Collection name
            episode_ids: List of episode IDs
            cluster_labels: Corresponding cluster labels
        """
        if not episode_ids:
            return

        # Persist cluster labels to KyroDB metadata using UpdateMetadata RPC.
        # Convention: cluster_id == 0 means "unclustered/noise".
        namespaced_collection = get_namespaced_collection(customer_id, collection)

        max_in_flight = 32
        semaphore = asyncio.Semaphore(max_in_flight)

        async def _update(doc_id: int, cluster_id_value: int) -> None:
            async with semaphore:
                await self.kyrodb_router.text_client.update_metadata(
                    doc_id=doc_id,
                    namespace=namespaced_collection,
                    metadata={"cluster_id": str(cluster_id_value)},
                    merge=True,
                )

        tasks: list[asyncio.Task[None]] = []
        for i, episode_id in enumerate(episode_ids):
            label = int(cluster_labels[i])
            cluster_id_value = 0 if label == -1 else int(label_to_cluster_id[label])
            tasks.append(asyncio.create_task(_update(int(episode_id), cluster_id_value)))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures = [r for r in results if isinstance(r, Exception)]
        if failures:
            logger.warning(
                "Cluster label persistence had failures",
                extra={
                    "customer_id": customer_id,
                    "collection": collection,
                    "attempted_updates": len(tasks),
                    "failed_updates": len(failures),
                },
            )

    async def _get_cluster_template(
        self, customer_id: str, cluster_id: int
    ) -> ClusterTemplate | None:
        """
        Retrieve cluster template from database.

        Args:
            customer_id: Customer ID
            cluster_id: Cluster ID

        Returns:
            ClusterTemplate if exists, else None
        """
        try:
            return await self.kyrodb_router.get_cluster_template(
                customer_id=customer_id,
                cluster_id=cluster_id,
            )
        except Exception as e:
            logger.warning(
                f"Failed to fetch cluster template {cluster_id} for {customer_id}: {e}",
                exc_info=True,
            )
            return None
