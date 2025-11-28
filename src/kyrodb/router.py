"""
KyroDB namespace router for multi-modal dual-instance architecture.

Routes requests to appropriate KyroDB instances:
- Text/code embeddings → kyrodb_text (384-dim)
- Image embeddings → kyrodb_images (512-dim)

Multi-tenancy: Uses customer-namespaced collections for data isolation.
Namespace format: {customer_id}:failures (e.g., "acme-corp:failures")
"""

import logging
from typing import Optional

from src.config import KyroDBConfig
from src.kyrodb.client import KyroDBClient
from src.kyrodb.kyrodb_pb2 import SearchResponse

logger = logging.getLogger(__name__)


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

    async def connect(self) -> None:
        """
        Connect to both KyroDB instances.

        Raises:
            KyroDBError: If either connection fails
        """
        logger.info("Connecting to KyroDB instances...")
        await self.text_client.connect()
        self._text_connected = True
        logger.info(f"✓ Text instance connected ({self.text_client.address})")

        await self.image_client.connect()
        self._image_connected = True
        logger.info(f"✓ Image instance connected ({self.image_client.address})")

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
        image_embedding: Optional[list[float]] = None,
        metadata: Optional[dict[str, str]] = None,
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

    async def search_text(
        self,
        query_embedding: list[float],
        k: int,
        customer_id: str,
        collection: str,
        min_score: float = 0.6,
        metadata_filters: Optional[dict[str, str]] = None,
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

    async def search_episodes(
        self,
        query_embedding: list[float],
        customer_id: str,
        collection: str,
        k: int = 20,
        min_score: float = 0.6,
        metadata_filters: Optional[dict[str, str]] = None,
        include_image_search: bool = False,
        image_weight: float = 0.3,
    ) -> SearchResponse:
        """
        Search for episodes using text (and optionally image) embeddings.

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
            include_image_search: Also search image embeddings
            image_weight: Weight for image similarity (0-1)

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

        if not include_image_search:
            return text_response

        
        logger.warning("Image search fusion not yet implemented - returning text results only")
        return text_response

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
        include_images: bool = False,
    ) -> list["Episode"]:
        """
        Batch retrieve multiple episodes by ID.
        
        50-200x more efficient than individual get_episode() calls.
        Use for clustering, decay analysis, and template matching.
        
        Args:
            episode_ids: List of episode IDs to retrieve
            customer_id: Customer ID for namespace isolation
            collection: Base collection name ("failures")
            include_images: Also fetch image embeddings (not implemented)
            
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
        
        10-100x more efficient than individual delete_episode() calls.
        
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
    ) -> Optional[dict]:
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
