"""
KyroDB namespace router for multi-modal dual-instance architecture.

Routes requests to appropriate KyroDB instances:
- Text/code embeddings → kyrodb_text (384-dim)
- Image embeddings → kyrodb_images (512-dim)

Uses namespace for logical collection separation within each instance.
"""

import logging
from typing import Optional

from src.config import KyroDBConfig
from src.kyrodb.client import KyroDBClient, KyroDBError
from src.kyrodb.kyrodb_pb2 import SearchResponse

logger = logging.getLogger(__name__)


class KyroDBRouter:
    """
    Routes requests to text and image KyroDB instances.

    Provides a unified interface for multi-modal episodic memory storage.
    """

    def __init__(self, config: KyroDBConfig):
        """
        Initialize router with dual KyroDB instances.

        Args:
            config: KyroDB configuration with text/image connection details
        """
        self.config = config

        # Initialize clients (lazy connection)
        self.text_client = KyroDBClient(
            host=config.text_host,
            port=config.text_port,
            timeout_seconds=config.request_timeout_seconds,
            max_retries=3,
        )

        self.image_client = KyroDBClient(
            host=config.image_host,
            port=config.image_port,
            timeout_seconds=config.request_timeout_seconds,
            max_retries=3,
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
        collection: str,
        text_embedding: list[float],
        image_embedding: Optional[list[float]] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> tuple[bool, bool]:
        """
        Insert episode into appropriate KyroDB instances.

        Args:
            episode_id: Unique episode ID (used as doc_id)
            collection: Namespace ("failures", "skills", "rules")
            text_embedding: Text/code embedding (384-dim)
            image_embedding: Optional image embedding (512-dim)
            metadata: Episode metadata (stored in both instances)

        Returns:
            tuple: (text_inserted, image_inserted)

        Raises:
            KyroDBError: If insertion fails critically
        """
        text_success = False
        image_success = False

        # Insert text embedding (required)
        try:
            response = await self.text_client.insert(
                doc_id=episode_id,
                embedding=text_embedding,
                namespace=collection,
                metadata=metadata,
            )
            text_success = response.success
            if not text_success:
                logger.error(
                    f"Text insertion failed for episode {episode_id}: {response.error}"
                )
        except Exception as e:
            logger.error(f"Text insertion error for episode {episode_id}: {e}")
            raise

        # Insert image embedding (optional)
        if image_embedding:
            try:
                response = await self.image_client.insert(
                    doc_id=episode_id,
                    embedding=image_embedding,
                    namespace=f"{collection}_images",  # Separate namespace
                    metadata=metadata,
                )
                image_success = response.success
                if not image_success:
                    logger.warning(
                        f"Image insertion failed for episode {episode_id}: {response.error}"
                    )
            except Exception as e:
                logger.warning(
                    f"Image insertion error for episode {episode_id}: {e} "
                    "(continuing with text-only)"
                )
                # Don't fail the whole operation if image insert fails
                image_success = False

        logger.debug(
            f"Episode {episode_id} inserted: text={text_success}, image={image_success}"
        )
        return (text_success, image_success)

    async def search_episodes(
        self,
        query_embedding: list[float],
        collection: str,
        k: int = 20,
        min_score: float = 0.6,
        metadata_filters: Optional[dict[str, str]] = None,
        include_image_search: bool = False,
        image_weight: float = 0.3,
    ) -> SearchResponse:
        """
        Search for episodes using text (and optionally image) embeddings.

        Args:
            query_embedding: Text query embedding
            collection: Namespace to search ("failures", "skills", "rules")
            k: Number of results to return
            min_score: Minimum similarity threshold
            metadata_filters: Optional metadata filters
            include_image_search: Also search image embeddings
            image_weight: Weight for image similarity (0-1)

        Returns:
            SearchResponse: Combined search results from text (and optionally images)

        Raises:
            KyroDBError: If search fails
        """
        # Primary search: text embeddings
        text_response = await self.text_client.search(
            query_embedding=query_embedding,
            k=k,
            namespace=collection,
            min_score=min_score,
            include_embeddings=False,  # Don't waste bandwidth
            metadata_filters=metadata_filters,
        )

        if not include_image_search:
            return text_response

        # TODO: Implement multi-modal fusion
        # For Phase 1, we'll just return text results
        # Phase 2 will implement proper score fusion:
        # combined_score = (1 - image_weight) * text_score + image_weight * image_score
        logger.warning("Image search fusion not yet implemented - returning text results only")
        return text_response

    async def delete_episode(
        self, episode_id: int, collection: str, delete_images: bool = True
    ) -> tuple[bool, bool]:
        """
        Delete episode from KyroDB instances.

        Args:
            episode_id: Episode ID to delete
            collection: Namespace
            delete_images: Also delete from image instance

        Returns:
            tuple: (text_deleted, image_deleted)
        """
        text_deleted = False
        image_deleted = False

        # Delete from text instance
        try:
            response = await self.text_client.delete(
                doc_id=episode_id, namespace=collection
            )
            text_deleted = response.success
        except Exception as e:
            logger.error(f"Text deletion failed for episode {episode_id}: {e}")

        # Delete from image instance
        if delete_images:
            try:
                response = await self.image_client.delete(
                    doc_id=episode_id, namespace=f"{collection}_images"
                )
                image_deleted = response.success
            except Exception as e:
                logger.warning(f"Image deletion failed for episode {episode_id}: {e}")
                image_deleted = False

        logger.debug(
            f"Episode {episode_id} deleted: text={text_deleted}, image={image_deleted}"
        )
        return (text_deleted, image_deleted)

    async def get_episode(
        self, episode_id: int, collection: str, include_image: bool = False
    ) -> Optional[dict]:
        """
        Retrieve episode by ID.

        Args:
            episode_id: Episode ID to retrieve
            collection: Namespace
            include_image: Also fetch image embedding

        Returns:
            dict with episode data, or None if not found

        Raises:
            KyroDBError: On retrieval failure
        """
        # Fetch from text instance
        try:
            response = await self.text_client.query(
                doc_id=episode_id, namespace=collection, include_embedding=False
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
                        namespace=f"{collection}_images",
                        include_embedding=False,
                    )
                    episode_data["image_found"] = image_response.found
                except Exception as e:
                    logger.warning(
                        f"Failed to fetch image for episode {episode_id}: {e}"
                    )
                    episode_data["image_found"] = False

            return episode_data

        except Exception as e:
            logger.error(f"Failed to retrieve episode {episode_id}: {e}")
            raise

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
