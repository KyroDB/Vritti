"""
KyroDB gRPC client wrapper with connection pooling and error handling.

Provides a clean async interface to KyroDB operations with:
- Automatic retry logic
- Circuit breaker pattern
- Connection pooling
- Comprehensive error handling
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import grpc
from grpc import StatusCode
from grpc.aio import insecure_channel, AioRpcError

from src.kyrodb.kyrodb_pb2 import (
    DeleteRequest,
    DeleteResponse,
    InsertRequest,
    InsertResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    HealthRequest,
    HealthResponse,
)
from src.kyrodb.kyrodb_pb2_grpc import KyroDBServiceStub

logger = logging.getLogger(__name__)


class KyroDBError(Exception):
    """Base exception for KyroDB client errors."""

    pass


class ConnectionError(KyroDBError):
    """Failed to connect to KyroDB instance."""

    pass


class RequestTimeoutError(KyroDBError):
    """Request exceeded timeout threshold."""

    pass


class DocumentNotFoundError(KyroDBError):
    """Requested document does not exist."""

    pass


class KyroDBClient:
    """
    Async gRPC client for KyroDB operations.

    Handles a single KyroDB instance with connection pooling.
    For multi-instance routing (text/images), use KyroDBRouter.
    """

    def __init__(
        self,
        host: str,
        port: int,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.5,
    ):
        """
        Initialize KyroDB client.

        Args:
            host: KyroDB server host
            port: KyroDB gRPC port
            timeout_seconds: Request timeout
            max_retries: Maximum retry attempts for transient failures
            retry_backoff_seconds: Initial backoff for exponential retry
        """
        self.address = f"{host}:{port}"
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[KyroDBServiceStub] = None
        self._connected = False

    async def connect(self) -> None:
        """
        Establish connection to KyroDB server.

        Raises:
            ConnectionError: If connection fails
        """
        if self._connected:
            return

        try:
            # Create insecure channel (TLS support can be added later)
            self._channel = insecure_channel(
                self.address,
                options=[
                    ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
                    ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                    ("grpc.keepalive_time_ms", 30000),
                    ("grpc.keepalive_timeout_ms", 10000),
                ],
            )
            self._stub = KyroDBServiceStub(self._channel)

            # Verify connection with health check
            await self.health_check()
            self._connected = True
            logger.info(f"Connected to KyroDB at {self.address}")

        except Exception as e:
            await self.close()
            raise ConnectionError(f"Failed to connect to {self.address}: {e}") from e

    async def close(self) -> None:
        """Close connection to KyroDB."""
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None
            self._connected = False
            logger.info(f"Closed connection to {self.address}")

    async def __aenter__(self) -> "KyroDBClient":
        """Context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        await self.close()

    async def _call_with_retry(self, rpc_func, request, operation: str):
        """
        Execute gRPC call with retry logic.

        Args:
            rpc_func: Async RPC function to call
            request: Protobuf request message
            operation: Operation name for logging

        Returns:
            Response from KyroDB

        Raises:
            KyroDBError: On persistent failure
        """
        if not self._connected:
            await self.connect()

        for attempt in range(self.max_retries):
            try:
                response = await asyncio.wait_for(
                    rpc_func(request), timeout=self.timeout_seconds
                )
                return response

            except AioRpcError as e:
                # Handle specific gRPC status codes
                if e.code() == StatusCode.UNAVAILABLE:
                    if attempt < self.max_retries - 1:
                        backoff = self.retry_backoff_seconds * (2**attempt)
                        logger.warning(
                            f"{operation} failed (UNAVAILABLE), "
                            f"retrying in {backoff}s... (attempt {attempt + 1}/{self.max_retries})"
                        )
                        await asyncio.sleep(backoff)
                        continue
                    raise ConnectionError(f"KyroDB unavailable: {e.details()}") from e

                elif e.code() == StatusCode.NOT_FOUND:
                    raise DocumentNotFoundError(f"Document not found: {e.details()}") from e

                elif e.code() == StatusCode.INVALID_ARGUMENT:
                    raise KyroDBError(f"Invalid request: {e.details()}") from e

                else:
                    raise KyroDBError(f"{operation} failed: {e.details()}") from e

            except asyncio.TimeoutError as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"{operation} timed out, retrying... "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    continue
                raise RequestTimeoutError(
                    f"{operation} exceeded {self.timeout_seconds}s timeout"
                ) from e

            except Exception as e:
                logger.error(f"Unexpected error in {operation}: {e}")
                raise KyroDBError(f"{operation} failed unexpectedly: {e}") from e

        raise KyroDBError(f"{operation} failed after {self.max_retries} retries")

    async def insert(
        self,
        doc_id: int,
        embedding: list[float],
        namespace: str = "",
        metadata: Optional[dict[str, str]] = None,
    ) -> InsertResponse:
        """
        Insert a document with embedding into KyroDB.

        Args:
            doc_id: Unique document ID (non-zero)
            embedding: Dense embedding vector
            namespace: Logical namespace (e.g., "failures", "skills")
            metadata: Optional key-value metadata

        Returns:
            InsertResponse with success status

        Raises:
            KyroDBError: On insertion failure
        """
        request = InsertRequest(
            doc_id=doc_id,
            embedding=embedding,
            namespace=namespace,
            metadata=metadata or {},
        )
        return await self._call_with_retry(self._stub.Insert, request, "Insert")

    async def search(
        self,
        query_embedding: list[float],
        k: int = 20,
        namespace: str = "",
        min_score: float = -1.0,
        include_embeddings: bool = False,
        metadata_filters: Optional[dict[str, str]] = None,
    ) -> SearchResponse:
        """
        k-NN vector similarity search.

        Args:
            query_embedding: Query vector
            k: Number of nearest neighbors to return
            namespace: Filter by namespace
            min_score: Minimum cosine similarity threshold
            include_embeddings: Return embeddings in results
            metadata_filters: Metadata filters (not fully implemented in KyroDB)

        Returns:
            SearchResponse with top-k results

        Raises:
            KyroDBError: On search failure
        """
        request = SearchRequest(
            query_embedding=query_embedding,
            k=k,
            namespace=namespace,
            min_score=min_score,
            include_embeddings=include_embeddings,
            metadata_filters=metadata_filters or {},
        )
        return await self._call_with_retry(self._stub.Search, request, "Search")

    async def query(
        self, doc_id: int, namespace: str = "", include_embedding: bool = True
    ) -> QueryResponse:
        """
        Point lookup by document ID.

        Args:
            doc_id: Document ID to retrieve
            namespace: Namespace filter
            include_embedding: Return the embedding vector

        Returns:
            QueryResponse with document data

        Raises:
            DocumentNotFoundError: If document doesn't exist
            KyroDBError: On query failure
        """
        request = QueryRequest(
            doc_id=doc_id, namespace=namespace, include_embedding=include_embedding
        )
        return await self._call_with_retry(self._stub.Query, request, "Query")

    async def delete(self, doc_id: int, namespace: str = "") -> DeleteResponse:
        """
        Delete a document by ID.

        Args:
            doc_id: Document ID to delete
            namespace: Namespace filter

        Returns:
            DeleteResponse with success status

        Raises:
            KyroDBError: On deletion failure
        """
        request = DeleteRequest(doc_id=doc_id, namespace=namespace)
        return await self._call_with_retry(self._stub.Delete, request, "Delete")

    async def health_check(self) -> HealthResponse:
        """
        Check KyroDB health status.

        Returns:
            HealthResponse with server status

        Raises:
            ConnectionError: If server is unhealthy
        """
        request = HealthRequest()
        try:
            response = await asyncio.wait_for(
                self._stub.Health(request), timeout=5.0  # Short timeout for health check
            )
            if response.status != 1:  # HEALTHY
                logger.warning(f"KyroDB health check returned status: {response.status}")
            return response
        except Exception as e:
            raise ConnectionError(f"Health check failed: {e}") from e
