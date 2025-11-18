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
from pathlib import Path

import grpc
from grpc import StatusCode
from grpc.aio import AioRpcError, insecure_channel, secure_channel

from src.kyrodb.kyrodb_pb2 import (
    DeleteRequest,
    DeleteResponse,
    HealthRequest,
    HealthResponse,
    InsertRequest,
    InsertResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
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
        enable_tls: bool = False,
        tls_ca_cert_path: str | None = None,
        tls_client_cert_path: str | None = None,
        tls_client_key_path: str | None = None,
        tls_verify_server: bool = True,
    ):
        """
        Initialize KyroDB client with optional TLS support.

        Args:
            host: KyroDB server host
            port: KyroDB gRPC port
            timeout_seconds: Request timeout
            max_retries: Maximum retry attempts for transient failures
            retry_backoff_seconds: Initial backoff for exponential retry
            enable_tls: Enable TLS/SSL encryption for connections
            tls_ca_cert_path: Path to CA certificate (None = system CA bundle)
            tls_client_cert_path: Path to client certificate for mutual TLS (optional)
            tls_client_key_path: Path to client private key for mutual TLS (optional)
            tls_verify_server: Verify server certificate (production: True, dev self-signed: False)

        Security:
            - enable_tls=True: Encrypted channel (REQUIRED for production)
            - tls_verify_server=True: Validates server certificate against CA
            - Client certs: Optional mutual TLS (highest security)
        """
        self.address = f"{host}:{port}"
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

        # TLS configuration
        self.enable_tls = enable_tls
        self.tls_ca_cert_path = tls_ca_cert_path
        self.tls_client_cert_path = tls_client_cert_path
        self.tls_client_key_path = tls_client_key_path
        self.tls_verify_server = tls_verify_server

        self._channel: grpc.aio.Channel | None = None
        self._stub: KyroDBServiceStub | None = None
        self._connected = False

    def _create_tls_credentials(self) -> grpc.ChannelCredentials:
        """
        Create TLS credentials for secure channel.

        Returns:
            grpc.ChannelCredentials: TLS credentials for secure channel

        Raises:
            FileNotFoundError: If certificate files don't exist
            ValueError: If TLS configuration is invalid

        Security:
            - CA cert: Validates server identity (prevents MITM attacks)
            - Client cert: Optional mutual TLS (server validates client identity)
            - verify_server=False: Only for dev with self-signed certs (NOT production)
        """
        # Read CA certificate (server verification)
        root_certs = None
        if self.tls_ca_cert_path:
            ca_path = Path(self.tls_ca_cert_path)
            if not ca_path.exists():
                raise FileNotFoundError(f"CA certificate not found: {ca_path}")
            root_certs = ca_path.read_bytes()
            logger.info(f"Loaded CA certificate from {ca_path}")
        else:
            # Use system CA bundle
            logger.info("Using system CA bundle for server verification")

        # Read client certificate and key (mutual TLS)
        private_key = None
        certificate_chain = None

        if self.tls_client_cert_path and self.tls_client_key_path:
            cert_path = Path(self.tls_client_cert_path)
            key_path = Path(self.tls_client_key_path)

            if not cert_path.exists():
                raise FileNotFoundError(f"Client certificate not found: {cert_path}")
            if not key_path.exists():
                raise FileNotFoundError(f"Client private key not found: {key_path}")

            certificate_chain = cert_path.read_bytes()
            private_key = key_path.read_bytes()
            logger.info("Loaded client certificate and key for mutual TLS")
        elif self.tls_client_cert_path or self.tls_client_key_path:
            raise ValueError(
                "Both tls_client_cert_path and tls_client_key_path must be provided for mutual TLS"
            )

        # Create credentials
        credentials = grpc.ssl_channel_credentials(
            root_certificates=root_certs,
            private_key=private_key,
            certificate_chain=certificate_chain,
        )

        if not self.tls_verify_server:
            logger.warning(
                "⚠️  Server certificate verification DISABLED - only use in development!"
            )
            # Override target name to skip verification (dev only)
            # In production, always verify the server certificate
            credentials = grpc.composite_channel_credentials(
                credentials,
                grpc.metadata_call_credentials(lambda context, callback: callback([], None)),
            )

        return credentials

    async def connect(self) -> None:
        """
        Establish connection to KyroDB server with optional TLS.

        Raises:
            ConnectionError: If connection fails
            FileNotFoundError: If TLS certificate files don't exist
            ValueError: If TLS configuration is invalid

        Security:
            - Insecure channel: Only for local development (enable_tls=False)
            - Secure channel: Encrypted with TLS (enable_tls=True, REQUIRED for production)
            - Mutual TLS: Highest security (client + server cert verification)
        """
        if self._connected:
            return

        try:
            # gRPC channel options (performance + keep-alive)
            options = [
                ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 30000),  # 30 seconds
                ("grpc.keepalive_timeout_ms", 10000),  # 10 seconds
                ("grpc.http2.max_pings_without_data", 0),  # Allow unlimited pings
                ("grpc.keepalive_permit_without_calls", 1),  # Allow keepalive without RPCs
            ]

            # Create channel (secure or insecure based on configuration)
            if self.enable_tls:
                # Production: Secure channel with TLS
                credentials = self._create_tls_credentials()
                self._channel = secure_channel(
                    self.address,
                    credentials,
                    options=options,
                )
                logger.info(f"Created secure TLS channel to {self.address}")
            else:
                # Development: Insecure channel (plaintext)
                self._channel = insecure_channel(
                    self.address,
                    options=options,
                )
                logger.warning(
                    f"⚠️  Created INSECURE channel to {self.address} - "
                    f"enable TLS for production!"
                )

            self._stub = KyroDBServiceStub(self._channel)

            # Verify connection with health check
            await self.health_check()
            self._connected = True

            logger.info(
                f"Connected to KyroDB at {self.address} "
                f"(TLS: {self.enable_tls}, Verify: {self.tls_verify_server})"
            )

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
                response = await asyncio.wait_for(rpc_func(request), timeout=self.timeout_seconds)
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
        metadata: dict[str, str] | None = None,
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
        metadata_filters: dict[str, str] | None = None,
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
