"""
Multi-modal embedding service for text, code, and images.

Handles:
- Text/code embeddings via sentence-transformers (384-dim)
- Image embeddings via CLIP (512-dim)
- Model caching and batching for performance
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cachetools import TTLCache
from PIL import Image

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from transformers import CLIPModel, CLIPProcessor

from src.config import EmbeddingConfig

os.environ.setdefault("PYTORCH_NO_SHARED_MEMORY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

logger = logging.getLogger(__name__)
_TORCH_SHARING_CONFIGURED = False
_TORCH_SHARING_LOCK = threading.Lock()


def _is_truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _configure_torch_sharing() -> None:
    """
    Configure torch multiprocessing sharing strategy to avoid leaked semaphore warnings.

    This is safe to call multiple times; configuration is applied once.
    """
    global _TORCH_SHARING_CONFIGURED
    if _TORCH_SHARING_CONFIGURED:
        return
    with _TORCH_SHARING_LOCK:
        if _TORCH_SHARING_CONFIGURED:
            return
        try:
            import multiprocessing as py_mp

            import torch.multiprocessing as torch_mp

            with contextlib.suppress(RuntimeError):
                py_mp.set_start_method("spawn", force=False)

            torch_mp.set_sharing_strategy("file_system")
            logger.info("Torch multiprocessing sharing strategy set to file_system")
        except Exception as e:
            logger.debug(f"Skipping torch sharing strategy config: {e}")
        finally:
            _TORCH_SHARING_CONFIGURED = True


class _AsyncTextEmbeddingBatcher:
    """Async batcher that coalesces concurrent text-embedding requests."""

    def __init__(
        self,
        embed_batch_fn: Callable[[list[str]], list[list[float]]],
        *,
        max_batch_size: int,
        max_wait_ms: int,
    ) -> None:
        self._embed_batch_fn = embed_batch_fn
        self._max_batch_size = max(1, int(max_batch_size))
        self._max_wait_s = max(0.0, float(max_wait_ms) / 1000.0)
        self._idle_shutdown_s = 2.0
        self._queue: asyncio.Queue[tuple[str, asyncio.Future[list[float]]] | None] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._start_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

    async def _ensure_worker(self) -> None:
        if self._worker_task and not self._worker_task.done():
            return
        async with self._start_lock:
            if self._worker_task and not self._worker_task.done():
                return
            self._worker_task = asyncio.create_task(self._run())

    async def embed(self, text: str) -> list[float]:
        await self._ensure_worker()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[float]] = loop.create_future()
        await self._queue.put((text, future))
        return await future

    async def shutdown(self) -> None:
        self._stop_event.set()
        await self._queue.put(None)
        if self._worker_task:
            await self._worker_task

    async def _run(self) -> None:
        while True:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=self._idle_shutdown_s)
            except asyncio.TimeoutError:
                if self._queue.empty() and not self._stop_event.is_set():
                    break
                continue
            if item is None:
                break
            if self._stop_event.is_set():
                _, future = item
                if not future.cancelled():
                    future.set_exception(RuntimeError("Embedding batcher shut down"))
                break
            text, future = item
            batch = [(text, future)]
            start = time.monotonic()
            shutdown_requested = False

            while len(batch) < self._max_batch_size:
                timeout = self._max_wait_s - (time.monotonic() - start)
                if timeout <= 0:
                    break
                try:
                    next_item = await asyncio.wait_for(self._queue.get(), timeout)
                except asyncio.TimeoutError:
                    break
                if next_item is None:
                    shutdown_requested = True
                    break
                batch.append(next_item)

            texts = [entry[0] for entry in batch]
            try:
                embeddings = await asyncio.to_thread(self._embed_batch_fn, texts)
                if len(embeddings) != len(batch):
                    raise ValueError(
                        "Text embedding batch size mismatch: " f"{len(embeddings)} != {len(batch)}"
                    )
            except Exception as exc:
                for _, fut in batch:
                    if not fut.cancelled():
                        fut.set_exception(exc)
                if shutdown_requested:
                    break
                continue

            for embedding, (_, fut) in zip(embeddings, batch, strict=False):
                if not fut.cancelled():
                    fut.set_result(embedding)

            if shutdown_requested or self._stop_event.is_set():
                break

        # Drain remaining queued items and cancel their futures.
        while not self._queue.empty():
            leftover = self._queue.get_nowait()
            if leftover is None:
                continue
            _, fut = leftover
            if not fut.cancelled():
                fut.set_exception(RuntimeError("Embedding batcher shut down"))


class EmbeddingService:
    """
    Singleton service for generating embeddings across modalities.

    Caches models in memory for performance.
    Thread-safe for concurrent requests.
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize embedding models.

        Args:
            config: Embedding configuration
        """
        _configure_torch_sharing()
        self.config = config
        self._apply_hf_runtime_policy()
        self._text_model: SentenceTransformer | None = None
        self._clip_model: CLIPModel | None = None
        self._clip_processor: CLIPProcessor | None = None
        self._device = self._get_device()
        self._text_lock = threading.Lock()
        self._clip_lock = threading.Lock()
        self._text_batcher: _AsyncTextEmbeddingBatcher | None = None
        self._text_batcher_lock = threading.Lock()
        self._text_cache_lock = threading.Lock()
        self._text_cache: TTLCache[str, list[float]] | None = None

        if self.config.text_cache_size > 0 and self.config.text_cache_ttl_seconds > 0:
            self._text_cache = TTLCache(
                maxsize=self.config.text_cache_size,
                ttl=self.config.text_cache_ttl_seconds,
            )

        logger.info(f"EmbeddingService initialized (device: {self._device})")

    def _apply_hf_runtime_policy(self) -> None:
        """
        Apply Hugging Face runtime policy.

        Offline mode is opt-in via `EMBEDDING_OFFLINE_MODE=true`. We do NOT force
        offline mode by default because first-time model bootstrap requires network.
        """
        if self.config.offline_mode:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            logger.warning(
                "Embedding offline mode enabled (EMBEDDING_OFFLINE_MODE=true). "
                "Model downloads are disabled."
            )

    def _is_hf_offline_mode_active(self) -> bool:
        return (
            self.config.offline_mode
            or _is_truthy_env(os.getenv("HF_HUB_OFFLINE"))
            or _is_truthy_env(os.getenv("TRANSFORMERS_OFFLINE"))
        )

    @staticmethod
    def _resolve_repo_candidates(model_name: str, namespace_hint: str) -> list[str]:
        candidates: list[str] = [model_name]
        if "/" not in model_name:
            candidates.append(f"{namespace_hint}/{model_name}")
        return candidates

    def _verify_model_cached_for_offline(
        self,
        *,
        model_name: str,
        namespace_hint: str,
    ) -> None:
        local_path = Path(model_name).expanduser()
        if local_path.exists():
            return

        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "Offline preflight requires `huggingface_hub` to verify local model cache."
            ) from e

        for repo_id in self._resolve_repo_candidates(model_name, namespace_hint):
            try:
                snapshot_download(repo_id=repo_id, local_files_only=True)
                return
            except Exception:
                continue

        raise RuntimeError(f"Model cache missing for '{model_name}'")

    def validate_offline_model_preflight(self) -> None:
        """
        Fail fast in offline mode if required models are not preloaded.

        This is intended for startup preflight in air-gapped/enterprise deployments.
        """
        if not self._is_hf_offline_mode_active():
            return

        missing: list[str] = []
        try:
            self._verify_model_cached_for_offline(
                model_name=self.config.text_model_name,
                namespace_hint="sentence-transformers",
            )
        except RuntimeError:
            missing.append(f"text model: {self.config.text_model_name}")

        try:
            self._verify_model_cached_for_offline(
                model_name=self.config.image_model_name,
                namespace_hint="openai",
            )
        except RuntimeError:
            missing.append(f"image model: {self.config.image_model_name}")

        if missing:
            missing_lines = "\n".join(f"- {item}" for item in missing)
            raise RuntimeError(
                "Embedding offline mode is enabled, but required models are not present in local cache.\n"
                f"{missing_lines}\n"
                "Preload before startup, for example:\n"
                f"  python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{self.config.text_model_name}')\"\n"
                f"  python -c \"from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('{self.config.image_model_name}'); CLIPProcessor.from_pretrained('{self.config.image_model_name}')\"\n"
                "Or disable offline mode with EMBEDDING_OFFLINE_MODE=false."
            )

    def _get_cached_text_embedding(self, text: str) -> list[float] | None:
        if not self._text_cache:
            return None
        with self._text_cache_lock:
            return self._text_cache.get(text)

    def _set_cached_text_embedding(self, text: str, embedding: list[float]) -> None:
        if not self._text_cache:
            return
        with self._text_cache_lock:
            self._text_cache[text] = embedding

    def _get_text_batcher(self) -> _AsyncTextEmbeddingBatcher:
        if self._text_batcher is None:
            with self._text_batcher_lock:
                if self._text_batcher is None:
                    self._text_batcher = _AsyncTextEmbeddingBatcher(
                        self.embed_texts_batch,
                        max_batch_size=self.config.text_batch_size,
                        max_wait_ms=self.config.text_batcher_max_wait_ms,
                    )
        return self._text_batcher

    async def shutdown(self) -> None:
        batcher = None
        with self._text_batcher_lock:
            batcher = self._text_batcher
            self._text_batcher = None
        if batcher is not None:
            await batcher.shutdown()

    def _embed_text_direct(self, text: str) -> list[float]:
        # SentenceTransformer encode is not guaranteed to be thread-safe across concurrent calls.
        # Guard both lazy init and inference to avoid crashes under load.
        with self._text_lock:
            model = self._load_text_model()

            import torch

            with torch.no_grad():
                embedding = model.encode(
                    text,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,  # L2 normalization for cosine similarity
                )

        # Convert to list
        return [float(x) for x in embedding.cpu().tolist()]

    def _get_device(self) -> str:
        """
        Determine optimal device for inference.

        Returns:
            str: "cuda", "mps", or "cpu"
        """
        if not self.config.enable_gpu:
            return "cpu"

        import torch

        if self.config.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif self.config.device == "mps" and torch.backends.mps.is_available():
            return "mps"
        else:
            logger.warning(f"Requested device '{self.config.device}' unavailable, using CPU")
            return "cpu"

    def _load_text_model(self) -> SentenceTransformer:
        """
        Lazy-load text embedding model.

        Returns:
            SentenceTransformer: Cached model instance
        """
        if self._text_model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading text model: {self.config.text_model_name}")
            self._text_model = SentenceTransformer(self.config.text_model_name, device=self._device)
            actual_dim = self._text_model.get_sentence_embedding_dimension()

            if actual_dim != self.config.text_dimension:
                logger.error(
                    f"Text model dimension mismatch! "
                    f"Config: {self.config.text_dimension}, "
                    f"Actual: {actual_dim}"
                )
                raise ValueError(
                    f"Text embedding dimension mismatch: "
                    f"expected {self.config.text_dimension}, got {actual_dim}"
                )

            logger.info(f"Text model loaded ({actual_dim}-dim)")

        return self._text_model

    def _load_clip_models(self) -> tuple[CLIPModel, CLIPProcessor]:
        """
        Lazy-load CLIP model and processor.

        Returns:
            tuple: (CLIPModel, CLIPProcessor)
        """
        if self._clip_model is None or self._clip_processor is None:
            from transformers import CLIPModel, CLIPProcessor

            logger.info(f"Loading CLIP model: {self.config.image_model_name}")

            clip_processor = CLIPProcessor.from_pretrained(self.config.image_model_name)
            clip_model = CLIPModel.from_pretrained(self.config.image_model_name)
            clip_model.to(self._device)
            clip_model.eval()  # Inference mode

            self._clip_processor = clip_processor
            self._clip_model = clip_model

            logger.info(f"CLIP model loaded (device: {self._device})")

        clip_model = self._clip_model
        clip_processor = self._clip_processor
        assert clip_model is not None
        assert clip_processor is not None
        return clip_model, clip_processor

    def embed_text(self, text: str) -> list[float]:
        """
        Generate text embedding.

        Args:
            text: Input text (goal, error message, reflection, etc.)

        Returns:
            list[float]: Embedding vector (384-dim by default)

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        cached = self._get_cached_text_embedding(text)
        if cached is not None:
            return cached

        embedding = self._embed_text_direct(text)
        self._set_cached_text_embedding(text, embedding)
        return embedding

    async def embed_text_async(self, text: str) -> list[float]:
        """
        Async wrapper for text embeddings (batch-aware).
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        cached = self._get_cached_text_embedding(text)
        if cached is not None:
            return cached

        if self.config.text_batch_size > 1:
            embedding = await self._get_text_batcher().embed(text)
            self._set_cached_text_embedding(text, embedding)
            return embedding

        embedding = await asyncio.to_thread(self._embed_text_direct, text)
        self._set_cached_text_embedding(text, embedding)
        return embedding

    def embed_texts_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts (batched for performance).

        Args:
            texts: List of input texts

        Returns:
            list[list[float]]: List of embedding vectors

        Raises:
            ValueError: If any text is empty
        """
        if not texts:
            return []

        # Validate all texts
        if any(not t or not t.strip() for t in texts):
            raise ValueError("Cannot embed empty texts in batch")

        cached_embeddings: list[list[float] | None] = []
        missing_texts: list[str] = []
        missing_indices: list[int] = []

        for idx, text in enumerate(texts):
            cached = self._get_cached_text_embedding(text)
            if cached is None:
                cached_embeddings.append(None)
                missing_texts.append(text)
                missing_indices.append(idx)
            else:
                cached_embeddings.append(cached)

        if missing_texts:
            with self._text_lock:
                model = self._load_text_model()

                import torch

                with torch.no_grad():
                    embeddings = model.encode(
                        missing_texts,
                        batch_size=self.config.text_batch_size,
                        convert_to_tensor=True,
                        show_progress_bar=len(texts) > 100,  # Show progress for large batches
                        normalize_embeddings=True,
                    )

            new_embeddings = embeddings.cpu().tolist()
            if len(new_embeddings) != len(missing_indices):
                raise ValueError(
                    "Text embedding batch size mismatch: "
                    f"{len(new_embeddings)} != {len(missing_indices)}"
                )
            for idx, embedding in zip(missing_indices, new_embeddings, strict=False):
                cached_embeddings[idx] = embedding
                self._set_cached_text_embedding(texts[idx], embedding)
        results: list[list[float]] = []
        for embedding in cached_embeddings:
            if embedding is None:
                raise ValueError("Text embedding batch missing result")
            results.append(embedding)
        return results

    def embed_code(self, code: str) -> list[float]:
        """
        Generate embedding for code diff.

        Uses same text model as goal embeddings (code is just structured text).

        Args:
            code: Code snippet or diff

        Returns:
            list[float]: Embedding vector
        """
        # Code is treated as text (sentence-transformers handles it well)
        return self.embed_text(code)

    def embed_image(self, image_path: str | Path) -> list[float]:
        """
        Generate image embedding using CLIP.

        Args:
            image_path: Path to image file

        Returns:
            list[float]: Image embedding (512-dim)

        Raises:
            FileNotFoundError: If image doesn't exist
            ValueError: If image cannot be loaded
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}") from e

        with self._clip_lock:
            clip_model, clip_processor = self._load_clip_models()

            inputs = clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            import torch

            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            embedding = [float(x) for x in image_features[0].cpu().tolist()]

        # Validate dimension
        if len(embedding) != self.config.image_dimension:
            logger.error(
                f"Image embedding dimension mismatch! "
                f"Expected: {self.config.image_dimension}, Got: {len(embedding)}"
            )
            raise ValueError(
                f"Image embedding dimension mismatch: "
                f"expected {self.config.image_dimension}, got {len(embedding)}"
            )

        return embedding

    def embed_image_bytes(self, image_bytes: bytes) -> list[float]:
        """
        Generate image embedding from raw image bytes using CLIP.

        Args:
            image_bytes: Raw image bytes

        Returns:
            list[float]: Image embedding (512-dim)

        Raises:
            ValueError: If bytes cannot be decoded into an image
        """
        if not image_bytes:
            raise ValueError("Image bytes are empty")

        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to decode image bytes: {e}") from e

        with self._clip_lock:
            clip_model, clip_processor = self._load_clip_models()

            inputs = clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            import torch

            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            embedding = [float(x) for x in image_features[0].cpu().tolist()]

        if len(embedding) != self.config.image_dimension:
            logger.error(
                f"Image embedding dimension mismatch! "
                f"Expected: {self.config.image_dimension}, Got: {len(embedding)}"
            )
            raise ValueError(
                f"Image embedding dimension mismatch: "
                f"expected {self.config.image_dimension}, got {len(embedding)}"
            )

        return embedding

    async def embed_image_bytes_async(self, image_bytes: bytes) -> list[float]:
        """Async wrapper for image embeddings."""
        embedding = await asyncio.to_thread(self.embed_image_bytes, image_bytes)
        return embedding

    def embed_images_batch(self, image_paths: list[str | Path]) -> list[list[float]]:
        """
        Generate embeddings for multiple images (batched).

        Args:
            image_paths: List of image file paths

        Returns:
            list[list[float]]: List of image embeddings

        Raises:
            FileNotFoundError: If any image doesn't exist
            ValueError: If any image cannot be loaded
        """
        if not image_paths:
            return []

        # Load all images
        images = []
        for path in image_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            try:
                images.append(Image.open(path).convert("RGB"))
            except Exception as e:
                raise ValueError(f"Failed to load image {path}: {e}") from e

        with self._clip_lock:
            clip_model, clip_processor = self._load_clip_models()

            inputs = clip_processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            import torch

            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            feature_dim = int(image_features.shape[-1])
            if feature_dim != self.config.image_dimension:
                raise ValueError(
                    "Image embedding dimension mismatch. "
                    f"Expected: {self.config.image_dimension}, Got: {feature_dim}"
                )

            return [[float(v) for v in row] for row in image_features.cpu().tolist()]

    def warmup(self) -> None:
        """
        Warm up models by generating dummy embeddings.

        Call this at service startup to avoid cold start latency on first request.
        """
        logger.info("Warming up embedding models...")

        self.embed_text("warmup text")
        logger.info("Text model warmed up")

    def get_info(self) -> dict[str, Any]:
        """
        Get information about loaded models.

        Returns:
            dict: Model info (names, dimensions, device)
        """
        return {
            "text_model": self.config.text_model_name,
            "text_dimension": self.config.text_dimension,
            "image_model": self.config.image_model_name,
            "image_dimension": self.config.image_dimension,
            "device": self._device,
            "text_model_loaded": self._text_model is not None,
            "clip_model_loaded": self._clip_model is not None,
        }
