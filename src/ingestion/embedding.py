"""
Multi-modal embedding service for text, code, and images.

Handles:
- Text/code embeddings via sentence-transformers (384-dim)
- Image embeddings via CLIP (512-dim)
- Model caching and batching for performance
"""

import logging
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from src.config import EmbeddingConfig

logger = logging.getLogger(__name__)


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
        self.config = config
        self._text_model: Optional[SentenceTransformer] = None
        self._clip_model: Optional[CLIPModel] = None
        self._clip_processor: Optional[CLIPProcessor] = None
        self._device = self._get_device()

        logger.info(f"EmbeddingService initialized (device: {self._device})")

    def _get_device(self) -> str:
        """
        Determine optimal device for inference.

        Returns:
            str: "cuda", "mps", or "cpu"
        """
        if not self.config.enable_gpu:
            return "cpu"

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

            logger.info(f"✓ Text model loaded ({actual_dim}-dim)")

        return self._text_model

    def _load_clip_models(self) -> tuple[CLIPModel, CLIPProcessor]:
        """
        Lazy-load CLIP model and processor.

        Returns:
            tuple: (CLIPModel, CLIPProcessor)
        """
        if self._clip_model is None or self._clip_processor is None:
            logger.info(f"Loading CLIP model: {self.config.image_model_name}")

            self._clip_processor = CLIPProcessor.from_pretrained(self.config.image_model_name)
            self._clip_model = CLIPModel.from_pretrained(self.config.image_model_name)
            self._clip_model.to(self._device)
            self._clip_model.eval()  # Inference mode

            logger.info(f"✓ CLIP model loaded (device: {self._device})")

        return self._clip_model, self._clip_processor

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

        model = self._load_text_model()

        # Generate embedding
        with torch.no_grad():
            embedding = model.encode(
                text,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True,  # L2 normalization for cosine similarity
            )

        # Convert to list
        return embedding.cpu().tolist()

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

        model = self._load_text_model()

        # Batch encode
        with torch.no_grad():
            embeddings = model.encode(
                texts,
                batch_size=self.config.text_batch_size,
                convert_to_tensor=True,
                show_progress_bar=len(texts) > 100,  # Show progress for large batches
                normalize_embeddings=True,
            )

        # Convert to list of lists
        return embeddings.cpu().tolist()

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

    def embed_image(self, image_path: Union[str, Path]) -> list[float]:
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

        clip_model, clip_processor = self._load_clip_models()

        # Preprocess image
        inputs = clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Generate embedding
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            # Normalize for cosine similarity
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # Convert to list
        embedding = image_features[0].cpu().tolist()

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

    def embed_images_batch(self, image_paths: list[Union[str, Path]]) -> list[list[float]]:
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

        clip_model, clip_processor = self._load_clip_models()

        # Batch process
        inputs = clip_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            # Normalize
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # Convert to list of lists
        return image_features.cpu().tolist()

    def warmup(self) -> None:
        """
        Warm up models by generating dummy embeddings.

        Call this at service startup to avoid cold start latency on first request.
        """
        logger.info("Warming up embedding models...")

        # Warm up text model
        self.embed_text("warmup text")
        logger.info("✓ Text model warmed up")

        # Warm up CLIP model (skip if no images will be used)
        # TODO: Make this configurable
        # self.embed_image("path/to/dummy/image.png")
        # logger.info("✓ CLIP model warmed up")

    def get_info(self) -> dict[str, any]:
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
