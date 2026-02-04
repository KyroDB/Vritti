import os

import pytest

from src.config import EmbeddingConfig
from src.ingestion.embedding import EmbeddingService


def test_embedding_offline_mode_default_is_opt_in() -> None:
    config = EmbeddingConfig()
    assert config.offline_mode is False


def test_embedding_service_does_not_force_offline_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    _service = EmbeddingService(EmbeddingConfig(offline_mode=False))

    assert os.getenv("HF_HUB_OFFLINE") is None
    assert os.getenv("TRANSFORMERS_OFFLINE") is None


def test_offline_preflight_fails_fast_with_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    service = EmbeddingService(
        EmbeddingConfig(
            offline_mode=True,
            text_model_name="all-MiniLM-L6-v2",
            image_model_name="openai/clip-vit-base-patch32",
        )
    )

    def _missing_model(*, model_name: str, namespace_hint: str) -> None:
        raise RuntimeError(f"missing:{model_name}:{namespace_hint}")

    monkeypatch.setattr(service, "_verify_model_cached_for_offline", _missing_model)

    with pytest.raises(RuntimeError) as exc:
        service.validate_offline_model_preflight()

    message = str(exc.value)
    assert "Embedding offline mode is enabled" in message
    assert "text model: all-MiniLM-L6-v2" in message
    assert "image model: openai/clip-vit-base-patch32" in message
    assert "EMBEDDING_OFFLINE_MODE=false" in message


def test_offline_preflight_noop_when_online_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")

    service = EmbeddingService(EmbeddingConfig(offline_mode=False))

    def _should_not_be_called(*, model_name: str, namespace_hint: str) -> None:
        raise AssertionError(
            f"cache check should not run in online mode: {model_name} ({namespace_hint})"
        )

    monkeypatch.setattr(service, "_verify_model_cached_for_offline", _should_not_be_called)
    service.validate_offline_model_preflight()
