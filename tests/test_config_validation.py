"""
Unit tests for config validation functionality.

Tests the validate_configuration method to ensure logging works correctly
in all code paths.
"""

import logging
from unittest.mock import patch

import pytest

from src.config import (
    EmbeddingConfig,
    KyroDBConfig,
    LLMConfig,
    Settings,
)


@pytest.fixture
def settings_with_api_key() -> Settings:
    """Create settings with LLM API key configured."""
    return Settings(
        kyrodb=KyroDBConfig(),
        embedding=EmbeddingConfig(
            text_dimension=200,  # Non-standard to trigger warning
            image_dimension=768,  # Standard
        ),
        llm=LLMConfig(
            openrouter_api_key="sk-test-key-12345"  # Valid key
        ),
    )


@pytest.fixture
def settings_without_api_key() -> Settings:
    """Create settings without LLM API key."""
    return Settings(
        kyrodb=KyroDBConfig(),
        embedding=EmbeddingConfig(
            text_dimension=200,  # Non-standard to trigger warning
            image_dimension=768,  # Standard
        ),
        llm=LLMConfig(
            openrouter_api_key=""  # No key
        ),
    )


@pytest.fixture
def settings_all_nonstandard() -> Settings:
    """Create settings with all non-standard dimensions."""
    return Settings(
        kyrodb=KyroDBConfig(),
        embedding=EmbeddingConfig(
            text_dimension=200,  # Non-standard
            image_dimension=600,  # Non-standard
        ),
        llm=LLMConfig(
            openrouter_api_key=""  # No key
        ),
    )


def test_validate_configuration_with_api_key(settings_with_api_key, caplog):
    """
    Test validate_configuration when API key is present.
    
    This should only trigger warnings for non-standard embedding dimensions,
    not for missing API key.
    """
    with caplog.at_level(logging.WARNING):
        settings_with_api_key.validate_configuration()
    
    # Should have warning for non-standard text dimension
    assert any("Non-standard text embedding dimension: 200" in record.message 
               for record in caplog.records)
    
    # Should NOT have warning for missing API key
    assert not any("LLM API key not configured" in record.message 
                   for record in caplog.records)


def test_validate_configuration_without_api_key(settings_without_api_key, caplog):
    """
    Test validate_configuration when API key is missing.
    
    This is the critical test case for the bug fix - logging must be imported
    at the method level, not inside the first if block.
    """
    with caplog.at_level(logging.WARNING):
        settings_without_api_key.validate_configuration()
    
    # Should have warning for missing API key
    assert any("LLM API key not configured" in record.message 
               for record in caplog.records)
    
    # Should have warning for non-standard text dimension
    # This is where the bug would occur - logging would not be imported
    assert any("Non-standard text embedding dimension: 200" in record.message 
               for record in caplog.records)


def test_validate_configuration_all_warnings(settings_all_nonstandard, caplog):
    """
    Test validate_configuration with all warnings triggered.
    
    Verifies that logging works correctly for all warning conditions
    even when the first condition (missing API key) is True.
    """
    with caplog.at_level(logging.WARNING):
        settings_all_nonstandard.validate_configuration()
    
    # Should have all warnings
    assert any("LLM API key not configured" in record.message 
               for record in caplog.records)
    assert any("Non-standard text embedding dimension: 200" in record.message 
               for record in caplog.records)
    assert any("Non-standard image embedding dimension: 600" in record.message 
               for record in caplog.records)


def test_validate_configuration_standard_dimensions():
    """
    Test validate_configuration with standard dimensions.
    
    Should only warn about missing API key, not dimensions.
    """
    settings = Settings(
        kyrodb=KyroDBConfig(),
        embedding=EmbeddingConfig(
            text_dimension=384,  # Standard
            image_dimension=512,  # Standard
        ),
        llm=LLMConfig(
            openrouter_api_key=""  # No key
        ),
    )
    
    with patch('logging.warning') as mock_warning:
        settings.validate_configuration()
        
        # Should only be called once for API key
        assert mock_warning.call_count == 1
        assert "LLM API key not configured" in str(mock_warning.call_args)


def test_validate_configuration_no_warnings():
    """
    Test validate_configuration with valid configuration.
    
    Should not log any warnings.
    """
    settings = Settings(
        kyrodb=KyroDBConfig(),
        embedding=EmbeddingConfig(
            text_dimension=384,  # Standard
            image_dimension=512,  # Standard
        ),
        llm=LLMConfig(
            openrouter_api_key="sk-test-valid-key-12345"  # Valid key
        ),
    )
    
    with patch('logging.warning') as mock_warning:
        settings.validate_configuration()
        
        # Should not be called at all
        assert mock_warning.call_count == 0
