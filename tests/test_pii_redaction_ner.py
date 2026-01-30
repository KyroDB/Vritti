"""
Tests for NER-based PII redaction 
"""

import pytest
from src.utils.pii_redaction import AdvancedPIIRedactor, redact_all

# Skip tests if Presidio not installed or model not found
try:
    import presidio_analyzer
    import spacy
    nlp = spacy.load("en_core_web_lg")
    HAS_DEPENDENCIES = True
except Exception:
    HAS_DEPENDENCIES = False


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Presidio/Spacy dependencies missing")
class TestAdvancedPIIRedaction:
    """Test suite for NER-based redaction."""

    def test_redactor_initialization(self):
        """Test that redactor initializes correctly."""
        redactor = AdvancedPIIRedactor()
        assert redactor.analyzer is not None
        assert redactor.anonymizer is not None

    def test_ner_person_redaction(self):
        """Test redaction of names."""
        redactor = AdvancedPIIRedactor()
        # Use a standard phone format that Presidio definitely recognizes
        text = "Contact John Doe at 212-555-0123."
        redacted = redactor.redact(text)
        
        # Should redact name. Phone might be redacted depending on model confidence,
        # but our primary goal here is testing the Name redaction which Regex misses.
        assert "John Doe" not in redacted
        assert "[PERSON]" in redacted or "[PER]" in redacted

    def test_ner_location_redaction(self):
        """Test redaction of locations."""
        redactor = AdvancedPIIRedactor()
        text = "Server located in New York City near Central Park."
        redacted = redactor.redact(text)
        
        assert "New York City" not in redacted
        assert "[LOCATION]" in redacted or "[LOC]" in redacted

    def test_hybrid_redaction(self):
        """Test combined Regex + NER redaction via redact_all."""
        # Text with API key (Regex) and Name (NER)
        text = (
            "User Alice Smith committed API key "
            "sk-abcdef1234567890abcdef1234567890 "
            "to the repository."
        )
        
        redacted = redact_all(text, use_ner=True)
        
        # Regex should catch API key
        assert "sk-abcdef" not in redacted
        assert "[API_KEY]" in redacted
        
        # NER should catch Name
        assert "Alice Smith" not in redacted
        assert "[PERSON]" in redacted

    def test_ner_fallback_graceful(self):
        """Test that system falls back gracefully if NER fails/disabled."""
        text = "Email me at test@example.com"
        
        # Disable NER
        redacted = redact_all(text, use_ner=False)
        assert "test@example.com" not in redacted
        assert "[EMAIL]" in redacted

    def test_complex_trace_redaction(self):
        """Test redaction on a realistic error trace."""
        trace = """
        Error: Connection failed to database.
        User: admin_user (John Admin)
        Host: 192.168.1.55
        Location: AWS data center in Virginia
        Key: AKIAIOSFODNN7EXAMPLE
        """
        
        redacted = redact_all(trace, use_ner=True)
        
        assert "192.168.1.55" not in redacted  # Regex IP
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted  # Regex AWS Key
        assert "John Admin" not in redacted  # NER Person
        # "Virginia" should be detected as location
        assert "Virginia" not in redacted
        assert "[LOCATION]" in redacted or "[LOC]" in redacted
