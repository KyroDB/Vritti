"""
PII (Personally Identifiable Information) redaction utilities.

Removes sensitive data from error traces, logs, and code diffs before storage.
Uses regex patterns optimized for performance with compilation caching.
"""

import logging
import re
from re import Match, Pattern
from typing import Optional

logger = logging.getLogger(__name__)

# Compile patterns once at module load for performance
_EMAIL_PATTERN: Pattern = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", re.IGNORECASE
)

_IPV4_PATTERN: Pattern = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
)

_IPV6_PATTERN: Pattern = re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b")

# API keys, tokens, secrets (common patterns)
_API_KEY_PATTERNS: list[Pattern] = [
    re.compile(r'(?i)api[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?'),
    re.compile(r'(?i)secret[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?'),
    re.compile(r'(?i)access[_-]?token["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-\.]{20,})["\']?'),
    re.compile(r"(?i)bearer\s+([a-zA-Z0-9_\-\.]{20,})"),
    re.compile(r"(?i)authorization:\s*bearer\s+([a-zA-Z0-9_\-\.]{20,})"),
    # AWS
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS Access Key
    re.compile(
        r'(?i)aws[_-]?secret[_-]?access[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9/+=]{40})["\']?'
    ),
    # GitHub
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),  # GitHub Personal Access Token
    re.compile(r"gho_[a-zA-Z0-9]{36}"),  # GitHub OAuth Token
    re.compile(r"github_pat_[a-zA-Z0-9]{22}_[a-zA-Z0-9]{59}"),
    # OpenAI - match both standalone and in "API key: sk-..." format
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),  # OpenAI API Key (any length >=20)
]

# URLs with auth credentials
_URL_WITH_AUTH_PATTERN: Pattern = re.compile(
    r"(https?://)[^:@\s]+:[^:@\s]+@([^\s]+)", re.IGNORECASE
)

# Private keys (PEM format)
_PRIVATE_KEY_PATTERN: Pattern = re.compile(
    r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |DSA )?PRIVATE KEY-----",
    re.MULTILINE,
)

# SSH keys
_SSH_KEY_PATTERN: Pattern = re.compile(r"ssh-(?:rsa|dss|ed25519)\s+[A-Za-z0-9+/]+={0,2}(?:\s+\S+)?")

# Credit card numbers (basic pattern)
_CREDIT_CARD_PATTERN: Pattern = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")

# Social Security Numbers (US)
_SSN_PATTERN: Pattern = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Phone numbers (international format)
_PHONE_PATTERN: Pattern = re.compile(
    r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)

# Filesystem paths that might contain usernames
_HOME_PATH_PATTERN: Pattern = re.compile(r"/(?:home|Users)/([^/\s]+)", re.IGNORECASE)


def redact_email(text: str, replacement: str = "[EMAIL]") -> str:
    """
    Redact email addresses.

    Args:
        text: Input text
        replacement: Replacement string

    Returns:
        str: Text with emails redacted
    """
    return _EMAIL_PATTERN.sub(replacement, text)


def redact_ip_addresses(text: str, replacement: str = "[IP_ADDRESS]") -> str:
    """
    Redact IPv4 and IPv6 addresses.

    Args:
        text: Input text
        replacement: Replacement string

    Returns:
        str: Text with IP addresses redacted
    """
    text = _IPV4_PATTERN.sub(replacement, text)
    text = _IPV6_PATTERN.sub(replacement, text)
    return text


def redact_api_keys(text: str, replacement: str = "[API_KEY]") -> str:
    """
    Redact API keys, tokens, and secrets.

    Handles multiple common patterns for API keys across different services.

    Args:
        text: Input text
        replacement: Replacement string

    Returns:
        str: Text with API keys redacted
    """
    for pattern in _API_KEY_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_urls_with_auth(text: str) -> str:
    """
    Redact credentials in URLs (e.g., https://user:pass@example.com).

    Preserves the URL structure but removes credentials.

    Args:
        text: Input text

    Returns:
        str: Text with URL credentials redacted
    """
    return _URL_WITH_AUTH_PATTERN.sub(r"\1[REDACTED]@\2", text)


def redact_private_keys(text: str, replacement: str = "[PRIVATE_KEY]") -> str:
    """
    Redact private keys (PEM format).

    Args:
        text: Input text
        replacement: Replacement string

    Returns:
        str: Text with private keys redacted
    """
    return _PRIVATE_KEY_PATTERN.sub(replacement, text)


def redact_ssh_keys(text: str, replacement: str = "[SSH_KEY]") -> str:
    """
    Redact SSH public keys.

    Args:
        text: Input text
        replacement: Replacement string

    Returns:
        str: Text with SSH keys redacted
    """
    return _SSH_KEY_PATTERN.sub(replacement, text)


def redact_credit_cards(text: str, replacement: str = "[CREDIT_CARD]") -> str:
    """
    Redact credit card numbers.

    Args:
        text: Input text
        replacement: Replacement string

    Returns:
        str: Text with credit card numbers redacted
    """
    return _CREDIT_CARD_PATTERN.sub(replacement, text)


def redact_ssn(text: str, replacement: str = "[SSN]") -> str:
    """
    Redact Social Security Numbers (US format).

    Args:
        text: Input text
        replacement: Replacement string

    Returns:
        str: Text with SSNs redacted
    """
    return _SSN_PATTERN.sub(replacement, text)


def redact_phone_numbers(text: str, replacement: str = "[PHONE]") -> str:
    """
    Redact phone numbers.

    Args:
        text: Input text
        replacement: Replacement string

    Returns:
        str: Text with phone numbers redacted
    """
    return _PHONE_PATTERN.sub(replacement, text)


def redact_home_paths(text: str, replacement: str = "[USER]") -> str:
    """
    Redact usernames in filesystem paths.

    Examples:
        /home/john/file.txt → /home/[USER]/file.txt
        /Users/alice/Documents → /Users/[USER]/Documents

    Args:
        text: Input text
        replacement: Replacement string for username

    Returns:
        str: Text with usernames in paths redacted
    """

    def replace_username(match: Match[str]) -> str:
        prefix = match.group(0).split("/")[1]  # 'home' or 'Users'
        return f"/{prefix}/{replacement}"

    return _HOME_PATH_PATTERN.sub(replace_username, text)


# --- Advanced NER-based Redaction ---

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig

    _HAS_PRESIDIO = True
except ImportError:
    _HAS_PRESIDIO = False


class AdvancedPIIRedactor:
    """
    NER-based PII redaction using Microsoft Presidio.
    
    Provides context-aware redaction for names, locations, and other entities
    that regex struggles with.
    """

    def __init__(self):
        if not _HAS_PRESIDIO:
            raise ImportError(
                "Presidio not installed. Install 'presidio-analyzer' and 'presidio-anonymizer'."
            )
        
        # Initialize engines (loads NLP model, so do this once)
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        # Default entities to redact via NER
        # NOTE: Presidio's built-in recognizers are US-centric. For comprehensive
        # international PII detection, consider adding custom recognizers for:
        # - International driver's licenses (UK, EU, etc.)
        # - National ID numbers (beyond US SSN)
        # - Tax IDs from other countries
        # See: https://microsoft.github.io/presidio/supported_entities/
        self.default_entities = [
            # Universal entities
            "PERSON",
            "LOCATION",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "IP_ADDRESS",
            "CREDIT_CARD",
            "CRYPTO",
            # US-specific (limited international coverage)
            "US_SSN",
            "US_DRIVER_LICENSE",
            "US_BANK_NUMBER",
            "US_PASSPORT",
            # International (where available in Presidio)
            "IBAN_CODE",              # European banking
            "UK_NHS",                 # UK National Health Service number
            "AU_ABN",                 # Australian Business Number
            "AU_ACN",                 # Australian Company Number
            "AU_TFN",                 # Australian Tax File Number
            "AU_MEDICARE",            # Australian Medicare number
            "IN_PAN",                 # Indian Permanent Account Number
            "IN_AADHAAR",             # Indian Aadhaar number (national ID)
            "IN_VEHICLE_REGISTRATION", # Indian vehicle registration
            "IN_PASSPORT",            # Indian passport
            "SG_NRIC_FIN",            # Singapore National Registration ID
            "ES_NIF",                 # Spanish Tax ID
            "IT_FISCAL_CODE",         # Italian Fiscal Code
            "IT_DRIVER_LICENSE",      # Italian Driver's License
            "IT_VAT_CODE",            # Italian VAT
            "IT_PASSPORT",            # Italian Passport
            "IT_IDENTITY_CARD",       # Italian Identity Card
        ]

    def redact(
        self,
        text: str,
        entities: Optional[list[str]] = None,
        language: str = "en",
    ) -> str:
        """
        Redact PII using NER analysis.

        Args:
            text: Input text
            entities: List of entity types to redact (default: all common PII).
                     Accepted types include universal (PERSON, LOCATION, EMAIL_ADDRESS,
                     PHONE_NUMBER, IP_ADDRESS, CREDIT_CARD, CRYPTO), US-specific
                     (US_SSN, US_DRIVER_LICENSE, US_BANK_NUMBER, US_PASSPORT), and
                     international (IBAN_CODE, UK_NHS, AU_*, IN_*, SG_NRIC_FIN, ES_NIF,
                     IT_* - see self.default_entities for complete list).
            language: Language code (default: "en")

        Returns:
            Redacted text
            
        Raises:
            ValueError: If invalid entity types are provided
        """
        if not text:
            return ""

        # Validate entities against supported set
        if entities:
            invalid_entities = set(entities) - set(self.default_entities)
            if invalid_entities:
                logger.warning(
                    f"Invalid entity types requested: {invalid_entities}. "
                    f"Supported types: {self.default_entities}. Filtering out invalid types."
                )
                # Filter to only valid entities
                entities_to_check = [e for e in entities if e in self.default_entities]
                if not entities_to_check:
                    logger.error("No valid entity types provided, using defaults")
                    entities_to_check = self.default_entities
            else:
                entities_to_check = entities
        else:
            entities_to_check = self.default_entities

        # Analyze text for PII entities
        results = self.analyzer.analyze(
            text=text,
            language=language,
            entities=entities_to_check,
        )

        # Define anonymization operators
        # We use "replace" with the entity type name
        operators = {
            entity: OperatorConfig("replace", {"new_value": f"[{entity}]"})
            for entity in entities_to_check
        }

        # Anonymize
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators,
        )

        return anonymized_result.text


# Global instance (lazy loaded)
_REDACTOR_INSTANCE: Optional[AdvancedPIIRedactor] = None


def get_redactor() -> Optional[AdvancedPIIRedactor]:
    """Get or create global AdvancedPIIRedactor instance."""
    global _REDACTOR_INSTANCE
    if _REDACTOR_INSTANCE is None and _HAS_PRESIDIO:
        try:
            _REDACTOR_INSTANCE = AdvancedPIIRedactor()
        except Exception as e:
            # Fallback if model not downloaded (e.g., en_core_web_lg)
            logger.warning("Failed to initialize Presidio: %s", e, exc_info=True)
            return None
    return _REDACTOR_INSTANCE


def redact_all(
    text: str,
    redact_emails: bool = True,
    redact_ips: bool = True,
    redact_keys: bool = True,
    redact_urls: bool = True,
    redact_private_keys_flag: bool = True,
    redact_ssh: bool = True,
    redact_cards: bool = True,
    redact_ssns: bool = True,
    redact_phones: bool = False,
    redact_paths: bool = True,
    use_ner: bool = True, 
) -> str:
    """
    Apply comprehensive PII redaction.

    Combines fast regex patterns with advanced NER (if enabled and available)
    for maximum security and reliability.

    Strategy:
    1. Apply regex for structure-heavy PII (API keys, SSH keys, paths)
    2. Apply NER for context-heavy PII (Names, Locations) if use_ner is True
    """
    # 1. Regex-based redaction (Fast, deterministic, handles specific formats)
    
    # Always redact API keys/secrets first (highest risk)
    if redact_keys:
        text = redact_api_keys(text)

    if redact_private_keys_flag:
        text = redact_private_keys(text)

    if redact_ssh:
        text = redact_ssh_keys(text)

    if redact_urls:
        text = redact_urls_with_auth(text)

    if redact_paths:
        text = redact_home_paths(text)

    # 2. NER-based redaction (Context-aware)
    if use_ner:
        redactor = get_redactor()
        if redactor:
            # Use NER for Names, Locations, and other entities
            # We skip entities already handled well by regex if we want,
            # but Presidio is often better at avoiding false positives for things like phones.
            # However, for consistency with legacy behavior, we can let Presidio handle
            # the complex ones.
            
            ner_entities = ["PERSON", "LOCATION", "US_DRIVER_LICENSE", "IBAN_CODE"]
            
            # If regex flags are on, let NER handle them too for better coverage
            if redact_emails: ner_entities.append("EMAIL_ADDRESS")
            if redact_ips: ner_entities.append("IP_ADDRESS")
            if redact_cards: ner_entities.append("CREDIT_CARD")
            if redact_ssns: ner_entities.append("US_SSN")
            if redact_phones: ner_entities.append("PHONE_NUMBER")

            text = redactor.redact(text, entities=ner_entities)
            
            # Return early if NER handled the common types
            # But we still might want to run regex for things NER missed or specific formats
            # For now, we'll assume NER + specific Regex (Keys/Paths) is sufficient
            return text

    # 3. Fallback Regex (if NER disabled or failed)
    if redact_emails:
        text = redact_email(text)

    if redact_ips:
        text = redact_ip_addresses(text)

    if redact_cards:
        text = redact_credit_cards(text)

    if redact_ssns:
        text = redact_ssn(text)

    if redact_phones:
        text = redact_phone_numbers(text)

    return text
