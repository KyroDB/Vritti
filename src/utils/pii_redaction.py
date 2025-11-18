"""
PII (Personally Identifiable Information) redaction utilities.

Removes sensitive data from error traces, logs, and code diffs before storage.
Uses regex patterns optimized for performance with compilation caching.
"""

import re
from re import Pattern

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

    def replace_username(match):
        prefix = match.group(0).split("/")[1]  # 'home' or 'Users'
        return f"/{prefix}/{replacement}"

    return _HOME_PATH_PATTERN.sub(replace_username, text)


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
    redact_phones: bool = False,  # Can have false positives
    redact_paths: bool = True,
) -> str:
    """
    Apply all PII redaction rules.

    This is the primary function for episodic memory ingestion.
    Designed for performance with compiled regex patterns.

    Args:
        text: Input text (error trace, log, code diff)
        redact_emails: Redact email addresses
        redact_ips: Redact IP addresses
        redact_keys: Redact API keys and tokens
        redact_urls: Redact credentials in URLs
        redact_private_keys_flag: Redact private keys
        redact_ssh: Redact SSH keys
        redact_cards: Redact credit card numbers
        redact_ssns: Redact SSNs
        redact_phones: Redact phone numbers (may have false positives)
        redact_paths: Redact usernames in filesystem paths

    Returns:
        str: Fully redacted text

    Example:
        >>> error = "Error: API key sk-abc123... failed at user@example.com"
        >>> redact_all(error)
        "Error: API key [API_KEY] failed at [EMAIL]"
    """
    if redact_emails:
        text = redact_email(text)

    if redact_ips:
        text = redact_ip_addresses(text)

    if redact_keys:
        text = redact_api_keys(text)

    if redact_urls:
        text = redact_urls_with_auth(text)

    if redact_private_keys_flag:
        text = redact_private_keys(text)

    if redact_ssh:
        text = redact_ssh_keys(text)

    if redact_cards:
        text = redact_credit_cards(text)

    if redact_ssns:
        text = redact_ssn(text)

    if redact_phones:
        text = redact_phone_numbers(text)

    if redact_paths:
        text = redact_home_paths(text)

    return text
