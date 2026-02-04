"""
Tests for API key hashing and validation storage behavior.
"""

import pytest

from src.models.customer import APIKeyCreate, CustomerCreate
from src.storage.database import CustomerDatabase


@pytest.mark.asyncio
async def test_api_key_validation_uses_scrypt_hash(tmp_path):
    db = CustomerDatabase(db_path=str(tmp_path / "customers.db"))
    await db.initialize()

    await db.create_customer(
        CustomerCreate(
            customer_id="auth-test",
            organization_name="Auth Test",
            email="auth-test@example.com",
        )
    )

    plaintext_key, api_key = await db.create_api_key(
        APIKeyCreate(customer_id="auth-test", name="primary")
    )

    assert api_key.key_hash.startswith("scrypt$")

    validated = await db.validate_api_key(plaintext_key)
    assert validated is not None
    assert validated.customer_id == "auth-test"

    await db.close()


@pytest.mark.asyncio
async def test_api_key_validation_rejects_tampered_secret(tmp_path):
    db = CustomerDatabase(db_path=str(tmp_path / "customers.db"))
    await db.initialize()

    await db.create_customer(
        CustomerCreate(
            customer_id="auth-test-2",
            organization_name="Auth Test 2",
            email="auth-test-2@example.com",
        )
    )

    plaintext_key, _ = await db.create_api_key(
        APIKeyCreate(customer_id="auth-test-2", name="primary")
    )
    key_payload = plaintext_key.removeprefix("em_live_")
    key_id, secret = key_payload.split("_", maxsplit=1)
    replacement = "0" if secret[-1] != "0" else "1"
    tampered = f"em_live_{key_id}_{secret[:-1]}{replacement}"

    validated = await db.validate_api_key(tampered)
    assert validated is None

    await db.close()


@pytest.mark.asyncio
async def test_api_key_validation_rejects_legacy_format(tmp_path):
    db = CustomerDatabase(db_path=str(tmp_path / "customers.db"))
    await db.initialize()

    await db.create_customer(
        CustomerCreate(
            customer_id="auth-test-3",
            organization_name="Auth Test 3",
            email="auth-test-3@example.com",
        )
    )

    legacy_key = "em_live_" + ("a" * 64)
    validated = await db.validate_api_key(legacy_key)
    assert validated is None

    await db.close()
