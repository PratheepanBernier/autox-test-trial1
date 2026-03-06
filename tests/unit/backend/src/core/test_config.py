import os
import sys
import types
import pytest

from pydantic_settings import BaseSettings
from backend.src.core.config import Settings

@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    # Clear relevant environment variables before each test
    for var in [
        "APP_NAME", "API_V1_STR", "GROQ_API_KEY", "QA_MODEL", "VISION_MODEL",
        "EMBEDDING_MODEL", "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "SIMILARITY_THRESHOLD"
    ]:
        monkeypatch.delenv(var, raising=False)

def test_settings_default_values():
    s = Settings()
    assert s.APP_NAME == "Logistics Document Intelligence Assistant"
    assert s.API_V1_STR == "/api/v1"
    assert s.GROQ_API_KEY == "gsk_your_api_key"
    assert s.QA_MODEL == "llama-3.3-70b-versatile"
    assert s.VISION_MODEL == "llama-3.2-11b-vision-preview"
    assert s.EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.CHUNK_SIZE == 1000
    assert s.CHUNK_OVERLAP == 200
    assert s.TOP_K == 4
    assert s.SIMILARITY_THRESHOLD == 0.5

def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("APP_NAME", "Test App")
    monkeypatch.setenv("API_V1_STR", "/test/v2")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    monkeypatch.setenv("QA_MODEL", "test-qa-model")
    monkeypatch.setenv("VISION_MODEL", "test-vision-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setenv("CHUNK_SIZE", "1234")
    monkeypatch.setenv("CHUNK_OVERLAP", "321")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.75")
    s = Settings()
    assert s.APP_NAME == "Test App"
    assert s.API_V1_STR == "/test/v2"
    assert s.GROQ_API_KEY == "gsk_test_key"
    assert s.QA_MODEL == "test-qa-model"
    assert s.VISION_MODEL == "test-vision-model"
    assert s.EMBEDDING_MODEL == "test-embedding-model"
    assert s.CHUNK_SIZE == 1234
    assert s.CHUNK_OVERLAP == 321
    assert s.TOP_K == 7
    assert s.SIMILARITY_THRESHOLD == 0.75

def test_settings_partial_env_override(monkeypatch):
    monkeypatch.setenv("APP_NAME", "Partial App")
    s = Settings()
    assert s.APP_NAME == "Partial App"
    # All others should be defaults
    assert s.API_V1_STR == "/api/v1"
    assert s.GROQ_API_KEY == "gsk_your_api_key"
    assert s.QA_MODEL == "llama-3.3-70b-versatile"
    assert s.VISION_MODEL == "llama-3.2-11b-vision-preview"
    assert s.EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.CHUNK_SIZE == 1000
    assert s.CHUNK_OVERLAP == 200
    assert s.TOP_K == 4
    assert s.SIMILARITY_THRESHOLD == 0.5

def test_settings_invalid_int_env(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "not_an_int")
    with pytest.raises(ValueError):
        Settings()

def test_settings_invalid_float_env(monkeypatch):
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "not_a_float")
    with pytest.raises(ValueError):
        Settings()

def test_settings_boundary_values(monkeypatch):
    # Test edge/boundary values for int/float fields
    monkeypatch.setenv("CHUNK_SIZE", "0")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")
    monkeypatch.setenv("TOP_K", "0")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.0")
    s = Settings()
    assert s.CHUNK_SIZE == 0
    assert s.CHUNK_OVERLAP == 0
    assert s.TOP_K == 0
    assert s.SIMILARITY_THRESHOLD == 0.0

    monkeypatch.setenv("CHUNK_SIZE", str(2**31-1))
    monkeypatch.setenv("CHUNK_OVERLAP", str(2**31-1))
    monkeypatch.setenv("TOP_K", str(2**31-1))
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "1.0")
    s2 = Settings()
    assert s2.CHUNK_SIZE == 2**31-1
    assert s2.CHUNK_OVERLAP == 2**31-1
    assert s2.TOP_K == 2**31-1
    assert s2.SIMILARITY_THRESHOLD == 1.0

def test_settings_config_class_attributes():
    # Test that Config class attributes are set as expected
    assert Settings.Config.case_sensitive is True
    assert Settings.Config.env_file == ".env"
    assert Settings.Config.extra == "ignore"

def test_settings_repr_and_eq():
    s1 = Settings()
    s2 = Settings()
    assert repr(s1) == repr(s2)
    assert s1 == s2

def test_settings_with_kwargs():
    s = Settings(
        APP_NAME="Custom Name",
        CHUNK_SIZE=555,
        SIMILARITY_THRESHOLD=0.99
    )
    assert s.APP_NAME == "Custom Name"
    assert s.CHUNK_SIZE == 555
    assert s.SIMILARITY_THRESHOLD == 0.99
    # Unspecified fields should be defaults
    assert s.API_V1_STR == "/api/v1"
    assert s.GROQ_API_KEY == "gsk_your_api_key"
    assert s.QA_MODEL == "llama-3.3-70b-versatile"
    assert s.VISION_MODEL == "llama-3.2-11b-vision-preview"
    assert s.EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.CHUNK_OVERLAP == 200
    assert s.TOP_K == 4

def test_settings_extra_fields_ignored():
    # Extra fields should be ignored due to Config.extra = "ignore"
    s = Settings(APP_NAME="Extra Test", EXTRA_FIELD="should be ignored")
    assert s.APP_NAME == "Extra Test"
    # Should not have attribute 'EXTRA_FIELD'
    assert not hasattr(s, "EXTRA_FIELD")
