# source_hash: 19f86d3713912848
import pytest
from unittest import mock
from pydantic_settings import BaseSettings
from backend.src.core.config import Settings

def test_settings_default_values_are_set_correctly():
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
    monkeypatch.setenv("API_V1_STR", "/test/api")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    monkeypatch.setenv("QA_MODEL", "test-qa-model")
    monkeypatch.setenv("VISION_MODEL", "test-vision-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setenv("CHUNK_SIZE", "1234")
    monkeypatch.setenv("CHUNK_OVERLAP", "321")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.99")
    s = Settings()
    assert s.APP_NAME == "Test App"
    assert s.API_V1_STR == "/test/api"
    assert s.GROQ_API_KEY == "gsk_test_key"
    assert s.QA_MODEL == "test-qa-model"
    assert s.VISION_MODEL == "test-vision-model"
    assert s.EMBEDDING_MODEL == "test-embedding-model"
    assert s.CHUNK_SIZE == 1234
    assert s.CHUNK_OVERLAP == 321
    assert s.TOP_K == 7
    assert s.SIMILARITY_THRESHOLD == 0.99

def test_settings_type_casting_from_env(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "42")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")
    monkeypatch.setenv("TOP_K", "1")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.0")
    s = Settings()
    assert isinstance(s.CHUNK_SIZE, int)
    assert s.CHUNK_SIZE == 42
    assert isinstance(s.CHUNK_OVERLAP, int)
    assert s.CHUNK_OVERLAP == 0
    assert isinstance(s.TOP_K, int)
    assert s.TOP_K == 1
    assert isinstance(s.SIMILARITY_THRESHOLD, float)
    assert s.SIMILARITY_THRESHOLD == 0.0

def test_settings_invalid_type_raises_validation_error(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "not_an_int")
    with pytest.raises(Exception) as excinfo:
        Settings()
    assert "CHUNK_SIZE" in str(excinfo.value)

def test_settings_boundary_conditions(monkeypatch):
    # Test zero and negative values for int/float fields
    monkeypatch.setenv("CHUNK_SIZE", "0")
    monkeypatch.setenv("CHUNK_OVERLAP", "-1")
    monkeypatch.setenv("TOP_K", "0")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "-0.1")
    s = Settings()
    assert s.CHUNK_SIZE == 0
    assert s.CHUNK_OVERLAP == -1
    assert s.TOP_K == 0
    assert s.SIMILARITY_THRESHOLD == -0.1

def test_settings_config_class_attributes():
    assert Settings.Config.case_sensitive is True
    assert Settings.Config.env_file == ".env"
    assert Settings.Config.extra == "ignore"

def test_settings_equivalent_paths(monkeypatch):
    # Reconciliation: env var and direct instantiation should yield same value
    monkeypatch.setenv("APP_NAME", "ReconTest")
    s_env = Settings()
    s_direct = Settings(APP_NAME="ReconTest")
    assert s_env.APP_NAME == s_direct.APP_NAME

def test_settings_ignore_extra_fields():
    # Extra fields should be ignored due to extra="ignore"
    s = Settings(**{"APP_NAME": "A", "EXTRA_FIELD": "should be ignored"})
    assert not hasattr(s, "EXTRA_FIELD")
    assert s.APP_NAME == "A"

def test_settings_partial_env(monkeypatch):
    # Only some env vars set, rest should use defaults
    monkeypatch.setenv("APP_NAME", "PartialEnv")
    monkeypatch.setenv("CHUNK_SIZE", "555")
    s = Settings()
    assert s.APP_NAME == "PartialEnv"
    assert s.CHUNK_SIZE == 555
    # Check a default value
    assert s.QA_MODEL == "llama-3.3-70b-versatile"

def test_settings_case_sensitive(monkeypatch):
    # Should not pick up lower-case env var due to case_sensitive=True
    monkeypatch.setenv("app_name", "lowercase")
    s = Settings()
    assert s.APP_NAME == "Logistics Document Intelligence Assistant"
