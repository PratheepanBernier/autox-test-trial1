import pytest
from unittest import mock
from pydantic_settings import BaseSettings
from backend.src.core.config import Settings, settings

def test_settings_default_values_are_set_correctly():
    # Happy path: All defaults are as expected
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
    # Happy path: Environment variables override defaults
    monkeypatch.setenv("APP_NAME", "Test App")
    monkeypatch.setenv("API_V1_STR", "/test/api")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    monkeypatch.setenv("QA_MODEL", "test-qa-model")
    monkeypatch.setenv("VISION_MODEL", "test-vision-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setenv("CHUNK_SIZE", "1234")
    monkeypatch.setenv("CHUNK_OVERLAP", "321")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.9")
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
    assert s.SIMILARITY_THRESHOLD == 0.9

def test_settings_invalid_chunk_size_type(monkeypatch):
    # Error handling: Invalid type for CHUNK_SIZE
    monkeypatch.setenv("CHUNK_SIZE", "not_an_int")
    with pytest.raises(ValueError):
        Settings()

def test_settings_invalid_similarity_threshold_type(monkeypatch):
    # Error handling: Invalid type for SIMILARITY_THRESHOLD
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "not_a_float")
    with pytest.raises(ValueError):
        Settings()

def test_settings_boundary_chunk_size(monkeypatch):
    # Boundary: CHUNK_SIZE at 0 and negative
    monkeypatch.setenv("CHUNK_SIZE", "0")
    s = Settings()
    assert s.CHUNK_SIZE == 0
    monkeypatch.setenv("CHUNK_SIZE", "-1")
    s = Settings()
    assert s.CHUNK_SIZE == -1

def test_settings_boundary_similarity_threshold(monkeypatch):
    # Boundary: SIMILARITY_THRESHOLD at 0.0 and 1.0
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.0")
    s = Settings()
    assert s.SIMILARITY_THRESHOLD == 0.0
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "1.0")
    s = Settings()
    assert s.SIMILARITY_THRESHOLD == 1.0

def test_settings_extra_env_vars_are_ignored(monkeypatch):
    # Edge: Extra env vars not in Settings are ignored
    monkeypatch.setenv("UNRELATED_ENV_VAR", "should_be_ignored")
    s = Settings()
    assert not hasattr(s, "UNRELATED_ENV_VAR")

def test_settings_case_sensitive(monkeypatch):
    # Edge: Case sensitivity in env vars
    monkeypatch.setenv("app_name", "lowercase_app")
    s = Settings()
    # Should not override because of case sensitivity
    assert s.APP_NAME == "Logistics Document Intelligence Assistant"

def test_settings_config_class_attributes():
    # Regression: Config class attributes are preserved
    assert Settings.Config.case_sensitive is True
    assert Settings.Config.env_file == ".env"
    assert Settings.Config.extra == "ignore"

def test_settings_instance_isolation(monkeypatch):
    # Isolation: Each instance is independent
    monkeypatch.setenv("APP_NAME", "Instance1")
    s1 = Settings()
    monkeypatch.setenv("APP_NAME", "Instance2")
    s2 = Settings()
    assert s1.APP_NAME == "Instance1"
    assert s2.APP_NAME == "Instance2"

def test_settings_global_instance_matches_direct_instance(monkeypatch):
    # Reconciliation: settings global instance matches direct instantiation
    monkeypatch.setenv("APP_NAME", "ReconTest")
    s_direct = Settings()
    # Re-import to force reload if needed
    from backend.src.core import config
    assert config.settings.APP_NAME == s_direct.APP_NAME
    assert config.settings.dict() == s_direct.dict()
