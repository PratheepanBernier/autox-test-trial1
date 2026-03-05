import os
import sys
import types
import pytest

from pydantic_settings import BaseSettings
from backend.src.core.config import Settings, settings

def test_settings_default_values_are_correct():
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

def test_settings_partial_env_override(monkeypatch):
    monkeypatch.setenv("APP_NAME", "Partial App")
    s = Settings()
    assert s.APP_NAME == "Partial App"
    # All others should be default
    assert s.API_V1_STR == "/api/v1"
    assert s.GROQ_API_KEY == "gsk_your_api_key"
    assert s.QA_MODEL == "llama-3.3-70b-versatile"
    assert s.VISION_MODEL == "llama-3.2-11b-vision-preview"
    assert s.EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.CHUNK_SIZE == 1000
    assert s.CHUNK_OVERLAP == 200
    assert s.TOP_K == 4
    assert s.SIMILARITY_THRESHOLD == 0.5

def test_settings_type_coercion(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "42")
    monkeypatch.setenv("CHUNK_OVERLAP", "7")
    monkeypatch.setenv("TOP_K", "2")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.123")
    s = Settings()
    assert isinstance(s.CHUNK_SIZE, int)
    assert s.CHUNK_SIZE == 42
    assert isinstance(s.CHUNK_OVERLAP, int)
    assert s.CHUNK_OVERLAP == 7
    assert isinstance(s.TOP_K, int)
    assert s.TOP_K == 2
    assert isinstance(s.SIMILARITY_THRESHOLD, float)
    assert s.SIMILARITY_THRESHOLD == 0.123

def test_settings_invalid_type_raises(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "not_an_int")
    with pytest.raises(ValueError):
        Settings()

def test_settings_boundary_conditions(monkeypatch):
    # Test zero and negative values for int/float fields
    monkeypatch.setenv("CHUNK_SIZE", "0")
    monkeypatch.setenv("CHUNK_OVERLAP", "-1")
    monkeypatch.setenv("TOP_K", "0")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.0")
    s = Settings()
    assert s.CHUNK_SIZE == 0
    assert s.CHUNK_OVERLAP == -1
    assert s.TOP_K == 0
    assert s.SIMILARITY_THRESHOLD == 0.0

def test_settings_case_sensitive(monkeypatch):
    # Should not pick up lower-case env var due to case_sensitive=True
    monkeypatch.setenv("app_name", "lowercase app")
    s = Settings()
    assert s.APP_NAME == "Logistics Document Intelligence Assistant"

def test_settings_extra_env_ignored(monkeypatch):
    # Should ignore extra env vars not defined in Settings
    monkeypatch.setenv("UNRELATED_ENV", "should_be_ignored")
    s = Settings()
    assert not hasattr(s, "UNRELATED_ENV")

def test_settings_config_class_attributes():
    assert hasattr(Settings.Config, "case_sensitive")
    assert Settings.Config.case_sensitive is True
    assert hasattr(Settings.Config, "env_file")
    assert Settings.Config.env_file == ".env"
    assert hasattr(Settings.Config, "extra")
    assert Settings.Config.extra == "ignore"

def test_settings_singleton_instance_is_settings():
    from backend.src.core import config
    assert isinstance(config.settings, Settings)
    # Should reflect default values
    assert config.settings.APP_NAME == "Logistics Document Intelligence Assistant"
    assert config.settings.API_V1_STR == "/api/v1"
    assert config.settings.GROQ_API_KEY == "gsk_your_api_key"
    assert config.settings.QA_MODEL == "llama-3.3-70b-versatile"
    assert config.settings.VISION_MODEL == "llama-3.2-11b-vision-preview"
    assert config.settings.EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.settings.CHUNK_SIZE == 1000
    assert config.settings.CHUNK_OVERLAP == 200
    assert config.settings.TOP_K == 4
    assert config.settings.SIMILARITY_THRESHOLD == 0.5

def test_settings_equivalent_paths(monkeypatch):
    # Reconciliation: env and direct instantiation should match if env is set
    monkeypatch.setenv("APP_NAME", "Recon App")
    s1 = Settings()
    from backend.src.core import config
    s2 = Settings()
    assert s1.APP_NAME == s2.APP_NAME == "Recon App"
    # settings singleton may have been instantiated before env set, so skip that comparison

def test_settings_repr_and_str():
    s = Settings()
    r = repr(s)
    st = str(s)
    assert "Settings" in r
    assert "Settings" in st
    assert "APP_NAME" in r
    assert "APP_NAME" in st
