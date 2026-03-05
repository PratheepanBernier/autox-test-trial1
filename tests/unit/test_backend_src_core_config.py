import os
import sys
import types
import pytest

from pydantic_settings import BaseSettings
from backend.src.core import config

def test_settings_default_values():
    s = config.Settings()
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
    monkeypatch.setenv("CHUNK_SIZE", "123")
    monkeypatch.setenv("CHUNK_OVERLAP", "45")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.99")
    s = config.Settings()
    assert s.APP_NAME == "Test App"
    assert s.API_V1_STR == "/test/api"
    assert s.GROQ_API_KEY == "gsk_test_key"
    assert s.QA_MODEL == "test-qa-model"
    assert s.VISION_MODEL == "test-vision-model"
    assert s.EMBEDDING_MODEL == "test-embedding-model"
    assert s.CHUNK_SIZE == 123
    assert s.CHUNK_OVERLAP == 45
    assert s.TOP_K == 7
    assert s.SIMILARITY_THRESHOLD == 0.99

def test_settings_partial_env_override(monkeypatch):
    monkeypatch.setenv("APP_NAME", "Partial App")
    s = config.Settings()
    assert s.APP_NAME == "Partial App"
    # All other values should be defaults
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
        config.Settings()

def test_settings_invalid_float_env(monkeypatch):
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "not_a_float")
    with pytest.raises(ValueError):
        config.Settings()

def test_settings_case_sensitive(monkeypatch):
    # Should not pick up lower-case env var due to case_sensitive=True
    monkeypatch.setenv("app_name", "lowercase app")
    s = config.Settings()
    assert s.APP_NAME == "Logistics Document Intelligence Assistant"

def test_settings_extra_env_ignored(monkeypatch):
    monkeypatch.setenv("EXTRA_SETTING", "should_be_ignored")
    s = config.Settings()
    assert not hasattr(s, "EXTRA_SETTING")

def test_settings_config_class_attributes():
    assert config.Settings.Config.case_sensitive is True
    assert config.Settings.Config.env_file == ".env"
    assert config.Settings.Config.extra == "ignore"

def test_settings_instance_is_singleton_like():
    # The module-level settings instance should be of type Settings and have default values
    s = config.settings
    assert isinstance(s, config.Settings)
    assert s.APP_NAME == "Logistics Document Intelligence Assistant"
    assert s.API_V1_STR == "/api/v1"

def test_settings_repr_and_str():
    s = config.Settings()
    r = repr(s)
    st = str(s)
    assert "Settings" in r
    assert "Settings" in st
    assert "APP_NAME" in r
    assert "APP_NAME" in st
    assert "Logistics Document Intelligence Assistant" in r
    assert "Logistics Document Intelligence Assistant" in st
