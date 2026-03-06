import os
import sys
import types
import pytest

from pydantic_settings import BaseSettings
from backend.src.core import config

@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    # Clear relevant environment variables before each test for isolation
    keys = [
        "APP_NAME", "API_V1_STR", "GROQ_API_KEY", "QA_MODEL", "VISION_MODEL",
        "EMBEDDING_MODEL", "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "SIMILARITY_THRESHOLD"
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)

def test_settings_default_values():
    # Arrange & Act
    s = config.Settings()
    # Assert
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
    # Arrange
    monkeypatch.setenv("APP_NAME", "Test App")
    monkeypatch.setenv("API_V1_STR", "/test/v2")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    monkeypatch.setenv("QA_MODEL", "test-qa-model")
    monkeypatch.setenv("VISION_MODEL", "test-vision-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setenv("CHUNK_SIZE", "1234")
    monkeypatch.setenv("CHUNK_OVERLAP", "321")
    monkeypatch.setenv("TOP_K", "9")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.99")
    # Act
    s = config.Settings()
    # Assert
    assert s.APP_NAME == "Test App"
    assert s.API_V1_STR == "/test/v2"
    assert s.GROQ_API_KEY == "gsk_test_key"
    assert s.QA_MODEL == "test-qa-model"
    assert s.VISION_MODEL == "test-vision-model"
    assert s.EMBEDDING_MODEL == "test-embedding-model"
    assert s.CHUNK_SIZE == 1234
    assert s.CHUNK_OVERLAP == 321
    assert s.TOP_K == 9
    assert s.SIMILARITY_THRESHOLD == 0.99

def test_settings_env_invalid_types(monkeypatch):
    # Arrange
    monkeypatch.setenv("CHUNK_SIZE", "not_an_int")
    monkeypatch.setenv("CHUNK_OVERLAP", "not_an_int")
    monkeypatch.setenv("TOP_K", "not_an_int")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "not_a_float")
    # Act & Assert
    with pytest.raises(ValueError):
        config.Settings()

def test_settings_env_boundary_values(monkeypatch):
    # Arrange
    monkeypatch.setenv("CHUNK_SIZE", "0")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")
    monkeypatch.setenv("TOP_K", "0")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.0")
    # Act
    s = config.Settings()
    # Assert
    assert s.CHUNK_SIZE == 0
    assert s.CHUNK_OVERLAP == 0
    assert s.TOP_K == 0
    assert s.SIMILARITY_THRESHOLD == 0.0

def test_settings_env_negative_values(monkeypatch):
    # Arrange
    monkeypatch.setenv("CHUNK_SIZE", "-1")
    monkeypatch.setenv("CHUNK_OVERLAP", "-10")
    monkeypatch.setenv("TOP_K", "-5")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "-0.1")
    # Act
    s = config.Settings()
    # Assert
    assert s.CHUNK_SIZE == -1
    assert s.CHUNK_OVERLAP == -10
    assert s.TOP_K == -5
    assert s.SIMILARITY_THRESHOLD == -0.1

def test_settings_case_sensitive(monkeypatch):
    # Arrange
    # Lowercase env var should not override due to case_sensitive=True
    monkeypatch.setenv("app_name", "lowercase app")
    # Act
    s = config.Settings()
    # Assert
    assert s.APP_NAME == "Logistics Document Intelligence Assistant"

def test_settings_extra_env_ignored(monkeypatch):
    # Arrange
    monkeypatch.setenv("UNRELATED_ENV", "should_be_ignored")
    # Act
    s = config.Settings()
    # Assert
    assert not hasattr(s, "UNRELATED_ENV")

def test_settings_repr_and_str():
    # Arrange
    s = config.Settings()
    # Act
    r = repr(s)
    st = str(s)
    # Assert
    assert "Settings" in r
    assert "Settings" in st

def test_settings_config_class_attributes():
    # Arrange & Act
    conf = config.Settings.Config
    # Assert
    assert getattr(conf, "case_sensitive", None) is True
    assert getattr(conf, "env_file", None) == ".env"
    assert getattr(conf, "extra", None) == "ignore"

def test_settings_instance_is_singleton_like():
    # Arrange & Act
    s1 = config.Settings()
    s2 = config.Settings()
    # Assert
    assert s1.dict() == s2.dict()

def test_settings_global_instance_consistency():
    # Arrange & Act
    s = config.settings
    s2 = config.Settings()
    # Assert
    assert s.dict() == s2.dict()
    assert isinstance(s, config.Settings)
