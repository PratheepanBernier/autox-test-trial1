import os
import sys
import types
import pytest

from pydantic_settings import BaseSettings
from backend.src.core import config

@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    # Clear relevant environment variables before each test for isolation
    env_vars = [
        "APP_NAME", "API_V1_STR", "GROQ_API_KEY", "QA_MODEL", "VISION_MODEL",
        "EMBEDDING_MODEL", "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "SIMILARITY_THRESHOLD"
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)

def test_settings_defaults_are_correct():
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
    monkeypatch.setenv("API_V1_STR", "/test/api")
    monkeypatch.setenv("GROQ_API_KEY", "test_key")
    monkeypatch.setenv("QA_MODEL", "test-qa-model")
    monkeypatch.setenv("VISION_MODEL", "test-vision-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setenv("CHUNK_SIZE", "123")
    monkeypatch.setenv("CHUNK_OVERLAP", "45")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.99")
    # Act
    s = config.Settings()
    # Assert
    assert s.APP_NAME == "Test App"
    assert s.API_V1_STR == "/test/api"
    assert s.GROQ_API_KEY == "test_key"
    assert s.QA_MODEL == "test-qa-model"
    assert s.VISION_MODEL == "test-vision-model"
    assert s.EMBEDDING_MODEL == "test-embedding-model"
    assert s.CHUNK_SIZE == 123
    assert s.CHUNK_OVERLAP == 45
    assert s.TOP_K == 7
    assert s.SIMILARITY_THRESHOLD == 0.99

@pytest.mark.parametrize(
    "env_var,value,expected_type,expected_value",
    [
        ("CHUNK_SIZE", "0", int, 0),
        ("CHUNK_SIZE", "2147483647", int, 2147483647),  # max 32-bit int
        ("CHUNK_OVERLAP", "0", int, 0),
        ("CHUNK_OVERLAP", "99999", int, 99999),
        ("TOP_K", "1", int, 1),
        ("TOP_K", "100", int, 100),
        ("SIMILARITY_THRESHOLD", "0.0", float, 0.0),
        ("SIMILARITY_THRESHOLD", "1.0", float, 1.0),
    ]
)
def test_settings_boundary_conditions(monkeypatch, env_var, value, expected_type, expected_value):
    # Arrange
    monkeypatch.setenv(env_var, value)
    # Act
    s = config.Settings()
    # Assert
    actual = getattr(s, env_var)
    assert isinstance(actual, expected_type)
    assert actual == expected_value

@pytest.mark.parametrize(
    "env_var,value,expected_exception",
    [
        ("CHUNK_SIZE", "not_an_int", ValueError),
        ("CHUNK_OVERLAP", "NaN", ValueError),
        ("TOP_K", "3.14", ValueError),
        ("SIMILARITY_THRESHOLD", "not_a_float", ValueError),
    ]
)
def test_settings_invalid_env_values(monkeypatch, env_var, value, expected_exception):
    # Arrange
    monkeypatch.setenv(env_var, value)
    # Act & Assert
    with pytest.raises(expected_exception):
        config.Settings()

def test_settings_config_class_attributes():
    # Arrange & Act
    conf = config.Settings.Config
    # Assert
    assert getattr(conf, "case_sensitive", None) is True
    assert getattr(conf, "env_file", None) == ".env"
    assert getattr(conf, "extra", None) == "ignore"

def test_settings_repr_and_eq():
    # Arrange
    s1 = config.Settings()
    s2 = config.Settings()
    # Act & Assert
    assert repr(s1) == repr(s2)
    assert s1 == s2

def test_settings_partial_env_override(monkeypatch):
    # Arrange
    monkeypatch.setenv("APP_NAME", "Partial App")
    # Act
    s = config.Settings()
    # Assert
    assert s.APP_NAME == "Partial App"
    # All others remain default
    assert s.API_V1_STR == "/api/v1"
    assert s.GROQ_API_KEY == "gsk_your_api_key"

def test_settings_extra_env_ignored(monkeypatch):
    # Arrange
    monkeypatch.setenv("UNRELATED_ENV", "should_be_ignored")
    # Act
    s = config.Settings()
    # Assert
    assert not hasattr(s, "UNRELATED_ENV")
    # All defaults remain
    assert s.APP_NAME == "Logistics Document Intelligence Assistant"

def test_settings_case_sensitive(monkeypatch):
    # Arrange
    monkeypatch.setenv("app_name", "lowercase_app")
    # Act
    s = config.Settings()
    # Assert: Should not pick up lowercase env var due to case_sensitive=True
    assert s.APP_NAME == "Logistics Document Intelligence Assistant"

def test_settings_instance_isolation(monkeypatch):
    # Arrange
    s1 = config.Settings()
    monkeypatch.setenv("APP_NAME", "Changed App")
    s2 = config.Settings()
    # Assert
    assert s1.APP_NAME == "Logistics Document Intelligence Assistant"
    assert s2.APP_NAME == "Changed App"

def test_global_settings_instance_is_settings():
    # Arrange & Act
    from backend.src.core.config import settings
    # Assert
    assert isinstance(settings, config.Settings)
    # Check that global instance matches defaults
    assert settings.APP_NAME == "Logistics Document Intelligence Assistant"
    assert settings.API_V1_STR == "/api/v1"
    assert settings.GROQ_API_KEY == "gsk_your_api_key"
    assert settings.QA_MODEL == "llama-3.3-70b-versatile"
    assert settings.VISION_MODEL == "llama-3.2-11b-vision-preview"
    assert settings.EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.CHUNK_SIZE == 1000
    assert settings.CHUNK_OVERLAP == 200
    assert settings.TOP_K == 4
    assert settings.SIMILARITY_THRESHOLD == 0.5
