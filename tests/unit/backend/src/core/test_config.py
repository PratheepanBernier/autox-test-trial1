import os
import pytest
from unittest import mock
from backend.src.core.config import Settings

@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    # Ensure environment variables do not leak between tests
    keys = [
        "APP_NAME", "API_V1_STR", "GROQ_API_KEY", "QA_MODEL", "VISION_MODEL",
        "EMBEDDING_MODEL", "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "SIMILARITY_THRESHOLD"
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    yield

def test_settings_default_values():
    # Arrange & Act
    s = Settings()
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
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    monkeypatch.setenv("QA_MODEL", "test-qa-model")
    monkeypatch.setenv("VISION_MODEL", "test-vision-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setenv("CHUNK_SIZE", "1234")
    monkeypatch.setenv("CHUNK_OVERLAP", "321")
    monkeypatch.setenv("TOP_K", "10")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.99")
    # Act
    s = Settings()
    # Assert
    assert s.APP_NAME == "Test App"
    assert s.API_V1_STR == "/test/api"
    assert s.GROQ_API_KEY == "gsk_test_key"
    assert s.QA_MODEL == "test-qa-model"
    assert s.VISION_MODEL == "test-vision-model"
    assert s.EMBEDDING_MODEL == "test-embedding-model"
    assert s.CHUNK_SIZE == 1234
    assert s.CHUNK_OVERLAP == 321
    assert s.TOP_K == 10
    assert s.SIMILARITY_THRESHOLD == 0.99

def test_settings_partial_env_override(monkeypatch):
    # Arrange
    monkeypatch.setenv("APP_NAME", "Partial App")
    monkeypatch.setenv("CHUNK_SIZE", "2048")
    # Act
    s = Settings()
    # Assert
    assert s.APP_NAME == "Partial App"
    assert s.CHUNK_SIZE == 2048
    # Defaults for others
    assert s.API_V1_STR == "/api/v1"
    assert s.GROQ_API_KEY == "gsk_your_api_key"
    assert s.QA_MODEL == "llama-3.3-70b-versatile"
    assert s.VISION_MODEL == "llama-3.2-11b-vision-preview"
    assert s.EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.CHUNK_OVERLAP == 200
    assert s.TOP_K == 4
    assert s.SIMILARITY_THRESHOLD == 0.5

def test_settings_invalid_int_env(monkeypatch):
    # Arrange
    monkeypatch.setenv("CHUNK_SIZE", "not_an_int")
    # Act & Assert
    with pytest.raises(ValueError):
        Settings()

def test_settings_invalid_float_env(monkeypatch):
    # Arrange
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "not_a_float")
    # Act & Assert
    with pytest.raises(ValueError):
        Settings()

def test_settings_boundary_values(monkeypatch):
    # Arrange
    monkeypatch.setenv("CHUNK_SIZE", "0")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")
    monkeypatch.setenv("TOP_K", "0")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.0")
    # Act
    s = Settings()
    # Assert
    assert s.CHUNK_SIZE == 0
    assert s.CHUNK_OVERLAP == 0
    assert s.TOP_K == 0
    assert s.SIMILARITY_THRESHOLD == 0.0

def test_settings_negative_values(monkeypatch):
    # Arrange
    monkeypatch.setenv("CHUNK_SIZE", "-1")
    monkeypatch.setenv("CHUNK_OVERLAP", "-10")
    monkeypatch.setenv("TOP_K", "-5")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "-0.1")
    # Act
    s = Settings()
    # Assert
    assert s.CHUNK_SIZE == -1
    assert s.CHUNK_OVERLAP == -10
    assert s.TOP_K == -5
    assert s.SIMILARITY_THRESHOLD == -0.1

def test_settings_config_class_attributes():
    # Arrange & Act
    s = Settings()
    # Assert
    assert hasattr(s.__config__, "case_sensitive")
    assert s.__config__.case_sensitive is True
    assert hasattr(s.__config__, "env_file")
    assert s.__config__.env_file == ".env"
    assert hasattr(s.__config__, "extra")
    assert s.__config__.extra == "ignore"

def test_settings_repr_and_str():
    # Arrange
    s = Settings()
    # Act
    r = repr(s)
    st = str(s)
    # Assert
    assert "Settings" in r
    assert "Settings" in st
    assert "APP_NAME" in r
    assert "APP_NAME" in st

def test_settings_equivalent_paths(monkeypatch):
    # Arrange
    monkeypatch.setenv("QA_MODEL", "llama-3.3-70b-versatile")
    # Act
    s_env = Settings()
    s_default = Settings(QA_MODEL="llama-3.3-70b-versatile")
    # Assert
    assert s_env.QA_MODEL == s_default.QA_MODEL
    assert s_env.dict() == s_default.dict()
